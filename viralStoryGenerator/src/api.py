# viralStoryGenerator/src/api.py
"""
HTTP API backend for ViralStoryGenerator.
"""
import hmac
import json
import uuid
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import redis
import time
import tempfile

import uvicorn

from viralStoryGenerator.models import (
    StoryGenerationRequest,
    JobResponse,
    HealthResponse,
    JobStatusResponse
)

from viralStoryGenerator.utils.health_check import get_service_status
from viralStoryGenerator.src.api_worker import process_story_request as process_story_generation
from viralStoryGenerator.utils.redis_manager import RedisManager as RedisQueueManager
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.security import (
    is_safe_filename,
    is_file_in_directory,
    is_valid_uuid,
    is_valid_voice_id,
    sanitize_input,
    validate_path_component
)
from viralStoryGenerator.src.api_handlers import (
    create_story_task,
    get_task_status
)


app_start_time = time.time()
router = APIRouter()

# Initialize FastAPI app
app = FastAPI(
    title=app_config.APP_TITLE,
    description=app_config.APP_DESCRIPTION,
    version=app_config.VERSION,
    # openapi_url=f"{app_config.http.API_PREFIX}/openapi.json",
    # docs_url=f"{app_config.http.API_PREFIX}/docs",
    # redoc_url=f"{app_config.http.API_PREFIX}/redoc"
)


API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Authentication dependency
async def get_api_key(request: Request, api_key: str = Depends(api_key_header)):
    # Skip authentication if API key security is disabled
    if not app_config.http.API_KEY_ENABLED:
        _logger.warning("API key security is disabled.")
        return None

    if not api_key:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required",
        )

    if not app_config.http.API_KEY or not hmac.compare_digest(api_key, app_config.http.API_KEY):
        _logger.warning(f"Invalid API Key received from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    _logger.debug(f"Valid API Key received from {request.client.host}")
    return api_key

# router = APIRouter(dependencies=[Depends(get_api_key)]) # Example global application

# --- Middleware ---

# Add CORS middleware
if "*" in app_config.http.CORS_ORIGINS and app_config.ENVIRONMENT == "production":
    _logger.critical("CRITICAL SECURITY WARNING: CORS_ORIGINS allows '*' in production environment!")
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.http.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"], # Consider restricting headers
)

# Middleware for basic security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    # response.headers["Content-Security-Policy"] = "default-src 'self'; object-src 'none';" # CSP is complex, enable carefully
    if request.url.scheme == "https":
         response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Active requests')
QUEUE_SIZE = Gauge('api_queue_size', 'API queue size')
RATE_LIMIT_HIT = Counter('api_rate_limit_hit_total', 'Rate limit exceeded count', ['client_ip', 'endpoint'])

# Initialize Redis client for rate limiting
redis_client = None
if app_config.http.RATE_LIMIT_ENABLED and app_config.redis.ENABLED:
    try:
        redis_client = redis.Redis(
            host=app_config.redis.HOST,
            port=app_config.redis.PORT,
            db=app_config.redis.DB,
            password=app_config.redis.PASSWORD,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2
        )
        redis_client.ping()  # Test connection
        _logger.info("Redis rate limiting enabled and connected")
    except Exception as e:
        _logger.warning(f"Failed to connect to Redis for rate limiting: {e}. Using local memory rate limiting fallback.")
        redis_client = None

class RateLimiter:
    """Rate limiter using sliding window algorithm"""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache = {}  # Fallback in-memory storage if Redis is unavailable

    async def check_rate_limit(self, client_ip: str, endpoint: str) -> Tuple[bool, int, int, int]:
        """
        Check if the client has exceeded rate limits

        Args:
            client_ip: IP address of the client
            endpoint: API endpoint being accessed

        Returns:
            Tuple of (is_allowed, current_count, limit, retry_after_seconds)
        """
        window = app_config.http.RATE_LIMIT_WINDOW
        limit = app_config.http.RATE_LIMIT_REQUESTS

        # Create a unique key for each client IP and endpoint combination
        rate_key = f"rate_limit:{client_ip}:{endpoint}"
        current_time = time.time()
        retry_after = 0

        # If Redis is available, use it for distributed rate limiting
        if self.redis:
            try:
                if self.redis.ping(): # Check connection before using
                    # Use Redis sorted set for sliding window
                    pipeline = self.redis.pipeline()
                    # Remove entries outside the current window
                    pipeline.zremrangebyscore(rate_key, 0, current_time - window)
                    # Add current request timestamp
                    pipeline.zadd(rate_key, {str(uuid.uuid4()): current_time}) # Use unique member for each request
                    # Count requests in the window
                    pipeline.zcount(rate_key, current_time - window, current_time)
                    # Set expiry to avoid leaking memory
                    pipeline.expire(rate_key, window * 2)
                    # Get the timestamp of the oldest request in the window if over limit
                    pipeline.zrange(rate_key, -limit, -limit, withscores=True) # Get the Nth oldest request (if N=limit, this is the one that caused overflow)

                    results = pipeline.execute()
                    request_count = results[2] # zcount result

                    is_allowed = request_count <= limit

                    if not is_allowed:
                        # Calculate Retry-After based on the oldest relevant timestamp
                        oldest_relevant_req = results[4]
                        if oldest_relevant_req:
                             oldest_ts = oldest_relevant_req[0][1]
                             retry_after = int(window - (current_time - oldest_ts) + 1) # Time until the oldest request expires + 1s buffer

                    return is_allowed, request_count, limit, max(0, retry_after)
                else:
                    _logger.warning("Redis connection lost for rate limiting. Falling back to local.")
                    self.redis = None # Mark Redis as unavailable
            except Exception as e:
                _logger.error(f"Redis error during rate limiting: {e}. Falling back to local.")
                self.redis = None # Mark Redis as unavailable

        # Fallback to local memory if Redis is unavailable
        if rate_key not in self.local_cache:
            self.local_cache[rate_key] = []

        # Remove expired timestamps
        self.local_cache[rate_key] = [ts for ts in self.local_cache[rate_key] if ts > current_time - window]

        request_count = len(self.local_cache[rate_key])
        is_allowed = request_count < limit # Allow *up to* the limit

        if is_allowed:
            # Add current request only if allowed
            self.local_cache[rate_key].append(current_time)
            request_count += 1 # Increment count after adding
            retry_after = 0
        else:
             # Calculate retry_after for local cache
             if self.local_cache[rate_key]:
                 oldest_ts = self.local_cache[rate_key][0]
                 retry_after = int(window - (current_time - oldest_ts) + 1)


        return is_allowed, request_count, limit, max(0, retry_after)

rate_limiter = RateLimiter(redis_client)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    start_time = time.time()
    response = None # Define response initially

    if app_config.http.RATE_LIMIT_ENABLED:
        client_ip = request.client.host
        endpoint = request.url.path

        # Skip rate limiting for certain endpoints
        if endpoint in ['/health', '/metrics', '/docs', '/openapi.json']:
            response = await call_next(request)
            return response

        is_allowed, current, limit, retry_after = await rate_limiter.check_rate_limit(client_ip, endpoint)

        # If request exceeds limit, return 429 Too Many Requests
        if not is_allowed:
            RATE_LIMIT_HIT.labels(client_ip=client_ip, endpoint=endpoint).inc()
            _logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}: {current}/{limit}")

            # Custom rate limit exceeded response
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Maximum {limit} requests per {app_config.http.RATE_LIMIT_WINDOW} second window."
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                    "Retry-After": str(retry_after)
                }
            )
            return response
        else:
             response = await call_next(request)
             response.headers["X-RateLimit-Limit"] = str(limit)
             response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current))
             # X-RateLimit-Reset might be complex to calculate accurately, omitting for simplicity or use retry_after logic if needed
    else:
        # If rate limiting is disabled, just proceed
        response = await call_next(request)

    return response

# Middleware for metrics and request logging
@app.middleware("http")
async def metrics_logging_middleware(request: Request, call_next):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    response = None
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        _logger.exception(f"Unhandled exception during request: {request.method} {request.url.path}") # Log exception details
        status_code = 500
        # Re-raise the exception to be caught by the global handler or FastAPI's default
        raise e from None
    finally:
        process_time = time.time() - start_time
        ACTIVE_REQUESTS.dec()
        endpoint_path = request.url.path
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint_path).observe(process_time)
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint_path, status=status_code).inc()
        _logger.info(f"{request.client.host} - \"{request.method} {request.url.path} HTTP/{request.scope['http_version']}\" {status_code} {process_time:.4f}s")

    return response


# Redis queue manager dependency
def get_queue_manager() -> RedisQueueManager:
    # Check if Redis is enabled globally
    if not app_config.redis.ENABLED:
         _logger.error("Attempted to use Queue Manager, but Redis is disabled in config.")
         raise HTTPException(status_code=503, detail="Queue service is disabled")

    try:
        # Consider creating one instance and reusing, or managing connections carefully
        manager = RedisQueueManager(
            queue_name=API_QUEUE_NAME,
            result_prefix=RESULT_PREFIX
        )
        if not manager.is_available():
             raise redis.exceptions.ConnectionError("Redis connection check failed in get_queue_manager")

        try:
            QUEUE_SIZE.set(manager.get_queue_length())
        except Exception as metric_e:
            _logger.warning(f"Could not update QUEUE_SIZE metric: {metric_e}")
            pass

        return manager
    except redis.exceptions.ConnectionError as e:
         _logger.error(f"Failed to connect to Redis queue: {str(e)}")
         raise HTTPException(status_code=503, detail="Redis queue service unavailable") # 503 Service Unavailable
    except Exception as e:
         _logger.error(f"Failed to initialize Redis queue manager: {str(e)}")
         raise HTTPException(status_code=500, detail="Internal server error initializing queue")


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _logger.exception(f"Unhandled exception caught by global handler for {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error",
                 "detail": "An unexpected error occurred. Please contact support if the issue persists."},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    _logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.method} {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


# --- API Endpoints ---

# Health and Metrics Endpoints
@router.get("/metrics", tags=["Health and Metrics"])
async def get_metrics():
    """
    Endpoint to expose Prometheus metrics.
    This is used by Prometheus to scrape metrics from the application.
    """
    _logger.debug("Metrics endpoint called.")
    return Response(content=generate_latest(), media_type="text/plain")

@router.get("/health", response_model=HealthResponse, tags=["Health and Metrics"])
async def health_check():
    """Health check endpoint for monitoring"""
    _logger.debug("Health check endpoint called.")
    # Fetch detailed service statuses
    service_status = await get_service_status()
    _logger.debug("Health check response generated.")
    return service_status

# Story Management Endpoints
@router.post("/stories", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED, tags=["Story Management"], dependencies=[Depends(get_api_key)])
async def create_new_story_task(
    topic: str,
    background_tasks: BackgroundTasks,
    sources_folder: Optional[str] = None,
    voice_id: Optional[str] = None
):
    """
    Create a new story generation task using the revised api_handlers structure.

    This endpoint now primarily validates input and queues the task.
    """
    _logger.info(f"Received request to create story task for topic: '{topic}'")

    # Validate topic length and content
    sanitized_topic = sanitize_input(topic, max_length=500)
    if not sanitized_topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty or contain only invalid characters.")

    # Validate sources_folder
    validated_sources_folder = None
    if sources_folder:
        if not validate_path_component(sources_folder):
             _logger.warning(f"Security: Invalid characters or structure in sources_folder: {sources_folder}")
             raise HTTPException(status_code=400, detail="Invalid sources folder name.")

        # Construct full path and verify it's within the allowed base directory
        base_sources_path = os.path.abspath(app_config.security.SOURCE_MATERIALS_PATH)
        full_path = os.path.abspath(os.path.join(base_sources_path, sources_folder))

        if not os.path.exists(full_path) or not os.path.isdir(full_path):
             _logger.warning(f"Sources folder not found: {full_path}")
             raise HTTPException(status_code=404, detail=f"Sources folder '{sources_folder}' not found.")

        if not is_file_in_directory(full_path, base_sources_path):
             _logger.critical(f"SECURITY BREACH ATTEMPT: Access denied for sources_folder outside allowed path: {full_path}")
             raise HTTPException(status_code=403, detail="Access to the specified folder is forbidden.")
        validated_sources_folder = sources_folder

    # Validate voice_id format
    if voice_id and not is_valid_voice_id(voice_id):
        _logger.warning(f"Invalid voice ID format received: {voice_id}")
        raise HTTPException(status_code=400, detail="Invalid voice ID format.")

    # Create and queue the task using the handler function
    try:
        task_info = create_story_task(
             topic=sanitized_topic,
             sources_folder=validated_sources_folder,
             voice_id=voice_id
        )
        _logger.info(f"Story task {task_info['task_id']} created/queued for topic: '{sanitized_topic}'")
        # Return 202 Accepted status code
        return JSONResponse(content=task_info, status_code=status.HTTP_202_ACCEPTED)
    except Exception as e:
        _logger.exception(f"Failed to create story task for topic '{sanitized_topic}': {e}")
        raise HTTPException(status_code=500, detail="Failed to create story generation task.")


@router.get("/stories/{task_id}", response_model=JobStatusResponse, tags=["Story Management"], dependencies=[Depends(get_api_key)])
async def check_story_status(task_id: str):
    """Check the status of a story generation task using api_handlers"""
    _logger.debug(f"Checking status for task_id: {task_id}")
    if not is_valid_uuid(task_id):
        _logger.warning(f"Invalid task ID format requested: {task_id}")
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    try:
        task_info = get_task_status(task_id)
        if task_info is None or task_info.get("status") == "not_found":
             _logger.warning(f"Task ID not found: {task_id}")
             raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

        _logger.debug(f"Status for task_id {task_id}: {task_info.get('status')}")
        return JobStatusResponse(**task_info)
    except HTTPException:
         raise
    except Exception as e:
         _logger.exception(f"Error retrieving status for task {task_id}: {e}")
         raise HTTPException(status_code=500, detail="Failed to retrieve task status.")


@router.get("/stories/{task_id}/download/{file_type}", tags=["Story Management"], dependencies=[Depends(get_api_key)])
async def download_story_file(task_id: str, file_type: str):
    """Download a generated file from a completed story task"""
    _logger.debug(f"Request to download file type '{file_type}' for task_id: {task_id}")
    if not is_valid_uuid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    allowed_file_types = ["story", "audio", "storyboard"]
    if file_type not in allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Must be one of: {', '.join(allowed_file_types)}"
        )

    task_info = get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not completed (status: {task_info.get('status')})")

    file_paths = task_info.get("file_paths", {})
    relative_file_path = file_paths.get(file_type)

    if not relative_file_path:
        raise HTTPException(status_code=404, detail=f"No {file_type} file available for task {task_id}")

    # --- Determine absolute path based on storage provider ---
    filename = os.path.basename(relative_file_path) # Get filename
    if not is_safe_filename(filename):
         _logger.warning(f"Security: Unsafe filename retrieved from task data: {filename}")
         raise HTTPException(status_code=500, detail="Internal error: Invalid filename stored for task.")

    # Define base directory based on file type for local checks
    expected_base_directory = None
    if file_type == "audio":
        expected_base_directory = os.path.abspath(app_config.storage.AUDIO_STORAGE_PATH)
        media_type = "audio/mpeg"
    elif file_type == "storyboard":
        expected_base_directory = os.path.abspath(app_config.storage.STORYBOARD_STORAGE_PATH)
        media_type = "application/json"
    else: # story
        expected_base_directory = os.path.abspath(app_config.storage.STORY_STORAGE_PATH)
        media_type = "text/plain"

    if app_config.storage.PROVIDER == "local":
        full_file_path = os.path.abspath(os.path.join(expected_base_directory, filename))

        if not os.path.exists(full_file_path) or not os.path.isfile(full_file_path):
             _logger.error(f"File not found on local disk: {full_file_path}")
             raise HTTPException(status_code=404, detail="File not found on server")

        # Ensure the resolved path is within the expected directory
        if not is_file_in_directory(full_file_path, expected_base_directory):
             _logger.critical(f"SECURITY BREACH ATTEMPT: Access denied for file outside storage directory: {full_file_path}")
             raise HTTPException(status_code=403, detail="Access denied")

        _logger.info(f"Serving local file: {full_file_path}")
        return FileResponse(path=full_file_path, media_type=media_type, filename=filename)
    else:
        # --- Cloud Storage Handling ---
        _logger.info(f"Retrieving file '{filename}' of type '{file_type}' from cloud storage: {app_config.storage.PROVIDER}")
        try:
            file_content = storage_manager.retrieve_file(filename=filename, file_type=file_type)

            if file_content is None:
                _logger.error(f"File '{filename}' not found in cloud storage.")
                raise HTTPException(status_code=404, detail="File not found in storage.")

            if isinstance(file_content, bytes):
                 return Response(content=file_content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})
            elif hasattr(file_content, 'read'): # Check if it's a file-like object/stream
                 # Use StreamingResponse for file-like objects
                 return StreamingResponse(file_content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})
            else:
                 _logger.error(f"Unexpected content type from storage manager: {type(file_content)}")
                 raise HTTPException(status_code=500, detail="Internal error retrieving file from storage.")

        except Exception as e:
            _logger.exception(f"Failed to retrieve file {filename} from cloud storage: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve file from storage.")


# --- File Serving Endpoints---
# These endpoints might be served directly by Nginx in production for efficiency.

@router.get("/audio/{filename}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def serve_audio_file(filename: str):
    """Serve audio files directly (publicly accessible?)"""
    _logger.debug(f"Request to serve audio file: {filename}")
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    try:
        # Serve_file might return a path (local) or data/error (cloud)
        serve_info = storage_manager.serve_file(filename, "audio")

        if isinstance(serve_info, str) and os.path.exists(serve_info):
             # Validate path again for safety
             local_base = os.path.abspath(app_config.storage.AUDIO_STORAGE_PATH)
             if not is_file_in_directory(serve_info, local_base):
                  _logger.critical(f"SECURITY BREACH ATTEMPT: Serve path outside allowed dir: {serve_info}")
                  raise HTTPException(status_code=403, detail="Access denied.")
             return FileResponse(path=serve_info, media_type="audio/mpeg", filename=filename)
        elif isinstance(serve_info, dict) and 'error' in serve_info:
             _logger.error(f"Storage manager error serving audio {filename}: {serve_info['error']}")
             raise HTTPException(status_code=500, detail="Error serving file from storage.")
        elif serve_info:
             _logger.warning(f"Unexpected return type {type(serve_info)} from storage_manager.serve_file for audio.")
             # Attempt basic response if possible, otherwise error
             try:
                 return Response(content=serve_info, media_type="audio/mpeg", headers={"Content-Disposition": f"attachment; filename={filename}"})
             except:
                  raise HTTPException(status_code=500, detail="Internal error serving file.")
        else:
             raise HTTPException(status_code=404, detail="Audio file not found.")

    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="Audio file not found.")
    except Exception as e:
         _logger.exception(f"Error serving audio file {filename}: {e}")
         raise HTTPException(status_code=500, detail="Internal server error serving file.")


@router.get("/api/audio/stream/{filename}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def stream_audio(filename: str, request: Request):
    """Stream audio file with support for range requests (publicly accessible?)"""
    _logger.debug(f"Request to stream audio file: {filename}")
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    range_header = request.headers.get("Range")

    file_size = None
    try:
        # --- Get file size---
        if app_config.storage.PROVIDER == "local":
            local_path = os.path.join(app_config.storage.AUDIO_STORAGE_PATH, filename)
            base_dir = os.path.abspath(app_config.storage.AUDIO_STORAGE_PATH)
            if not os.path.exists(local_path) or not is_file_in_directory(local_path, base_dir):
                 raise FileNotFoundError("File not found or access denied")
            file_size = os.path.getsize(local_path)
        else:
            # Placeholder: retrieve the whole file to get size (inefficient!)
            # TODO: Implement efficient metadata retrieval in StorageManager
            temp_file_info = storage_manager.serve_file(filename, "audio")
            if isinstance(temp_file_info, str) and os.path.exists(temp_file_info):
                file_size = os.path.getsize(temp_file_info)
                # Keep temp_file_info path for streaming later if needed
            else:
                 raise FileNotFoundError("Could not determine file size from storage")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    except Exception as e:
        _logger.exception(f"Error getting file size for streaming {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error accessing file for streaming.")

    # --- Handle Range Request ---
    start = 0
    end = file_size - 1
    status_code = status.HTTP_200_OK

    if range_header:
        try:
            range_val = range_header.strip().lower()
            if not range_val.startswith("bytes="):
                 raise ValueError("Invalid Range header format")
            parts = range_val.split("=")[1].split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1

            if start >= file_size or start < 0 or end < start:
                 raise ValueError("Invalid range values")

            end = min(end, file_size - 1)
            status_code = status.HTTP_206_PARTIAL_CONTENT

        except ValueError as ve:
             _logger.warning(f"Invalid Range header '{range_header}': {ve}")
             # Return 416 Range Not Satisfiable if range is invalid
             return Response(status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE)


    content_length = end - start + 1

    # --- Define Streaming Function ---
    async def file_streamer(start_byte: int, length: int):
        chunk_size = 64 * 1024 # 64KB
        bytes_yielded = 0
        try:
            stream = storage_manager.retrieve_file(
                filename=filename,
                file_type="audio",
                start_byte=start_byte,
                end_byte=start_byte + length - 1
            )

            if stream is None:
                 _logger.error(f"Streaming error: retrieve_file returned None for range {start_byte}-{start_byte+length-1}")
                 raise IOError("Failed to retrieve file stream chunk")

            # Handle different stream types
            if hasattr(stream, 'read'):
                 while bytes_yielded < length:
                      read_size = min(chunk_size, length - bytes_yielded)
                      chunk = stream.read(read_size)
                      if not chunk: break
                      yield chunk
                      bytes_yielded += len(chunk)
                 if hasattr(stream, 'close'): stream.close()

            elif hasattr(stream, 'chunks'): # Azure stream
                 async for chunk in stream.chunks():
                      yield chunk
                      bytes_yielded += len(chunk)
                 # TODO: Refine Azure streaming based on actual 'chunks' behavior with range

            elif isinstance(stream, bytes):
                 yield stream
                 bytes_yielded = len(stream)

            else:
                 _logger.error(f"Streaming error: Unexpected stream type {type(stream)}")
                 raise IOError("Unsupported stream type from storage manager")

            if bytes_yielded < length:
                _logger.warning(f"Streaming ended prematurely for {filename}: yielded {bytes_yielded}, expected {length}")

        except Exception as e:
            _logger.exception(f"Error during audio streaming for {filename}: {e}")
            # Yield nothing further on error
            yield b''


    # --- Prepare and Return Response ---
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/mpeg",
        "Cache-Control": "public, max-age=3600"
    }
    return StreamingResponse(
        file_streamer(start, content_length),
        status_code=status_code,
        headers=headers,
        media_type="audio/mpeg"
    )


@router.get("/story/{filename}", tags=["File Serving"])
async def serve_story_file(filename: str):
    """Serve story text files directly (publicly accessible?)"""
    _logger.debug(f"Request to serve story file: {filename}")
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    try:
        serve_info = storage_manager.serve_file(filename, "story")
        if isinstance(serve_info, str) and os.path.exists(serve_info):
             local_base = os.path.abspath(app_config.storage.STORY_STORAGE_PATH)
             if not is_file_in_directory(serve_info, local_base):
                 raise HTTPException(status_code=403, detail="Access denied.")
             return FileResponse(path=serve_info, media_type="text/plain", filename=filename)
        elif isinstance(serve_info, dict) and 'error' in serve_info:
             raise HTTPException(status_code=500, detail="Error serving file from storage.")
        elif serve_info: # Assume bytes or similar direct content
             return Response(content=serve_info, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={filename}"})
        else:
             raise HTTPException(status_code=404, detail="Story file not found.")
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="Story file not found.")
    except Exception as e:
         _logger.exception(f"Error serving story file {filename}: {e}")
         raise HTTPException(status_code=500, detail="Internal server error serving file.")


@router.get("/storyboard/{filename}", tags=["File Serving"])
async def serve_storyboard_file(filename: str):
    """Serve storyboard JSON files directly (publicly accessible?)"""
    _logger.debug(f"Request to serve storyboard file: {filename}")
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    try:
        serve_info = storage_manager.serve_file(filename, "storyboard")
        if isinstance(serve_info, str) and os.path.exists(serve_info):
             local_base = os.path.abspath(app_config.storage.STORYBOARD_STORAGE_PATH)
             if not is_file_in_directory(serve_info, local_base):
                 raise HTTPException(status_code=403, detail="Access denied.")
             return FileResponse(path=serve_info, media_type="application/json", filename=filename)
        elif isinstance(serve_info, dict) and 'error' in serve_info:
             raise HTTPException(status_code=500, detail="Error serving file from storage.")
        elif serve_info:
             return Response(content=serve_info, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}"})
        else:
             raise HTTPException(status_code=404, detail="Storyboard file not found.")
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="Storyboard file not found.")
    except Exception as e:
         _logger.exception(f"Error serving storyboard file {filename}: {e}")
         raise HTTPException(status_code=500, detail="Internal server error serving file.")


# --- Job Management Endpoints (Using Redis Queue Directly) ---
API_QUEUE_NAME = app_config.redis.QUEUE_NAME
RESULT_PREFIX = app_config.redis.RESULT_PREFIX

@router.post("/generate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Job Management"], dependencies=[Depends(get_api_key)])
async def generate_story_from_urls(
    request: StoryGenerationRequest,
    background_tasks: BackgroundTasks,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Generate a viral story from the provided URLs.
    This endpoint queues the request for processing and returns a job ID.
    Relies on a separate worker process consuming from the Redis queue.
    """
    _logger.info(f"Received request to generate story from URLs for topic: '{request.topic}'")
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Prepare job data matching worker expectations
        job_data = {
            "id": job_id,
            "data": {
                 "urls": [str(url) for url in request.urls],
                 "topic": request.topic,
                 "generate_audio": request.generate_audio,
                 "temperature": request.temperature if request.temperature is not None else app_config.llm.TEMPERATURE,
                 "chunk_size": request.chunk_size if request.chunk_size is not None else app_config.llm.CHUNK_SIZE
            },
             "request_time": time.time()
        }

        # Add request to the queue
        success = queue_manager.add_request(job_data)
        if not success:
            _logger.error(f"Failed to add job {job_id} to Redis queue.")
            raise HTTPException(status_code=500, detail="Failed to queue job for processing.")

        # for testing purposes, we can call the worker directly
        # background_tasks.add_task(process_story_generation, job_id, job_data['data']) # Remove this line

        _logger.info(f"Job {job_id} queued successfully for topic: '{request.topic}'")
        return JobResponse(
            job_id=job_id,
            message="Story generation job queued successfully. A worker will process it."
        )

    except HTTPException:
         raise
    except Exception as e:
        _logger.exception(f"Error queueing job for topic '{request.topic}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error queueing job.")


@router.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Job Management"], dependencies=[Depends(get_api_key)])
async def get_job_status(
    job_id: str,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Get the status of a story generation job queued via /generate endpoint.
    Returns the result if the job has completed.
    """
    _logger.debug(f"Checking status for job_id: {job_id}")
    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    try:
        result_data = queue_manager.get_result(job_id)

        if not result_data:
            if queue_manager.check_key_exists(job_id):
                _logger.debug(f"Job {job_id} found but no final result yet (pending/processing).")
                # Return a pending status conforming to the model
                return JobStatusResponse(status="pending", message="Job is pending or currently processing.")
            else:
                _logger.warning(f"Job ID not found in Redis: {job_id}")
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        _logger.debug(f"Result found for job {job_id}, status: {result_data.get('status')}")
        return JobStatusResponse(**result_data)

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception(f"Error checking job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error checking job status.")


# --- Queue Management Admin Endpoints ---
queue_router = APIRouter(
    prefix="/queue", # Prefix all routes in this router with /queue
    tags=["Queue Management"],
    dependencies=[Depends(get_api_key)] # Apply API key auth to all queue endpoints
)

@queue_router.get("/status")
async def get_queue_status(
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Get the current status of the job queue system.
    Returns metrics like queue length and processing jobs.
    """
    _logger.info("Request received for queue status.")
    try:
        # Get main queue length
        main_queue_length = queue_manager.get_queue_length()

        # Get processing queue length (if applicable, depends on manager impl)
        processing_length = 0
        if hasattr(queue_manager, 'get_processing_queue_length'): # Check if method exists
             processing_length = queue_manager.get_processing_queue_length()
        elif queue_manager.client: # Check Redis client directly if needed
            try:
                 processing_queue_name = f"{queue_manager.queue_name}_processing" # Standard pattern assumption
                 processing_length = queue_manager.client.llen(processing_queue_name)
            except Exception as e:
                 _logger.warning(f"Could not get processing queue length: {e}")


        # Add more stats if needed (e.g., failed jobs count, recent jobs)
        # Example: Get ~10 recent job statuses
        recent_jobs_info = []
        try:
            keys = queue_manager.client.keys(f"{queue_manager.result_prefix}*")
            # Sort keys potentially by time if needed, or just take sample
            sample_keys = keys[:10] # Get up to 10 keys
            for key in sample_keys:
                job_data_str = queue_manager.client.get(key)
                if job_data_str:
                    try:
                        job_info = json.loads(job_data_str)
                        job_id = key.replace(queue_manager.result_prefix, "")
                        recent_jobs_info.append({
                            "job_id": job_id,
                            "status": job_info.get("status", "unknown"),
                            "updated_at": job_info.get("updated_at") # Assuming worker adds this
                        })
                    except json.JSONDecodeError:
                        _logger.warning(f"Could not decode job data for key {key}")
        except Exception as e:
            _logger.warning(f"Could not retrieve recent job info: {e}")


        return {
            "status": "available",
            "main_queue_length": main_queue_length,
            "processing_jobs": processing_length,
            "recent_jobs_sample": recent_jobs_info
            # Add more detailed stats here
        }
    except Exception as e:
        _logger.exception(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@queue_router.post("/clear-stalled")
async def clear_stalled_jobs(
    max_age_seconds: int = 600, # Default: older than 10 minutes
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    EXPERIMENTAL: Clear potentially stalled jobs from the processing queue.
    Moves jobs older than max_age_seconds back to the main queue or marks as failed.
    Requires specific implementation in RedisManager or direct Redis commands.
    """
    _logger.warning(f"Request received to clear stalled jobs older than {max_age_seconds}s.")
    # This functionality is complex and depends heavily on how workers manage the processing state.
    # A robust implementation might involve:
    # 1. Workers heartbeating their progress.
    # 2. Using Redis streams with consumer groups.
    # 3. A separate monitoring process.
    # Providing a generic, safe implementation here is difficult.
    # Placeholder implementation (requires RedisManager support or direct commands):
    cleared_count = 0
    requeued_count = 0
    failed_count = 0
    try:
        # Example using BRPOPLPUSH pattern with timeout tracking (conceptual)
        processing_queue_name = f"{queue_manager.queue_name}_processing"
        all_processing = queue_manager.client.lrange(processing_queue_name, 0, -1)
        current_time = time.time()

        for item_str in all_processing:
            try:
                item_data = json.loads(item_str)
                job_id = item_data.get("id")
                # Need a timestamp when the job was moved to processing
                # Assuming RedisManager adds this when using get_next_request or similar
                processing_start_time = item_data.get("_processing_start_time") # Needs to be added by manager

                if job_id and processing_start_time and (current_time - processing_start_time > max_age_seconds):
                     _logger.warning(f"Job {job_id} appears stalled (processing since {processing_start_time}).")
                     # Remove from processing queue
                     removed = queue_manager.client.lrem(processing_queue_name, 0, item_str)
                     if removed > 0:
                          cleared_count += 1
                          # Option 1: Re-queue
                          # queue_manager.client.lpush(queue_manager.queue_name, item_str)
                          # requeued_count += 1
                          # Option 2: Mark as failed
                          queue_manager.update_task_error(job_id, f"Job stalled and cleared after {max_age_seconds}s")
                          failed_count += 1
            except Exception as item_e:
                 _logger.error(f"Error processing potentially stalled item: {item_str[:100]}... - {item_e}")

        _logger.info(f"Stalled job cleanup finished. Cleared: {cleared_count}, Requeued: {requeued_count}, Marked Failed: {failed_count}")
        return {
            "message": f"Stalled job cleanup attempted. Cleared: {cleared_count}, Requeued: {requeued_count}, Marked Failed: {failed_count}",
            "cleared_count": cleared_count,
            "requeued_count": requeued_count,
            "failed_count": failed_count
        }
    except Exception as e:
        _logger.exception(f"Error clearing stalled jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear stalled jobs: {str(e)}")


@queue_router.delete("/purge")
async def purge_queue(
    confirmation: str,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    PURGE ALL jobs from the queue system (main and processing).
    This is a DESTRUCTIVE operation. Requires confirmation query parameter.
    Does NOT delete job results by default.
    """
    CONFIRMATION_CODE = "CONFIRM_PURGE_ALL_JOBS_SERIOUSLY"
    if confirmation != CONFIRMATION_CODE:
        _logger.warning("Queue purge attempted without correct confirmation code.")
        raise HTTPException(status_code=400, detail=f"Invalid confirmation. Use confirmation='{CONFIRMATION_CODE}' to confirm this destructive action.")

    _logger.critical(f"Executing QUEUE PURGE operation with confirmation '{confirmation}'")
    try:
        # Clear main queue
        main_queue_name = queue_manager.queue_name
        main_removed = queue_manager.client.delete(main_queue_name)

        # Clear processing queue
        processing_queue_name = f"{queue_manager.queue_name}_processing"
        processing_removed = queue_manager.client.delete(processing_queue_name)

        _logger.critical(f"Queue system purged. Main queue ('{main_queue_name}') removed: {main_removed}. Processing queue ('{processing_queue_name}') removed: {processing_removed}.")
        return {
            "message": f"Queue system purged successfully.",
            "main_queue_cleared": bool(main_removed),
            "processing_queue_cleared": bool(processing_removed),
        }
    except Exception as e:
        _logger.exception(f"Error purging queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to purge queue: {str(e)}")


@queue_router.post("/job/{job_id}/retry")
async def retry_failed_job(
    job_id: str,
    background_tasks: BackgroundTasks, # Not used in this case, but kept for consistency
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Retry a FAILED job by re-queuing its original request data.
    """
    _logger.info(f"Request received to retry job_id: {job_id}")
    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    try:
        result_data = queue_manager.get_result(job_id)
        if not result_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found, cannot retry.")

        current_status = result_data.get("status")
        if current_status != "failed":
            raise HTTPException(status_code=400, detail=f"Only 'failed' jobs can be retried. Current status: '{current_status}'")

        # --- Retrieve original request data ---
        original_request_data = result_data.get("original_request_data")
        if not original_request_data:
             # Fallback: Try to reconstruct from available fields if possible (less reliable)
             _logger.warning(f"Original request data not found for job {job_id}. Attempting reconstruction.")
             if not all(k in result_data for k in ["urls", "topic"]): # Example check
                  raise HTTPException(status_code=400, detail="Cannot retry job: missing essential original request data.")
             original_request_data = {
                 "urls": result_data.get("sources", []),
                 "topic": result_data.get("topic", "Unknown Topic"),
                 "generate_audio": result_data.get("audio_url") is not None,
                 # TODO: Add temperature, chunk_size etc.
             }

        # Prepare new job payload for the queue
        new_job_payload = {
            "id": job_id,
            "data": original_request_data,
            "retry_of": job_id, # Add metadata indicating it's a retry
            "request_time": time.time()
        }

        # Re-queue the job
        success = queue_manager.add_request(new_job_payload)
        if not success:
            _logger.error(f"Failed to re-queue job {job_id} for retry.")
            raise HTTPException(status_code=500, detail="Failed to queue job for retry.")

        _logger.info(f"Job {job_id} successfully re-queued for retry.")
        return {
            "message": f"Job {job_id} has been re-queued for processing.",
            "job_id": job_id
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception(f"Error retrying job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error retrying job.")


# Include the routers in the main app
app.include_router(router, prefix="/api")
app.include_router(queue_router, prefix="/api")

# Ensure the directories exist
os.makedirs(app_config.storage.AUDIO_STORAGE_PATH, exist_ok=True)
os.makedirs(app_config.storage.STORY_STORAGE_PATH, exist_ok=True)
os.makedirs(app_config.storage.STORYBOARD_STORAGE_PATH, exist_ok=True)
app.mount("/static", StaticFiles(directory=app_config.storage.LOCAL_STORAGE_PATH), name="static") # Keep if direct static serving is desired alongside API endpoints

# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint providing basic app info and links"""
    return {
        "app_name": app_config.APP_TITLE,
        "version": app_config.VERSION,
        "description": app_config.APP_DESCRIPTION,
        "environment": app_config.ENVIRONMENT,
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

# --- Uvicorn Runner (for direct execution) ---
def start_api_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1, reload: bool = False, log_level: str = "info"):
    """Start the FastAPI server with uvicorn"""
    _logger.info(f"Attempting to start API server on {host}:{port} with {workers} worker(s)")
    _logger.info(f"Log level: {log_level}")
    _logger.info(f"Auto-reload: {'Enabled' if reload else 'Disabled'}")

    uvicorn.run(
        "viralStoryGenerator.src.api:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level.lower(),
        # Add SSL config if enabled
        ssl_keyfile=app_config.http.SSL_KEY_FILE if app_config.http.SSL_ENABLED else None,
        ssl_certfile=app_config.http.SSL_CERT_FILE if app_config.http.SSL_ENABLED else None,
    )

if __name__ == "__main__":
     # This allows running the API directly using `python -m viralStoryGenerator.src.api`
     host = app_config.http.HOST
     port = app_config.http.PORT
     workers = app_config.http.WORKERS
     log_level = app_config.LOG_LEVEL
     # Note: Reload is typically False for direct execution, controlled by external runner usually
     start_api_server(host=host, port=port, workers=workers, reload=False, log_level=log_level)
