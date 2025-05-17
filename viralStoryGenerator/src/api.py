# viralStoryGenerator/src/api.py
"""
HTTP API backend for ViralStoryGenerator.
"""
import hmac
import uuid
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import redis
import uvicorn
import aiofiles

from viralStoryGenerator.models.models import (
    StoryGenerationRequest,
    JobResponse,
    HealthResponse,
    JobStatusResponse,
    ClearStalledJobsResponse,
    SuccessResponse,
    FailureResponse,
    AllQueueStatusResponse,
    QueueStatusResponse,
    QueueConsumerGroup,
    QueueConsumerDetail,
    QueueRecentMessage,
    SingleQueueStatusResponse
)

from viralStoryGenerator.utils.health_check import get_service_status
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.storage_manager import storage_manager
import logging
from viralStoryGenerator.utils.security import (
    is_safe_filename,
    is_file_in_directory,
    is_valid_uuid,
    is_valid_voice_id,
    sanitize_input,
    validate_path_component
)
from viralStoryGenerator.src.api_handlers import create_story_task, get_task_status, get_message_broker

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

app_start_time = time.time()
router = APIRouter()

# Initialize FastAPI app
app = FastAPI(
    title=app_config.APP_TITLE,
    description=app_config.APP_DESCRIPTION,
    version=app_config.VERSION,
)


API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Authentication dependency
async def get_api_key(request: Request, api_key: str = Depends(api_key_header)):
    if not app_config.http.API_KEY_ENABLED:
        return

    if not api_key:
        _logger.warning(f"Missing API Key from {request.client.host if request.client else 'Unknown client'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Missing API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )

    if not app_config.http.API_KEY or not hmac.compare_digest(api_key, app_config.http.API_KEY):
        _logger.warning(f"Invalid API Key received from {request.client.host if request.client else 'Unknown client'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Invalid API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    _logger.debug(f"Valid API Key received from {request.client.host if request.client else 'Unknown client'}")
    return api_key


# --- Middleware ---

# Add CORS middleware
if "*" in app_config.http.CORS_ORIGINS and app_config.ENVIRONMENT == "production":
    _logger.critical("CRITICAL SECURITY WARNING: CORS_ORIGINS allows '*' in production environment!")
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.http.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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
        redis_client.ping()
        _logger.info("Redis rate limiting enabled and connected")
    except Exception as e:
        _logger.warning(f"Failed to connect to Redis for rate limiting: {e}. Using local memory rate limiting fallback.")
        redis_client = None

class RateLimiter:
    """Rate limiter using sliding window algorithm"""

    def __init__(self, redis_client_instance, limit: int = 100, period: int = 60):
        self.redis = redis_client_instance
        self.limit = limit
        self.period = period
        self.local_cache = {}

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
                pipe = self.redis.pipeline()
                # Record the current request timestamp
                pipe.zadd(rate_key, {str(current_time): current_time})
                # Remove timestamps older than the window
                pipe.zremrangebyscore(rate_key, 0, current_time - window)
                pipe.zcard(rate_key)
                pipe.expire(rate_key, window)
                results = pipe.execute()
                request_count = results[2] # zcard result

                is_allowed = request_count <= limit

                if not is_allowed:
                    oldest_timestamps = self.redis.zrange(rate_key, 0, 0, withscores=True)
                    if oldest_timestamps:
                        oldest_ts = oldest_timestamps[0][1]
                        retry_after = int(window - (current_time - oldest_ts))
                    RATE_LIMIT_HIT.labels(client_ip=client_ip, endpoint=endpoint).inc()
                return is_allowed, request_count, limit, retry_after
            except Exception as e:
                _logger.error(f"Redis error during rate limiting for {client_ip} at {endpoint}: {e}. Falling back to local cache.")

        if rate_key not in self.local_cache:
            self.local_cache[rate_key] = []

        # Remove expired timestamps
        self.local_cache[rate_key] = [ts for ts in self.local_cache[rate_key] if ts > current_time - window]

        request_count = len(self.local_cache[rate_key])
        is_allowed = request_count < limit # Allow *up to* the limit

        if is_allowed:
            self.local_cache[rate_key].append(current_time)
        else:
            if self.local_cache[rate_key]:
                oldest_ts_in_window = min(self.local_cache[rate_key])
                retry_after = int(window - (current_time - oldest_ts_in_window))
            RATE_LIMIT_HIT.labels(client_ip=client_ip, endpoint=endpoint).inc()


        return is_allowed, request_count, limit, max(0, retry_after)

    def is_allowed(self, key: str, endpoint: str) -> bool:
        """Check if a request is allowed under the rate limit."""
        if not self.redis:
            _logger.warning(f"Rate limiter: Redis not available for key {key}, endpoint {endpoint}. Allowing request (fallback).")
            return True

        current_time = int(time.time())
        redis_key = f"rate_limit:{key}:{endpoint}"

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(redis_key, 0, current_time - self.period)
        pipe.zadd(redis_key, {str(current_time): current_time})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, self.period)
        results = pipe.execute()

        count = results[2]
        return count <= self.limit

rate_limiter = RateLimiter(redis_client)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if app_config.http.RATE_LIMIT_ENABLED and redis_client:
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        if not rate_limiter.is_allowed(client_ip, endpoint):
            RATE_LIMIT_HIT.labels(client_ip=client_ip, endpoint=endpoint).inc()
            _logger.warning(f"Rate limit exceeded for {client_ip} at {endpoint}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
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
    except HTTPException as http_exc: # Capture FastAPI's own HTTPExceptions
        status_code = http_exc.status_code
        raise
    except Exception:
        raise
    finally:
        process_time = time.time() - start_time
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(process_time)

        client_host = request.client.host if request.client else "Unknown"
        _logger.info(f"{client_host} - \"{request.method} {request.url.path} HTTP/{request.scope['http_version']}\" {status_code} {process_time:.4f}s")

    return response

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

# Define allowed file types, their extensions, and media types
FILE_TYPE_DETAILS = {
    "audio": {"extension": "mp3", "media_type": "audio/mpeg", "storage_file_type": "audio"},
    "story": {"extension": "txt", "media_type": "text/plain", "storage_file_type": "story"},
    "storyboard": {"extension": "json", "media_type": "application/json", "storage_file_type": "storyboard"},
    "metadata": {"extension": "json", "media_type": "application/json", "storage_file_type": "metadata"},
}

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



# --- File Serving Endpoints---

@router.get("/files/{task_id}/{file_type_key}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def serve_task_file(task_id: str, file_type_key: str, request: Request):
    """
    Serves a file associated with a task_id.
    Supports range requests for audio files for streaming.
    - task_id: The ID of the task.
    - file_type_key: The type of file to serve (e.g., "audio", "story", "metadata", "storyboard").
    """
    _logger.debug(f"Request to serve file for task_id: {task_id}, file_type_key: {file_type_key}")

    if not is_valid_uuid(task_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task_id format.")

    if file_type_key not in FILE_TYPE_DETAILS:
        allowed_keys = ", ".join(FILE_TYPE_DETAILS.keys())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file_type_key. Allowed types are: {allowed_keys}")

    details = FILE_TYPE_DETAILS[file_type_key]
    storage_file_type = details["storage_file_type"]
    media_type = details["media_type"]
    file_extension = details["extension"]

    actual_filename = f"{task_id}_{storage_file_type}.{file_extension}"

    if not is_safe_filename(actual_filename):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Generated filename is invalid and potentially unsafe.")

    try:
        serve_info = storage_manager.serve_file(actual_filename, storage_file_type)

        if isinstance(serve_info, str) and os.path.exists(serve_info):  # Local file path
            local_file_path = serve_info

            # Security check: Ensure file is within the designated storage directory
            storage_path_attr = f"{storage_file_type.upper()}_STORAGE_PATH"
            if hasattr(app_config.storage, storage_path_attr):
                expected_base_dir = os.path.abspath(getattr(app_config.storage, storage_path_attr))
                if not is_file_in_directory(local_file_path, expected_base_dir):
                    _logger.error(f"Access denied: File {local_file_path} is not in expected directory {expected_base_dir}.")
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to file denied.")
            else:
                _logger.warning(f"Could not determine base storage directory for file_type '{storage_file_type}' for security check.")
                # Strict: raise error if path cannot be verified.
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cannot verify file location security.")

            if file_type_key == "audio" and request.headers.get("Range"):
                file_size = os.path.getsize(local_file_path)
                range_header = request.headers.get("Range")

                start_byte_str, end_byte_str = None, None
                if range_header and range_header.lower().startswith("bytes="):
                    parts = range_header.split("=")[1].split("-")
                    start_byte_str = parts[0]
                    if len(parts) > 1 and parts[1]:
                        end_byte_str = parts[1]

                start = int(start_byte_str) if start_byte_str and start_byte_str.isdigit() else 0
                end = int(end_byte_str) if end_byte_str and end_byte_str.isdigit() else file_size - 1

                if not (0 <= start < file_size and start <= end < file_size):
                    _logger.warning(f"Invalid range requested: {start}-{end} for file size {file_size}")
                    raise HTTPException(
                        status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                        detail=f"Range {start}-{end} not satisfiable for file size {file_size}",
                        headers={"Content-Range": f"bytes */{file_size}"}
                    )

                content_length = (end - start) + 1
                http_status_code = status.HTTP_206_PARTIAL_CONTENT

                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Content-Type": media_type,
                    "Cache-Control": "public, max-age=3600"
                }

                async def chunk_supplier(file_path: str, offset: int, length: int):
                    async with aiofiles.open(file_path, 'rb') as f:
                        await f.seek(offset)
                        remaining = length
                        while remaining > 0:
                            data_to_read = min(remaining, 65536)  # 64KB chunks
                            chunk = await f.read(data_to_read)
                            if not chunk:
                                break
                            yield chunk
                            remaining -= len(chunk)

                return StreamingResponse(
                    chunk_supplier(local_file_path, start, content_length),
                    status_code=http_status_code,
                    headers=headers,
                    media_type=media_type
                )
            else:
                return FileResponse(
                    path=local_file_path,
                    media_type=media_type,
                    filename=actual_filename
                )

        elif isinstance(serve_info, dict) and 'url' in serve_info:  # S3/Azure presigned URL
            return RedirectResponse(url=serve_info['url'])

        elif isinstance(serve_info, dict) and 'error' in serve_info:
            _logger.error(f"Storage manager error for {actual_filename} ({storage_file_type}): {serve_info['error']}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error serving file from storage: {serve_info['error']}")

        else:
            _logger.warning(f"File not found or unable to serve by storage_manager: task_id={task_id}, file_type_key={file_type_key}, constructed_filename={actual_filename}, serve_info: {serve_info}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{file_type_key.capitalize()} file not found for task {task_id}.")

    except FileNotFoundError:
        _logger.warning(f"Direct FileNotFoundError for task_id={task_id}, file_type_key={file_type_key}, constructed_filename={actual_filename}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{file_type_key.capitalize()} file not found.")
    except HTTPException as he:
        raise he
    except Exception as e:
        _logger.exception(f"Unexpected error serving file for task_id={task_id}, file_type_key={file_type_key}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while serving file.")

# --- Job Management Endpoints (Using Redis Queue Directly) ---
API_QUEUE_NAME = app_config.redis.QUEUE_NAME
RESULT_PREFIX = app_config.redis.RESULT_PREFIX

@router.post("/generate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Job Management"], dependencies=[Depends(get_api_key)])
async def generate_story_from_urls_endpoint(
    request_data: StoryGenerationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate a viral story from the provided URLs.
    This endpoint queues the request for processing and returns a job ID.
    Relies on a separate worker process consuming from the Redis stream.
    """
    _logger.info(f"Received request to generate story from URLs for topic: '{request_data.topic}'")
    try:
        if not request_data.topic:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Topic cannot be empty.")
        if not request_data.urls:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URLs list cannot be empty.")

        # Sanitize inputs
        sanitized_topic = sanitize_input(request_data.topic)
        sanitized_urls = [sanitize_input(str(url)) for url in request_data.urls]
        raw_voice_id = getattr(request_data, 'voice_id', None)
        sanitized_voice_id = sanitize_input(raw_voice_id) if raw_voice_id else None

        if sanitized_voice_id and not is_valid_voice_id(sanitized_voice_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid voice_id format.")

        job_id = str(uuid.uuid4())

        # Safely access other optional fields for the task payload
        include_storyboard = getattr(request_data, 'include_storyboard', None)
        custom_prompt = getattr(request_data, 'custom_prompt', None)
        output_format = getattr(request_data, 'output_format', None)
        temperature = getattr(request_data, 'temperature', None)
        chunk_size = getattr(request_data, 'chunk_size', None)

        task_payload = {
            "job_id": job_id,
            "job_type": "generate_story",
            "topic": sanitized_topic,
            "urls": sanitized_urls,
            "voice_id": sanitized_voice_id,
            "include_storyboard": include_storyboard if include_storyboard is not None \
                                 else app_config.storyboard.ENABLE_STORYBOARD_GENERATION,
            "request_time": time.time(),
            "custom_prompt": sanitize_input(custom_prompt) if custom_prompt else None,
            "output_format": sanitize_input(output_format) if output_format else "standard",
            "temperature": temperature if temperature is not None else app_config.llm.TEMPERATURE,
            "chunk_size": chunk_size if chunk_size is not None else app_config.llm.CHUNK_SIZE,
        }

        message_broker = get_message_broker()
        message_id = message_broker.publish_message(task_payload)

        if message_id:
            _logger.info(f"Job {job_id} for topic '{sanitized_topic}' queued successfully with message ID {message_id}.")
            return JobResponse(job_id=job_id, message="Story generation task queued.")
        else:
            _logger.error(f"Failed to queue job {job_id} for topic '{sanitized_topic}'.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to queue story generation task.")

    except HTTPException as he:
        _logger.warning(f"HTTPException during /generate: {he.detail}")
        raise he
    except Exception as e:
        _logger.exception(f"Unexpected error in /generate for topic '{request_data.topic}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Job Management"], dependencies=[Depends(get_api_key)])
async def get_job_status_endpoint(job_id: str):
    """Retrieve the status of a previously submitted job."""
    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job_id format.")

    _logger.debug(f"Fetching status for job_id: {job_id}")
    try:
        message_broker = get_message_broker()
        status_data = await message_broker.get_job_progress(job_id)

        if not status_data:
            _logger.info(f"Job status for {job_id} not found in Redis stream/results.")
            handler_status = get_task_status(job_id)
            if handler_status and handler_status.get("status") != "not_found":
                _logger.info(f"Job status for {job_id} found via api_handlers.get_task_status.")
                return JobStatusResponse(**handler_status)
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job with ID {job_id} not found.")

        response_data = {
            "job_id": status_data.get("job_id", job_id),
            "status": status_data.get("status", "unknown"),
            "message": status_data.get("message", "Status retrieved from Redis."),
            "created_at": status_data.get("created_at") or status_data.get("request_time"),
            "updated_at": status_data.get("updated_at") or status_data.get("timestamp"),
            "error": status_data.get("error"),
            "result": status_data.get("result"),
            "story_script": status_data.get("story_script", status_data.get("result", {}).get("story_script") if isinstance(status_data.get("result"), dict) else None),
            "storyboard": status_data.get("storyboard", status_data.get("result", {}).get("storyboard") if isinstance(status_data.get("result"), dict) else None),
            "audio_url": status_data.get("audio_url", status_data.get("result", {}).get("audio_url") if isinstance(status_data.get("result"), dict) else None),
            "video_url": status_data.get("video_url", status_data.get("result", {}).get("video_url") if isinstance(status_data.get("result"), dict) else None),
            "image_urls": status_data.get("image_urls", status_data.get("result", {}).get("image_urls") if isinstance(status_data.get("result"), dict) else None),
            "sources": status_data.get("sources", status_data.get("result", {}).get("sources") if isinstance(status_data.get("result"), dict) else None),
        }

        return JobStatusResponse(**response_data)

    except HTTPException as he:
        raise he
    except Exception as e:
        _logger.exception(f"Error retrieving status for job {job_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve job status: {str(e)}")

# --- Queue Management Admin Endpoints ---
queue_router = APIRouter(
    prefix="/queue",
    tags=["Queue Management"],
    dependencies=[Depends(get_api_key)]
)

@queue_router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """
    Get the current status of the job queue system (api_jobs stream).
    Returns metrics like queue length, groups, and recent job status counts.
    """
    _logger.info("Request received for queue status.")
    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)

        raw_status = await message_broker.get_queue_information()
        raw_consumer_groups = raw_status.get("consumer_groups", [])
        mapped_consumer_groups = []
        for group_data in raw_consumer_groups:
            if isinstance(group_data, dict):
                raw_consumer_details = group_data.get("consumer_details", [])
                mapped_consumer_details = []
                if isinstance(raw_consumer_details, list):
                    for detail_data in raw_consumer_details:
                        if isinstance(detail_data, dict):
                            try:
                                mapped_consumer_details.append(QueueConsumerDetail(**detail_data))
                            except Exception as e_detail:
                                _logger.warning(f"Failed to map consumer detail: {detail_data}, error: {e_detail}")
                        else:
                            _logger.warning(f"Skipping non-dict consumer detail: {detail_data}")
                else:
                     _logger.warning(f"consumer_details for group {group_data.get('group_name')} is not a list: {raw_consumer_details}")

                group_data["consumer_details"] = mapped_consumer_details
                try:
                    mapped_consumer_groups.append(QueueConsumerGroup(**group_data))
                except Exception as e_group:
                    _logger.warning(f"Failed to map consumer group: {group_data}, error: {e_group}")
            else:
                _logger.warning(f"Skipping non-dict consumer group: {group_data}")


        raw_recent_messages = raw_status.get("recent_messages", [])
        mapped_recent_messages = []
        for msg_data in raw_recent_messages:
            if isinstance(msg_data, dict):
                try:
                    mapped_recent_messages.append(QueueRecentMessage(**msg_data))
                except Exception as e_msg:
                    _logger.warning(f"Failed to map recent message: {msg_data}, error: {e_msg}")
            else:
                _logger.warning(f"Skipping non-dict recent message: {msg_data}")

        # Check for overall error status from get_queue_information
        if raw_status.get("status") == "error":
            _logger.error(f"Error reported by get_queue_information: {raw_status.get('error_message')}")
            return QueueStatusResponse(
                status="error",
                stream_length=0,
                consumer_groups=[],
                recent_messages=[],
            )

        mapped_status = QueueStatusResponse(
            status=raw_status.get("status", "available"),
            stream_length=raw_status.get("stream_length", 0),
            consumer_groups=mapped_consumer_groups,
            recent_messages=mapped_recent_messages
        )
        return mapped_status
    except Exception as e:
        _logger.exception(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@queue_router.post("/clear-stalled", response_model=ClearStalledJobsResponse)
async def clear_stalled_jobs(max_age_seconds: int = 600): # Default: older than 10 minutes
    """
    Clear potentially stalled jobs from the stream's consumer groups.
    Claims and processes messages that have been pending for longer than max_age_seconds.
    """
    _logger.warning(f"Request received to clear stalled jobs older than {max_age_seconds}s.")

    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        message_broker = RedisMessageBroker(redis_url=redis_url, stream_name="api_jobs")

        redis_client = message_broker.client

        # Get all consumer groups for the stream
        groups_info = await redis_client.xinfo_groups(app_config.redis.QUEUE_NAME)

        claimed_count = 0
        failed_to_claim_count = 0
        reprocessed_count = 0
        errors = []

        for group_info in groups_info:
            name_bytes = group_info.get(b'name')
            if name_bytes is None:
                continue
            group_name = name_bytes.decode()
            # Get pending messages for this group
            pending_info = await redis_client.xpending_range(
                app_config.redis.QUEUE_NAME,
                group_name,
                min="-",  # Start from oldest
                max="+",  # End with newest
                count=100  # Limit number of messages
            )

            for item in pending_info:
                message_id = item.get(b'message_id').decode()
                consumer = item.get(b'consumer').decode()
                idle_time = item.get(b'idle')

                if idle_time > max_age_seconds * 1000:  # Redis uses milliseconds
                    try:
                        claimed = await redis_client.xclaim(
                            app_config.redis.QUEUE_NAME,
                            group_name,
                            "stalled_job_processor",
                            min_idle_time=idle_time,
                            message_ids=[message_id]
                        )

                        if claimed:
                            claimed_count += 1
                            for msg_id, msg_data in claimed:
                                try:
                                    job_id = msg_data.get(b'job_id', b'unknown').decode()

                                    error_message = f"Job stalled: no progress for {idle_time/1000} seconds"

                                    message_broker.publish_message({
                                        "job_id": job_id,
                                        "status": "failed",
                                        "error": error_message,
                                        "timestamp": time.time()
                                    })

                                    failed_to_claim_count += 1

                                except Exception as msg_e:
                                    _logger.error(f"Error processing stalled message {message_id}: {msg_e}")

                            await redis_client.xack(app_config.redis.QUEUE_NAME, group_name, message_id)

                    except Exception as claim_e:
                        _logger.error(f"Error claiming stalled message {message_id}: {claim_e}")

        _logger.info(f"Stalled job cleanup finished. Claimed: {claimed_count}, Failed: {failed_to_claim_count}, Reprocessed: {reprocessed_count}")
        return {
            "message": f"Stalled job cleanup completed. Claimed: {claimed_count}, Failed: {failed_to_claim_count}, Reprocessed: {reprocessed_count}",
            "claimed_count": claimed_count,
            "failed_count": failed_to_claim_count,
            "reprocessed_count": reprocessed_count
        }
    except Exception as e:
        _logger.exception(f"Error clearing stalled jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear stalled jobs: {str(e)}")

@queue_router.delete("/purge", response_model=SuccessResponse, responses={400: {"model": FailureResponse}, 500: {"model": FailureResponse}})
async def purge_queue(confirmation: str):
    """
    PURGE ALL jobs from the Redis Stream.
    This is a DESTRUCTIVE operation. Requires confirmation query parameter.
    """
    CONFIRMATION_CODE = "CONFIRM_PURGE_ALL_JOBS_SERIOUSLY"
    if confirmation != CONFIRMATION_CODE:
        _logger.warning("Queue purge attempted without correct confirmation code.")
        raise HTTPException(status_code=400, detail=f"Invalid confirmation. Use confirmation='{CONFIRMATION_CODE}' to confirm this destructive action.")

    _logger.critical(f"Executing QUEUE PURGE operation with confirmation '{confirmation}'")
    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)

        redis_client = message_broker.client

        # Delete the entire stream
        stream_name_to_purge = app_config.redis.QUEUE_NAME
        stream_deleted = await redis_client.delete(stream_name_to_purge)

        _logger.info(f"Stream '{stream_name_to_purge}' deleted: {bool(stream_deleted)}.")

        new_message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=stream_name_to_purge)

        _logger.critical(f"Queue system purged. Stream '{stream_name_to_purge}' deleted: {bool(stream_deleted)}. Stream and group recreated by new broker instance.")
        return SuccessResponse(
            message=f"Queue system purged successfully.",
            detail=f"Stream '{stream_name_to_purge}' deleted: {bool(stream_deleted)}. Stream and consumer group were recreated."
        )
    except Exception as e:
        _logger.exception(f"Error purging queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to purge queue: {str(e)}")

@queue_router.post("/job/{job_id}/retry", response_model=SuccessResponse, responses={400: {"model": FailureResponse}, 404: {"model": FailureResponse}, 500: {"model": FailureResponse}})
async def retry_failed_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Retry a FAILED job by re-queuing a new message to the stream.
    """
    _logger.info(f"Request received to retry job_id: {job_id}")
    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)

        # Find the latest status for the job
        job_status_data = await message_broker.get_job_progress(job_id)

        if not job_status_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found, cannot retry.")

        current_status = job_status_data.get("status")
        if current_status != "failed":
            raise HTTPException(status_code=400, detail=f"Only 'failed' jobs can be retried. Current status: '{current_status}'")

        original_job_payload = job_status_data.get("data")
        if not original_job_payload or not isinstance(original_job_payload, dict):
            original_job_payload = {
                "topic": job_status_data.get("topic", "Unknown Topic"),
                "urls": job_status_data.get("urls", []),
                "voice_id": job_status_data.get("voice_id"),
                "include_storyboard": job_status_data.get("include_storyboard", app_config.storyboard.ENABLE_STORYBOARD_GENERATION),
                "custom_prompt": job_status_data.get("custom_prompt"),
                "output_format": job_status_data.get("output_format", "standard"),
                "temperature": job_status_data.get("temperature", app_config.llm.TEMPERATURE),
                "chunk_size": job_status_data.get("chunk_size", app_config.llm.CHUNK_SIZE),
            }

        new_job_id = str(uuid.uuid4())

        retry_task_payload = {
            "data": original_job_payload,
            "retry_of_job_id": job_id
        }

        message_id = message_broker.publish_message(message_data=retry_task_payload, job_id=new_job_id)

        if not message_id:
            _logger.error(f"Failed to re-queue job {job_id} for retry as {new_job_id}.")
            raise HTTPException(status_code=500, detail="Failed to queue job for retry.")

        message_broker.track_job_progress(
            job_id=job_id,
            status="retried",
            data={"retried_as_job_id": new_job_id, "message": f"Job retried as {new_job_id}"}
        )

        _logger.info(f"Job {job_id} successfully retried as new job {new_job_id} with message ID {message_id}.")
        return SuccessResponse(
            message=f"Job {job_id} has been retried as new job {new_job_id}.",
            detail=f"Original job_id: {job_id}, new_job_id: {new_job_id}, new_message_id: {message_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception(f"Error retrying job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error retrying job: {str(e)}")

@queue_router.get("/all-status", response_model=AllQueueStatusResponse)
async def get_all_queue_status():
    """Get status for all known Redis streams that look like job queues."""
    _logger.info("Request received for all queue statuses.")
    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"

        queue_names_to_check = [app_config.redis.QUEUE_NAME]
        queue_names_to_check = list(set(queue_names_to_check))

        all_statuses_dict: Dict[str, SingleQueueStatusResponse] = {}

        for queue_name in queue_names_to_check:
            try:
                broker = RedisMessageBroker(redis_url=redis_url, stream_name=queue_name)
                raw_status = await broker.get_queue_information()

                groups_list: List[QueueConsumerGroup] = []
                if raw_status.get("consumer_groups"):
                    for group_data in raw_status["consumer_groups"]:
                        consumers_detail_list: List[QueueConsumerDetail] = []
                        raw_consumers_data = group_data.get("consumer_details")

                        if isinstance(raw_consumers_data, list):
                            for cons_data_dict in raw_consumers_data:
                                if isinstance(cons_data_dict, dict):
                                     try:
                                         consumers_detail_list.append(QueueConsumerDetail(**cons_data_dict))
                                     except Exception as e_detail_all:
                                         _logger.warning(f"AllQueueStatus: Failed to map consumer detail for group {group_data.get('group_name')}: {cons_data_dict}, error: {e_detail_all}")
                                else:
                                    _logger.warning(f"AllQueueStatus: Unexpected consumer detail data type for group {group_data.get('group_name')}: {type(cons_data_dict)}")

                        group_data["consumer_details"] = consumers_detail_list
                        try:
                            groups_list.append(QueueConsumerGroup(**group_data))
                        except Exception as e_group_all:
                             _logger.warning(f"AllQueueStatus: Failed to map consumer group {group_data.get('group_name')}: {group_data}, error: {e_group_all}")


                recent_msgs_list: List[QueueRecentMessage] = []
                if raw_status.get("recent_messages"):
                    for msg_data in raw_status["recent_messages"]:
                        if isinstance(msg_data, dict):
                            try:
                                recent_msgs_list.append(QueueRecentMessage(**msg_data))
                            except Exception as e_msg_all:
                                _logger.warning(f"AllQueueStatus: Failed to map recent message: {msg_data}, error: {e_msg_all}")
                        else:
                             _logger.warning(f"AllQueueStatus: Skipping non-dict recent message: {msg_data}")


                if raw_status.get("status") == "error":
                     all_statuses_dict[queue_name] = SingleQueueStatusResponse(
                        stream_name=queue_name,
                        status="error",
                        error_message=raw_status.get("error_message", "Failed to retrieve queue status"),
                        stream_length=0,
                        consumer_groups=[],
                        recent_messages=[]
                    )
                else:
                    single_status = SingleQueueStatusResponse(
                        stream_name=queue_name,
                        status=raw_status.get("status", "unknown"),
                        stream_length=raw_status.get("stream_length", 0),
                        consumer_groups_count=len(groups_list),
                        consumer_groups=groups_list,
                        recent_messages_count=len(recent_msgs_list),
                        recent_messages=recent_msgs_list
                    )
                    all_statuses_dict[queue_name] = single_status

            except Exception as e:
                _logger.error(f"Failed to get status for queue {queue_name}: {e}")
                all_statuses_dict[queue_name] = SingleQueueStatusResponse(
                    stream_name=queue_name,
                    status="error",
                    error_message=str(e),
                    stream_length=0,
                    consumer_groups=[],
                    recent_messages=[]
                )

        return AllQueueStatusResponse(root=all_statuses_dict)

    except Exception as e:
        _logger.exception(f"Error getting all queue statuses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get all queue statuses: {str(e)}")

@queue_router.post("/config/storyboard", tags=["Configuration"])
async def toggle_storyboard_generation(enabled: bool):
    app_config.storyboard.ENABLE_STORYBOARD_GENERATION = enabled
    _logger.info(f"Storyboard generation dynamically set to: {enabled}")
    return {"message": f"Storyboard generation {'enabled' if enabled else 'disabled'}.", "new_status": enabled}

@queue_router.post("/config/image-generation", tags=["Configuration"])
async def toggle_image_generation(enabled: bool):
    app_config.ENABLE_IMAGE_GENERATION = enabled
    app_config.openAI.ENABLED = enabled
    _logger.info(f"Image generation dynamically set to: {enabled}")
    return {"message": f"Image generation {'enabled' if enabled else 'disabled'}.", "new_status": enabled}

@queue_router.post("/config/audio-generation", tags=["Configuration"])
async def toggle_audio_generation(enabled: bool):
    app_config.ENABLE_AUDIO_GENERATION = enabled
    app_config.elevenLabs.ENABLED = enabled
    _logger.info(f"Audio generation dynamically set to: {enabled}")
    return {"message": f"Audio generation {'enabled' if enabled else 'disabled'}.", "new_status": enabled}

app.include_router(router, prefix="/api")
app.include_router(queue_router, prefix="/api")

os.makedirs(app_config.storage.AUDIO_STORAGE_PATH, exist_ok=True)
os.makedirs(app_config.storage.STORY_STORAGE_PATH, exist_ok=True)
os.makedirs(app_config.storage.STORYBOARD_STORAGE_PATH, exist_ok=True)
app.mount("/static", StaticFiles(directory=app_config.storage.LOCAL_STORAGE_PATH), name="static")

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
        ssl_keyfile=app_config.http.SSL_KEY_FILE if app_config.http.SSL_ENABLED else None,
        ssl_certfile=app_config.http.SSL_CERT_FILE if app_config.http.SSL_ENABLED else None,
    )

if __name__ == "__main__":
     host = app_config.http.HOST
     port = app_config.http.PORT
     workers = app_config.http.WORKERS
     log_level = app_config.LOG_LEVEL
     start_api_server(host=host, port=port, workers=workers, reload=False, log_level=log_level)
