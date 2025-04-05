"""
HTTP API backend for ViralStoryGenerator.
"""
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
from viralStoryGenerator.src.api_worker import process_story_generation
from viralStoryGenerator.utils.redis_manager import RedisManager as RedisQueueManager
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.security import (
    is_safe_filename,
    is_file_in_directory,
    is_valid_uuid,
    is_valid_voice_id,
    sanitize_input
)
from viralStoryGenerator.src.api_handlers import (
    create_story_task,
    get_task_status
)


app_start_time = time.time()

router = APIRouter(
    prefix="/api",
    tags=["stories"],
    responses={404: {"description": "Not found"}}
)

# Initialize FastAPI app
app = FastAPI(
    title=app_config.APP_TITLE,
    description=app_config.APP_DESCRIPTION,
    version=app_config.VERSION
)

# Include the router in the app
app.include_router(router)

# Setup API security if enabled in config
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Authentication dependency
async def get_api_key(request: Request, api_key: str = Depends(api_key_header)):
    # Skip authentication if API key security is disabled
    if not app_config.http.API_KEY_ENABLED:
        return None

    if api_key != app_config.http.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key


queue_router = APIRouter(
    prefix="/api/queue",
    tags=["Queue Management"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(get_api_key)]  # Require API key for all endpoints in this router
)

# Add the new router to the app (add this after the existing app.include_router(router) line)
app.include_router(queue_router)

# Mount static file directory for local storage
os.makedirs(app_config.storage.LOCAL_STORAGE_PATH, exist_ok=True)
app.mount("/static", StaticFiles(directory=app_config.storage.LOCAL_STORAGE_PATH), name="static")

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Active requests')
QUEUE_SIZE = Gauge('api_queue_size', 'API queue size')
RATE_LIMIT_HIT = Counter('api_rate_limit_hit_total', 'Rate limit exceeded count', ['client_ip'])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.http.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client for rate limiting
redis_client = None
if app_config.http.RATE_LIMIT_ENABLED and app_config.redis.ENABLED:
    try:
        redis_client = redis.Redis(
            host=app_config.redis.HOST,
            port=app_config.redis.PORT,
            db=app_config.redis.DB,
            password=app_config.redis.PASSWORD,
            decode_responses=True
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

    async def check_rate_limit(self, client_ip: str, endpoint: str) -> Tuple[bool, int, int]:
        """
        Check if the client has exceeded rate limits

        Args:
            client_ip: IP address of the client
            endpoint: API endpoint being accessed

        Returns:
            Tuple of (is_allowed, current_count, limit)
        """
        window = app_config.http.RATE_LIMIT_WINDOW
        limit = app_config.http.RATE_LIMIT_REQUESTS

        # Create a unique key for each client IP and endpoint combination
        rate_key = f"rate_limit:{client_ip}:{endpoint}"
        current_time = time.time()

        # If Redis is available, use it for distributed rate limiting
        if self.redis and self.redis.ping():
            # Use Redis sorted set for sliding window
            # Remove entries outside the current window
            self.redis.zremrangebyscore(rate_key, 0, current_time - window)

            # Add current request
            pipeline = self.redis.pipeline()
            pipeline.zadd(rate_key, {str(current_time): current_time})
            pipeline.zcount(rate_key, 0, float('inf'))
            pipeline.expire(rate_key, window * 2)  # Set expiry to avoid leaking memory
            _, request_count, _ = pipeline.execute()

            is_allowed = request_count <= limit
            return is_allowed, request_count, limit

        # Fallback to local memory if Redis is unavailable
        if rate_key not in self.local_cache:
            self.local_cache[rate_key] = []

        # Remove expired timestamps
        self.local_cache[rate_key] = [ts for ts in self.local_cache[rate_key] if ts > current_time - window]

        # Add current request
        self.local_cache[rate_key].append(current_time)
        request_count = len(self.local_cache[rate_key])

        is_allowed = request_count <= limit
        return is_allowed, request_count, limit

# Create rate limiter instance
rate_limiter = RateLimiter(redis_client)

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    _logger.debug(f"Metrics middleware triggered for {request.method} {request.url.path}")
    ACTIVE_REQUESTS.inc()
    request_start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        request_duration = time.time() - request_start_time
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(request_duration)
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=status_code).inc()
        ACTIVE_REQUESTS.dec()

    _logger.debug(f"Metrics middleware completed for {request.method} {request.url.path}")
    return response

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    _logger.debug(f"Rate limit middleware triggered for {request.method} {request.url.path}")
    if app_config.http.RATE_LIMIT_ENABLED:
        client_ip = request.client.host
        endpoint = request.url.path

        # Skip rate limiting for certain endpoints
        if endpoint in ['/health', '/metrics']:
            return await call_next(request)

        # Check if the request can be processed
        is_allowed, current, limit = await rate_limiter.check_rate_limit(client_ip, endpoint)

        # If request exceeds limit, return 429 Too Many Requests
        if not is_allowed:
            RATE_LIMIT_HIT.labels(client_ip=client_ip).inc()
            _logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}: {current}/{limit}")

            # Calculate remaining window time for retry-after header
            retry_after = app_config.http.RATE_LIMIT_WINDOW

            # Custom rate limit exceeded response
            return JSONResponse(
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

    # Process the request normally
    response = await call_next(request)

    # Add rate limit headers to response if enabled
    if app_config.http.RATE_LIMIT_ENABLED:
        # Include rate limit information in response headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - current)

    _logger.debug(f"Rate limit middleware completed for {request.method} {request.url.path}")
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    _logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Redis queue for API requests
API_QUEUE_NAME = "api_requests"
RESULT_PREFIX = "api_result:"

# Redis queue manager dependency
def get_queue_manager() -> RedisQueueManager:
    try:
        manager = RedisQueueManager(
            queue_name=API_QUEUE_NAME,
            result_prefix=RESULT_PREFIX
        )

        # Update queue size metric
        try:
            QUEUE_SIZE.set(manager.get_queue_length())
        except:
            # Don't fail if we can't update metrics
            pass

        return manager
    except Exception as e:
        _logger.error(f"Failed to initialize Redis queue: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis queue service unavailable")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _logger.error(f"Uncaught exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "detail": str(exc)},
    )

# Health and Metrics Endpoints
@app.get("/metrics", tags=["Health and Metrics"])
async def get_metrics():
    """
    Endpoint to expose Prometheus metrics.
    This is used by Prometheus to scrape metrics from the application.
    """
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health", response_model=HealthResponse, tags=["Health and Metrics"])
async def health_check():
    """Health check endpoint for monitoring"""
    _logger.debug("Health check endpoint called.")

    # Fetch detailed service statuses
    service_status = await get_service_status()

    _logger.debug("Health check response generated.")
    return service_status

# Story Management Endpoints
@app.post("/api/stories", dependencies=[Depends(get_api_key)], tags=["Story Management"])
async def generate_story(
    topic: str,
    background_tasks: BackgroundTasks,
    sources_folder: Optional[str] = None,
    voice_id: Optional[str] = None
):
    """
    Create a new story generation task

    Parameters:
    - topic: Topic for the story
    - sources_folder: Optional folder with source material
    - voice_id: Optional ElevenLabs voice ID

    Returns:
    - task_id: ID to check status later
    - status: initial status of the task
    """
    _logger.debug(f"Generate story endpoint called with topic: {topic}")

    # Validate topic
    if not topic or len(topic) > 500:
        raise HTTPException(
            status_code=400,
            detail="Topic must be between 1 and 500 characters"
        )

    # Check for potential command injection characters
    if any(char in topic for char in ['&', '|', ';', '$', '`', '\\']):
        _logger.warning(f"Security: Potentially malicious topic detected: {topic}")
        raise HTTPException(
            status_code=400,
            detail="Topic contains invalid characters"
        )

    # Validate sources_folder if provided
    if sources_folder:
        # Prevent path traversal
        if '..' in sources_folder or sources_folder.startswith('/') or '\\' in sources_folder:
            _logger.warning(f"Security: Path traversal attempt detected in sources_folder: {sources_folder}")
            raise HTTPException(
                status_code=400,
                detail="Invalid sources folder path"
            )

        # Ensure the folder exists and is within the allowed directories
        full_path = os.path.abspath(os.path.join(app_config.storage.SOURCE_MATERIALS_PATH, sources_folder))
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            raise HTTPException(
                status_code=404,
                detail=f"Sources folder not found: {sources_folder}"
            )

        # Verify the folder is within the allowed source materials directory
        if not is_file_in_directory(full_path, app_config.storage.SOURCE_MATERIALS_PATH):
            _logger.warning(f"Security: Attempted access to folder outside allowed path: {full_path}")
            raise HTTPException(
                status_code=403,
                detail="Access to the specified folder is not allowed"
            )

    # Validate voice_id if provided (should match ElevenLabs voice ID format)
    if voice_id and not is_valid_voice_id(voice_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid voice ID format"
        )

    # Create the story task
    task = create_story_task(topic, sources_folder, voice_id)
    _logger.debug(f"Story generation task created for topic: {topic}")
    return task

@app.get("/api/stories/{task_id}", dependencies=[Depends(get_api_key)], tags=["Story Management"])
async def check_story_status(task_id: str):
    _logger.debug(f"Check story status endpoint called for task_id: {task_id}")
    """
    Check the status of a story generation task

    Parameters:
    - task_id: ID of the task to check

    Returns:
    - Task details including status and results if completed
    """
    task_info = get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    _logger.debug(f"Story status retrieved for task_id: {task_id}")
    return task_info

@app.get("/api/stories/{task_id}/download/{file_type}", dependencies=[Depends(get_api_key)], tags=["Story Management"])
async def download_story_file(task_id: str, file_type: str):
    """
    Download a generated file from a completed story task

    Parameters:
    - task_id: ID of the task
    - file_type: Type of file to download (story, audio, storyboard)

    Returns:
    - File download response
    """
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
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not completed")

    file_paths = task_info.get("file_paths", {})
    if file_type not in file_paths or not file_paths[file_type]:
        raise HTTPException(status_code=404, detail=f"No {file_type} file available for task {task_id}")

    file_path = file_paths[file_type]
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    expected_directory = None
    if file_type == "audio":
        expected_directory = app_config.storage.AUDIO_STORAGE_PATH
    elif file_type == "story":
        expected_directory = app_config.storage.STORY_STORAGE_PATH
    elif file_type == "storyboard":
        expected_directory = app_config.storage.STORYBOARD_STORAGE_PATH

    if not is_file_in_directory(file_path, expected_directory):
        _logger.warning(f"Security: Attempted access to file outside storage directory: {file_path}")
        raise HTTPException(status_code=403, detail="Access denied")

    # Set appropriate content type and filename
    filename = os.path.basename(file_path)

    # Validate filename is safe
    if not is_safe_filename(filename):
        _logger.warning(f"Security: Suspicious filename detected: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")

    if file_type == "audio":
        media_type = "audio/mpeg"
    elif file_type == "storyboard":
        media_type = "application/json"
    else:  # story text
        media_type = "text/plain"

    return FileResponse(path=file_path, media_type=media_type, filename=filename)

# File Serving Endpoints
@app.get("/audio/{filename}", tags=["File Serving"])
async def serve_audio_file(filename: str):
    """Serve audio files directly"""
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(app_config.storage.AUDIO_STORAGE_PATH, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    if not is_file_in_directory(file_path, app_config.storage.AUDIO_STORAGE_PATH):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )

@app.get("/api/audio/stream/{filename}")
async def stream_audio(
    filename: str,
    range: str = None
):
    """
    Stream audio file with support for range requests (needed for seeking in audio players)
    """
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Get file size and check existence
    file_size = None
    file_path = None

    if app_config.storage.PROVIDER == "local":
        # For local files, check the file system directly
        file_path = os.path.join(app_config.storage.AUDIO_STORAGE_PATH, filename)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        if not is_file_in_directory(file_path, app_config.storage.AUDIO_STORAGE_PATH):
            raise HTTPException(status_code=403, detail="Access denied")

        file_size = os.path.getsize(file_path)
    else:
        # For cloud storage, check if file exists by getting metadata
        try:
            # First try getting a full file to check its existence and size
            file_data = storage_manager.retrieve_file(filename, "audio")
            if not file_data:
                raise HTTPException(status_code=404, detail="Audio file not found in storage")

            # For S3 or Azure, we need the file size for range requests
            if isinstance(file_data, bytes):
                file_size = len(file_data)
                # Save to temporary file for streaming
                file_path = os.path.join(tempfile.gettempdir(), filename)
                with open(file_path, "wb") as f:
                    f.write(file_data)
            else:
                # This shouldn't happen as we're not specifying a range yet
                raise HTTPException(status_code=500, detail="Unexpected response format from storage")
        except Exception as e:
            _logger.error(f"Error retrieving audio from storage: {e}")
            raise HTTPException(status_code=404, detail="Audio file not found")

    # Handle range requests for audio seeking
    start = 0
    end = file_size - 1
    status_code = 200

    if range is not None:
        try:
            # Parse range header (e.g., "bytes=0-1023")
            range_header = range.replace("bytes=", "").split("-")
            start = int(range_header[0]) if range_header[0] else 0
            end = int(range_header[1]) if range_header[1] and range_header[1].isdigit() else file_size - 1

            # Validate range
            if end >= file_size:
                end = file_size - 1

            # Prevent negative ranges
            if start < 0:
                start = 0
            if end < 0:
                end = 0

            # Use 206 Partial Content for range requests
            if start > 0 or end < file_size - 1:
                status_code = 206
        except ValueError:
            # If range header is invalid, ignore it
            pass

    # Calculate content length
    content_length = max(0, end - start + 1)

    # Define headers
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Cache-Control": "public, max-age=3600"  # Allow caching for 1 hour
    }

    # Function to stream file content in chunks
    async def file_streamer():
        if app_config.storage.PROVIDER == "local" or file_path:
            # Stream from local file
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024  # 64KB chunks for efficient streaming

                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break

                    yield chunk
                    remaining -= len(chunk)
        else:
            # Stream directly from cloud storage
            try:
                # Get a streaming response with the specified range
                stream = storage_manager.retrieve_file(
                    filename=filename,
                    file_type="audio",
                    start_byte=start,
                    end_byte=end
                )

                # Handle different return types from different storage providers
                if hasattr(stream, 'read'):
                    # File-like object (S3)
                    remaining = content_length
                    chunk_size = 64 * 1024

                    while remaining > 0:
                        chunk = stream.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        yield chunk
                        remaining -= len(chunk)
                elif hasattr(stream, 'chunks'):
                    # Azure Blob storage download stream
                    async for chunk in stream.chunks():
                        yield chunk
                else:
                    # Unexpected type, try to convert to bytes and yield
                    if stream:
                        yield stream
            except Exception as e:
                _logger.error(f"Error streaming from storage: {e}")
                # We can't raise HTTP exceptions here, so just stop the stream
                yield b''

    return StreamingResponse(
        file_streamer(),
        media_type="audio/mpeg",
        headers=headers,
        status_code=status_code
    )

@app.get("/story/{filename}", tags=["File Serving"])
async def serve_story_file(filename: str):
    """Serve story text files directly"""
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(app_config.storage.STORY_STORAGE_PATH, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Story file not found")

    if not is_file_in_directory(file_path, app_config.storage.STORY_STORAGE_PATH):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=filename
    )

@app.get("/storyboard/{filename}", tags=["File Serving"])
async def serve_storyboard_file(filename: str):
    """Serve storyboard JSON files directly"""
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(app_config.storage.STORYBOARD_STORAGE_PATH, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Storyboard file not found")

    if not is_file_in_directory(file_path, app_config.storage.STORYBOARD_STORAGE_PATH):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=filename
    )

# Job Management Endpoints
@app.post("/api/generate", response_model=JobResponse, tags=["Job Management"])
async def generate_story_from_urls(
    request: StoryGenerationRequest,
    background_tasks: BackgroundTasks,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Generate a viral story from the provided URLs.
    This endpoint queues the request for processing and returns a job ID.
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Queue the job for processing
        request_data = {
            "id": job_id,  # Include job_id in the request data
            "urls": [str(url) for url in request.urls],
            "topic": request.topic,
            "generate_audio": request.generate_audio,
            "temperature": request.temperature or app_config.llm.TEMPERATURE,
            "chunk_size": request.chunk_size or int(os.environ.get("LLM_CHUNK_SIZE", app_config.llm.CHUNK_SIZE))
        }

        queue_manager.add_request(request_data)

        # Start processing in the background
        background_tasks.add_task(process_story_generation, job_id, request_data)

        return JobResponse(
            job_id=job_id,
            message="Story generation job queued successfully."
        )

    except Exception as e:
        _logger.error(f"Error queueing job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

@app.get("/api/status/{job_id}", response_model=JobStatusResponse, tags=["Job Management"])
async def get_job_status(
    job_id: str,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Get the status of a story generation job.
    Returns the result if the job has completed.
    """
    try:
        if not queue_manager.is_available():
            raise HTTPException(status_code=503, detail="Redis service unavailable")

        # Try to get the result for the job
        result = queue_manager.get_result(job_id)

        if not result:
            # Check if the job exists by checking for any record with this ID
            key_exists = queue_manager.check_key_exists(job_id)

            if not key_exists:
                # The job doesn't exist at all
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            # Job exists but is still processing
            return {"status": "pending", "message": "Job is still processing"}

        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check job status: {str(e)}")

@queue_router.get("/status")
async def get_queue_status(
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Get the current status of the queue system.
    Returns metrics about queue length, processing jobs, and recent failures.
    """
    if not queue_manager.is_available():
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    try:
        # Get main queue length
        queue_length = queue_manager.get_queue_length()

        # Get processing queue length
        processing_queue = f"{queue_manager.queue_name}_processing"
        processing_length = queue_manager.client.llen(processing_queue)

        # Get statistics on recent jobs
        recent_jobs = []
        keys = queue_manager.client.keys(f"{queue_manager.result_prefix}*")
        recent_count = min(10, len(keys))  # Show up to 10 recent jobs

        for key in keys[:recent_count]:
            job_data = queue_manager.client.get(key)
            if job_data:
                job_info = json.loads(job_data)
                job_id = key.replace(queue_manager.result_prefix, "")
                recent_jobs.append({
                    "job_id": job_id,
                    "status": job_info.get("status", "unknown"),
                    "created_at": job_info.get("created_at"),
                    "updated_at": job_info.get("updated_at")
                })

        return {
            "status": "available",
            "queue_length": queue_length,
            "processing_jobs": processing_length,
            "recent_jobs": recent_jobs
        }
    except Exception as e:
        _logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@queue_router.post("/clear-stalled")
async def clear_stalled_jobs(
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Clear stalled jobs from the processing queue.
    This is useful when jobs have crashed or timed out and remained in the processing queue.
    """
    if not queue_manager.is_available():
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    try:
        processing_queue = f"{queue_manager.queue_name}_processing"
        stalled_jobs = queue_manager.client.lrange(processing_queue, 0, -1)

        cleared_count = 0
        recovered_count = 0

        # Process each potentially stalled job
        for job_data in stalled_jobs:
            try:
                job = json.loads(job_data)
                job_id = job.get("id")

                # Check if this job has been in the processing queue too long (over 10 minutes)
                # This is a basic implementation - you could enhance this with actual timestamp tracking
                result = queue_manager.get_result(job_id)
                if result and result.get("updated_at"):
                    job_time = result.get("updated_at")
                    current_time = time.time()

                    # If job has been processing for more than 10 minutes, consider it stalled
                    if current_time - job_time > 600:  # 10 minutes in seconds
                        # Remove from processing queue
                        queue_manager.client.lrem(processing_queue, 0, job_data)

                        # Option: re-queue the job for another attempt
                        # Uncomment this section to enable job recovery
                        # queue_manager.client.lpush(queue_manager.queue_name, job_data)
                        # recovered_count += 1

                        # Update the job status to failed
                        queue_manager.store_result(job_id, {
                            "status": "failed",
                            "error": "Job stalled and was cleared from processing queue",
                            "created_at": result.get("created_at", time.time()),
                            "updated_at": time.time()
                        })

                        cleared_count += 1
            except Exception as job_error:
                _logger.error(f"Error processing stalled job: {str(job_error)}")

        return {
            "message": f"Cleared {cleared_count} stalled jobs, recovered {recovered_count} for reprocessing",
            "cleared": cleared_count,
            "recovered": recovered_count
        }
    except Exception as e:
        _logger.error(f"Error clearing stalled jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear stalled jobs: {str(e)}")

@queue_router.delete("/purge")
async def purge_queue(
    confirmation: str,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Purge all jobs from the queue system.
    This is a destructive operation and requires a confirmation code.
    """
    if not queue_manager.is_available():
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    # Safety check: require explicit confirmation
    if confirmation != "CONFIRM_PURGE_ALL_JOBS":
        raise HTTPException(status_code=400, detail="Invalid confirmation code. Use 'CONFIRM_PURGE_ALL_JOBS' to confirm this destructive action.")

    try:
        # Clear main queue
        queue_length = queue_manager.get_queue_length()
        queue_manager.client.delete(queue_manager.queue_name)

        # Clear processing queue
        processing_queue = f"{queue_manager.queue_name}_processing"
        processing_length = queue_manager.client.llen(processing_queue)
        queue_manager.client.delete(processing_queue)

        # You might want to keep job results for historical purposes
        # Uncomment this section to also delete job results
        # result_keys = queue_manager.client.keys(f"{queue_manager.result_prefix}*")
        # if result_keys:
        #     queue_manager.client.delete(*result_keys)
        # deleted_results = len(result_keys) if result_keys else 0

        return {
            "message": f"Queue system purged successfully",
            "queued_jobs_removed": queue_length,
            "processing_jobs_removed": processing_length,
            # "result_records_removed": deleted_results
        }
    except Exception as e:
        _logger.error(f"Error purging queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to purge queue: {str(e)}")

@queue_router.get("/job/{job_id}/retry")
async def retry_failed_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Retry a failed job by re-queuing it for processing
    """
    if not queue_manager.is_available():
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    try:
        # Get the original job data
        result = queue_manager.get_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if result.get("status") != "failed":
            raise HTTPException(status_code=400, detail=f"Only failed jobs can be retried. Current status: {result.get('status')}")

        # Create a new job from the original data
        original_data = result.get("original_request")
        if not original_data:
            raise HTTPException(status_code=400, detail="Cannot retry job: missing original request data")

        # Generate a new job ID or reuse the existing one
        new_job_id = job_id  # Or use str(uuid.uuid4()) for a completely new job

        # Prepare request data
        request_data = {
            "id": new_job_id,
            **original_data
        }

        # Add to queue
        success = queue_manager.add_request(request_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to queue job for retry")

        # Start processing in the background
        background_tasks.add_task(process_story_generation, new_job_id, request_data)

        return {
            "message": f"Job {job_id} has been requeued for processing",
            "job_id": new_job_id
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error retrying job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {str(e)}")

def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server with uvicorn"""
    _logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("viralStoryGenerator.src.api:app", host=host, port=port, reload=reload)
