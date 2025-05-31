# viralStoryGenerator/src/api.py
"""
HTTP API backend for ViralStoryGenerator.
"""
import datetime
import uuid
import time
import os
from typing import List, Dict, Any, Optional, Tuple, Union, BinaryIO, TYPE_CHECKING, AsyncIterator # Added AsyncIterator
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import redis
import tempfile
from urllib.parse import urlparse

import uvicorn
import aiofiles
import asyncio

if TYPE_CHECKING:
    from botocore.response import StreamingBody

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
    SingleQueueStatusResponse,
    ContentDetailItem
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
    version=app_config.VERSION
)

# Include the router in the app
app.include_router(router)

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
if "*" in app_config.http.CORS_ORIGINS and app_config.ENVIRONMENT == "production":
    _logger.critical("CRITICAL SECURITY WARNING: CORS_ORIGINS allows '*' in production environment!")
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
            if not is_allowed:
                oldest_requests = self.redis.zrange(rate_key, 0, 0, withscores=True)
                if oldest_requests:
                    oldest_ts = oldest_requests[0][1]
                    retry_after = int(window - (current_time - oldest_ts) + 1)
                else:
                    retry_after = window
            return is_allowed, request_count, limit, retry_after

        # Fallback to local memory if Redis is unavailable
        if rate_key not in self.local_cache:
            self.local_cache[rate_key] = []

        # Remove expired timestamps
        self.local_cache[rate_key] = [ts for ts in self.local_cache[rate_key] if ts > current_time - window]

        self.local_cache[rate_key].append(current_time)
        request_count = len(self.local_cache[rate_key])

        is_allowed = request_count <= limit
        if not is_allowed:
            if self.local_cache[rate_key]:
                oldest_ts = self.local_cache[rate_key][0]
                retry_after = int(window - (current_time - oldest_ts) + 1)
            else:
                retry_after = window
        return is_allowed, request_count, limit, retry_after

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

    limit_for_header: Optional[int] = None
    current_for_header: Optional[int] = None

    if app_config.http.RATE_LIMIT_ENABLED:
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path

        # Skip rate limiting for certain endpoints
        if endpoint in ['/health', '/metrics', '/openapi.json'] or endpoint.startswith("/static"):
            pass
        else:
            is_allowed, current_requests, actual_limit, retry_after_seconds = await rate_limiter.check_rate_limit(client_ip, endpoint)

            limit_for_header = actual_limit
            current_for_header = current_requests

            if not is_allowed:
                RATE_LIMIT_HIT.labels(client_ip=client_ip).inc()
                _logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}: {current_requests}/{actual_limit}")

                headers_429 = {"Retry-After": str(retry_after_seconds)} if retry_after_seconds > 0 else {}
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"Rate limit exceeded. Please try again after {retry_after_seconds} seconds."},
                    headers=headers_429
                )

    response = await call_next(request)

    if app_config.http.RATE_LIMIT_ENABLED and limit_for_header is not None and current_for_header is not None:
        response.headers["X-RateLimit-Limit"] = str(limit_for_header)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit_for_header - current_for_header))

    _logger.debug(f"Rate limit middleware completed for {request.method} {request.url.path}")
    return response

# Middleware for metrics and request logging
@app.middleware("http")
async def metrics_logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    _logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Redis queue for API requests
API_QUEUE_NAME = "api_requests"
RESULT_PREFIX = "api_result:"

# Redis queue manager dependency
async def get_queue_manager() -> RedisMessageBroker:
    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}/{app_config.redis.DB}"
        if app_config.redis.PASSWORD:
            redis_url = f"redis://:{app_config.redis.PASSWORD}@{app_config.redis.HOST}:{app_config.redis.PORT}/{app_config.redis.DB}"

        manager = RedisMessageBroker(
            redis_url=redis_url,
            stream_name=API_QUEUE_NAME
        )

        await manager.initialize()

        try:
            QUEUE_SIZE.set(await manager.get_stream_length())
        except:
            pass

        return manager
    except Exception as e:
        _logger.error(f"Failed to initialize Redis queue: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis queue service unavailable")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _logger.exception(f"Unhandled exception caught by global handler for {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "detail": str(exc)},
    )


# Setup API security if enabled in config
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Authentication dependency
async def get_api_key(request: Request, api_key: str = Depends(api_key_header)):
    if not app_config.http.API_KEY_ENABLED:
        return None

    if api_key != app_config.http.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

# Health and Metrics Endpoints
@app.get("/metrics", tags=["Health and Metrics"])
async def get_metrics():
    """
    Endpoint to expose Prometheus metrics.
    This is used by Prometheus to scrape metrics from the application.
    """
    _logger.debug("Metrics endpoint called.")
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health", response_model=HealthResponse, tags=["Health and Metrics"])
async def health_check():
    """Health check endpoint for monitoring"""
    _logger.debug("Health check endpoint called.")
    service_status = await get_service_status()
    _logger.debug("Health check response generated.")
    return service_status

# Story Management Endpoints
@app.post("/stories", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED, tags=["Story Generation"], dependencies=[Depends(get_api_key)])
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

    sanitized_topic = sanitize_input(topic, max_length=500)
    if not sanitized_topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty or contain only invalid characters.")

    validated_sources_folder = None
    if sources_folder:
        if not validate_path_component(sources_folder):
             _logger.warning(f"Security: Invalid characters or structure in sources_folder: {sources_folder}")
             raise HTTPException(status_code=400, detail="Invalid sources folder name.")

        base_sources_path = os.path.abspath(app_config.security.SOURCE_MATERIALS_PATH)
        full_path = os.path.abspath(os.path.join(base_sources_path, sources_folder))

        if not os.path.exists(full_path) or not os.path.isdir(full_path):
             _logger.warning(f"Sources folder not found: {full_path}")
             raise HTTPException(status_code=404, detail=f"Sources folder '{sources_folder}' not found.")

        if not is_file_in_directory(full_path, base_sources_path):
             _logger.critical(f"SECURITY BREACH ATTEMPT: Access denied for sources_folder outside allowed path: {full_path}")
             raise HTTPException(status_code=403, detail="Access to the specified folder is forbidden.")
        validated_sources_folder = sources_folder

    if voice_id and not is_valid_voice_id(voice_id):
        _logger.warning(f"Invalid voice ID format received: {voice_id}")
        raise HTTPException(status_code=400, detail="Invalid voice ID format.")

    task = create_story_task(topic, sources_folder, voice_id)
    _logger.debug(f"Story generation task created for topic: {topic}")
    return task

@app.get("/api/stories/{task_id}", response_model=JobStatusResponse, tags=["Story Management"], dependencies=[Depends(get_api_key)])
async def check_story_status(
    task_id: str,
    queue_manager: RedisMessageBroker = Depends(get_queue_manager)
):
    _logger.debug(f"Check story status endpoint called for task_id: {task_id}")

    try:
        uuid.UUID(task_id, version=4)
    except ValueError:
        _logger.warning(f"Invalid task_id format: {task_id}. Must be a valid UUID v4.")
        raise HTTPException(status_code=400, detail="Invalid task_id format. Must be a valid UUID v4.")

    try:
        job_progress = await queue_manager.get_job_progress(task_id)

        status_val = "pending"
        message = "Job status not yet available or job not found."
        story_details: Optional[List[ContentDetailItem]] = None
        storyboard_details: Optional[List[ContentDetailItem]] = None
        audio_url_resp: Optional[str] = None
        sources_resp: Optional[List[str]] = None
        error_message: Optional[str] = None
        created_at_resp: Optional[str] = None
        updated_at_resp: Optional[str] = None
        processing_time: Optional[float] = None

        if not job_progress:
            _logger.info(f"No job progress found for task_id: {task_id}. Returning pending status.")
        else:
            _logger.info(f"Job progress for {task_id}: {job_progress}")
            status_val = job_progress.get("status", "unknown")
            data = job_progress.get("data", {})
            created_at_resp = job_progress.get("created_at")
            updated_at_resp = job_progress.get("last_updated")
            processing_time = job_progress.get("processing_time_seconds")

            if status_val == "completed":
                message = "Job completed successfully."
                if isinstance(data, dict):
                    story_content = data.get("story_script")
                    story_url = data.get("story_url")
                    if story_content or story_url:
                        story_details = [ContentDetailItem(url=story_url, content=story_content)]

                    storyboard_content = data.get("storyboard_details")
                    storyboard_url = data.get("storyboard_url")
                    if storyboard_content or storyboard_url:
                        storyboard_details = [ContentDetailItem(url=storyboard_url, content=storyboard_content)]

                    audio_url_resp = data.get("audio_url")
                    sources_resp = data.get("sources")
                else:
                    _logger.warning(f"Task {task_id} is 'completed' but 'data' field is not a dictionary: {data}")
                    status_val = "failed"
                    error_message = "Completed job has malformed data."
                    message = "Job data is inconsistent."

            elif status_val == "processing":
                progress_detail = job_progress.get("progress_detail", "Processing...")
                message = f"Job is currently processing. Detail: {progress_detail}"
            elif status_val == "failed":
                error_message = job_progress.get("error", "Job failed due to an unknown error.")
                message = f"Job failed: {error_message}"
            elif status_val == "pending":
                message = "Job is pending and has not started processing yet."
            else:
                message = f"Job is in an '{status_val}' state."

        return JobStatusResponse(
            job_id=task_id,
            status=status_val,
            message=message,
            story=story_details,
            storyboard=storyboard_details,
            audio_url=audio_url_resp,
            sources=sources_resp,
            error=error_message,
            created_at=created_at_resp,
            updated_at=updated_at_resp,
            processing_time_seconds=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error retrieving status for task {task_id}: {str(e)}", exc_info=True)
        return JobStatusResponse(
            job_id=task_id,
            status="failed",
            message="Failed to retrieve job status due to an internal server error.",
            error=str(e)
        )

@app.get("/api/stories/{task_id}/download/{file_type}", dependencies=[Depends(get_api_key)], tags=["Story Management"])
async def download_story_file(task_id: str, file_type: str, queue_manager: RedisMessageBroker = Depends(get_queue_manager)):
    """
    Download a generated file from a completed story task

    Parameters:
    - task_id: ID of the task
    - file_type: Type of file to download (story, audio, storyboard)

    Returns:
    - File download response
    """
    FILE_TYPE_DETAILS = {
        "story": {"path_config": app_config.storage.STORY_STORAGE_PATH, "media_type": "text/plain"},
        "audio": {"path_config": app_config.storage.AUDIO_STORAGE_PATH, "media_type": "audio/mpeg"},
        "storyboard": {"path_config": app_config.storage.STORYBOARD_STORAGE_PATH, "media_type": "application/json"}
    }

    if not is_valid_uuid(task_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task_id format.")

    if file_type not in FILE_TYPE_DETAILS:
        allowed_keys = ", ".join(FILE_TYPE_DETAILS.keys())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file_type. Allowed types are: {allowed_keys}")

    task_info_response = await check_story_status(task_id, queue_manager)

    if not task_info_response: # Should not happen if check_story_status always returns JobStatusResponse
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or status unavailable.")

    if task_info_response.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not completed. Status: {task_info_response.status}")

    file_detail = FILE_TYPE_DETAILS[file_type]
    expected_directory = file_detail["path_config"]
    media_type = file_detail["media_type"]

    filename = None
    file_path = None # Only used for local storage

    if file_type == "audio":
        if task_info_response.audio_url:
            if app_config.storage.PROVIDER == "local":
                parsed_url = urlparse(task_info_response.audio_url)
                if parsed_url.path.startswith("/static/"):
                    filename = os.path.basename(parsed_url.path)
                    file_path = os.path.join(expected_directory, filename)
                elif not parsed_url.scheme and not parsed_url.netloc:
                    filename = os.path.basename(task_info_response.audio_url)
                    file_path = os.path.join(expected_directory, filename)
                else:
                    raise HTTPException(status_code=501, detail=f"Cannot directly download external or unresolvable local {file_type} URL: {task_info_response.audio_url}")
            else: # Cloud
                # Assume audio_url might be the full public URL or contain the filename/key
                # For retrieve_file, we need the filename/key.
                # This logic needs to be robust based on how URLs/keys are stored.
                # A common pattern: filename is task_id + extension.
                filename = f"{task_id}.mp3" # Placeholder - needs reliable determination
        else:
            raise HTTPException(status_code=404, detail=f"Audio URL not found for task {task_id}.")

    elif file_type == "story":
        if task_info_response.story and task_info_response.story[0].url:
             # Prefer URL if available, extract filename if local, or use as key if cloud
            if app_config.storage.PROVIDER == "local":
                parsed_url = urlparse(task_info_response.story[0].url)
                if parsed_url.path.startswith("/static/"): # e.g. /static/story/task_id.txt
                    filename = os.path.basename(parsed_url.path)
                    file_path = os.path.join(expected_directory, filename)
                else: # Assume it's just the filename if not a full static path
                    filename = os.path.basename(task_info_response.story[0].url)
                    file_path = os.path.join(expected_directory, filename)
            else: # Cloud - URL might be public, or path is the key
                filename = os.path.basename(urlparse(task_info_response.story[0].url).path) # Extract from URL path
                if not filename: filename = f"{task_id}.txt" # Fallback
        elif task_info_response.story and task_info_response.story[0].content: # Fallback to generating filename if no URL
            filename = f"{task_id}.txt"
            if app_config.storage.PROVIDER == "local":
                 file_path = os.path.join(expected_directory, filename)
                 # Note: This implies the file must exist if content was in response.
                 # Worker should save it with this convention if only content is provided.
        else:
            raise HTTPException(status_code=404, detail=f"Story content/URL not found for task {task_id}.")

    elif file_type == "storyboard":
        if task_info_response.storyboard and task_info_response.storyboard[0].url:
            if app_config.storage.PROVIDER == "local":
                parsed_url = urlparse(task_info_response.storyboard[0].url)
                if parsed_url.path.startswith("/static/"):
                    filename = os.path.basename(parsed_url.path)
                    file_path = os.path.join(expected_directory, filename)
                else:
                    filename = os.path.basename(task_info_response.storyboard[0].url)
                    file_path = os.path.join(expected_directory, filename)
            else: # Cloud
                filename = os.path.basename(urlparse(task_info_response.storyboard[0].url).path)
                if not filename: filename = f"{task_id}.json" # Fallback
        elif task_info_response.storyboard and task_info_response.storyboard[0].content:
            filename = f"{task_id}.json"
            if app_config.storage.PROVIDER == "local":
                file_path = os.path.join(expected_directory, filename)
        else:
            raise HTTPException(status_code=404, detail=f"Storyboard content/URL not found for task {task_id}.")

    if not filename:
        raise HTTPException(status_code=404, detail=f"Could not determine filename for {file_type} of task {task_id}.")

    if not is_safe_filename(filename): # Final safety check on derived filename
        _logger.warning(f"Security: Unsafe filename derived: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename derived.")

    if app_config.storage.PROVIDER == "local":
        if not file_path or not os.path.exists(file_path) or not os.path.isfile(file_path):
            _logger.error(f"Local file not found for task {task_id}, type {file_type}. Expected at: {file_path}. Task info: {task_info_response}")
            raise HTTPException(status_code=404, detail=f"File not found on server for {file_type}.")

        if not is_file_in_directory(file_path, expected_directory):
            _logger.warning(f"Security: Attempted access to file outside storage directory: {file_path} (expected under {expected_directory})")
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(path=file_path, media_type=media_type, filename=filename)
    else: # Cloud storage
        try:
            # storage_manager.retrieve_file expects filename (key) and file_type (for path prefixing if needed by manager)
            file_content_stream = storage_manager.retrieve_file(filename=filename, file_type=file_type)
            if file_content_stream is None:
                _logger.error(f"Cloud file not found: {filename}, type: {file_type} for task {task_id}. Task info: {task_info_response}")
                raise HTTPException(status_code=404, detail=f"File not found in cloud storage for {file_type}.")

            # For non-streaming types like bytes, wrap in a simple iterator for StreamingResponse
            async def _stream_bytes_if_needed(content):
                if isinstance(content, bytes):
                    yield content
                elif hasattr(content, '__aiter__'): # Azure async stream
                    async for chunk in content:
                        yield chunk
                elif hasattr(content, 'read'): # S3 StreamingBody or other file-like
                    while True:
                        chunk = await asyncio.to_thread(content.read, 64 * 1024)
                        if not chunk:
                            break
                        yield chunk
                    if hasattr(content, 'close') and callable(getattr(content, 'close')):
                         await asyncio.to_thread(content.close)
                else: # Should not happen if retrieve_file returns known types
                    _logger.error(f"Unexpected content type from storage_manager: {type(content)}")
                    yield b''


            return StreamingResponse(_stream_bytes_if_needed(file_content_stream), media_type=media_type, headers={'Content-Disposition': f'attachment; filename="{filename}"'})
        except Exception as e:
            _logger.error(f"Error retrieving file from cloud storage: {filename}, type: {file_type}. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving file from storage: {str(e)}")

# File Serving Endpoints
@app.get("/audio/{filename}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def serve_audio_file(filename: str):
    """Serve audio files directly from local storage."""
    if not is_safe_filename(filename): # Initial quick check
        raise HTTPException(status_code=400, detail="Invalid filename")

    if app_config.storage.PROVIDER != "local":
        raise HTTPException(status_code=501,
                            detail=f"Direct file serving via /audio/ is only supported for 'local' storage provider. Current provider: {app_config.storage.PROVIDER}. Use /api/audio/stream/ for other providers.")

    try:
        # For local provider, storage_manager.serve_file returns a validated path string
        local_file_path = storage_manager.serve_file(filename=filename, file_type="audio")

        if not isinstance(local_file_path, str): # Should be guaranteed by serve_file for local
             _logger.error(f"Storage manager's serve_file (local) did not return a string path for {filename}. Got: {type(local_file_path)}")
             raise HTTPException(status_code=500, detail="Internal server error: Storage manager misconfiguration.")

        # The path from storage_manager.serve_file (via _get_validated_local_path)
        # should already be checked for existence and safety.
        return FileResponse(
            path=local_file_path,
            media_type="audio/mpeg",
            filename=filename
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied.")
    except ValueError as ve: # e.g. from is_safe_filename inside storage_manager
        _logger.warning(f"ValueError from storage_manager.serve_file for {filename}: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        _logger.exception(f"Unexpected error serving local audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/api/audio/stream/{filename}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def stream_audio(
    filename: str,
    range_header: Optional[str] = Depends(lambda request: request.headers.get("range")) # Correctly get range header
):
    """
    Stream audio file with support for range requests (needed for seeking in audio players)
    """
    if not is_safe_filename(filename): # Initial check
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_size: Optional[int] = None
    local_file_path_for_streaming: Optional[str] = None
    retrieved_content_holder: Optional[Union[bytes, BinaryIO, "StreamingBody", AsyncIterator[bytes]]] = None
    temp_file_to_clean: Optional[str] = None

    start: int = 0
    end: int = -1
    status_code_resp: int = 200

    try:
        if app_config.storage.PROVIDER == "local":
            try:
                resolved_path = storage_manager.serve_file(filename=filename, file_type="audio")
                if not isinstance(resolved_path, str):
                    _logger.error(f"Storage manager (local) did not return path string for {filename}")
                    raise HTTPException(status_code=500, detail="Storage configuration error for local files.")

                file_size = await asyncio.to_thread(os.path.getsize, resolved_path)
                local_file_path_for_streaming = resolved_path
            except FileNotFoundError:
                _logger.warning(f"Local audio file not found via storage_manager: {filename}")
                raise HTTPException(status_code=404, detail="Audio file not found.")
            except PermissionError:
                _logger.warning(f"Permission denied for local audio file via storage_manager: {filename}")
                raise HTTPException(status_code=403, detail="Access denied to audio file.")
            except ValueError as ve:
                _logger.warning(f"ValueError for local audio file from storage_manager: {filename} - {ve}")
                raise HTTPException(status_code=400, detail=f"Invalid filename or path: {ve}")
            except Exception as e:
                _logger.exception(f"Error accessing local file {filename} info via storage_manager: {e}")
                raise HTTPException(status_code=500, detail="Error accessing file information.")

        else:  # Cloud providers (S3, Azure)
            retrieved_content = storage_manager.retrieve_file(filename=filename, file_type="audio")

            if retrieved_content is None:
                _logger.warning(f"Cloud audio file not found or access issue: {filename} (Provider: {app_config.storage.PROVIDER})")
                raise HTTPException(status_code=404, detail=f"{app_config.storage.PROVIDER.upper()} audio file not found or access issue.")

            retrieved_content_holder = retrieved_content

            if hasattr(retrieved_content_holder, 'content_length'): # S3 StreamingBody with Boto3
                file_size = getattr(retrieved_content_holder, 'content_length', None)
            elif hasattr(retrieved_content_holder, 'response_metadata') and 'ContentLength' in getattr(retrieved_content_holder, 'response_metadata', {}).get('HTTPHeaders', {}): # Fallback for S3
                 file_size = int(getattr(retrieved_content_holder, 'response_metadata')['HTTPHeaders']['ContentLength'])
            elif hasattr(retrieved_content_holder, 'size'): # Azure BlobClient download stream
                 file_size = getattr(retrieved_content_holder, 'size', None)
            elif isinstance(retrieved_content_holder, bytes):
                file_size = len(retrieved_content_holder)
                # Ensure tmp_file.name is treated as str for type consistency if linters are strict
                # aiofiles.tempfile.NamedTemporaryFile().name is str.
                async with aiofiles.tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=".mp3") as tmp_file:
                    await tmp_file.write(retrieved_content_holder)
                    local_file_path_for_streaming = str(tmp_file.name)
                    temp_file_to_clean = str(tmp_file.name)
                _logger.debug(f"Cloud content (bytes) written to temporary file: {local_file_path_for_streaming}")

        if file_size is not None:
            end = file_size - 1

        if range_header:
            if file_size is not None:
                try:
                    range_parts = range_header.replace("bytes=", "").split("-")
                    start = int(range_parts[0]) if range_parts[0] else 0
                    parsed_end_val = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] and range_parts[1].isdigit() else (file_size - 1)
                    end = min(parsed_end_val, file_size - 1)

                    if end < start or start < 0:
                        raise ValueError("Invalid range: end before start or negative start.")

                    if start > 0 or end < (file_size - 1):
                        status_code_resp = 206
                    else:
                        status_code_resp = 200

                except ValueError:
                    _logger.warning(f"Invalid range header '{range_header}' for known file size. Resetting to full stream.")
                    start = 0
                    end = file_size - 1
                    status_code_resp = 200
            else:
                _logger.warning("Range header received, but file size from cloud is unknown. Attempting to honor range if possible.")
                try:
                    range_parts = range_header.replace("bytes=", "").split("-")
                    parsed_start = int(range_parts[0]) if range_parts[0] else 0
                    parsed_end = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] and range_parts[1].isdigit() else -1

                    start = parsed_start
                    if start < 0: raise ValueError("Invalid range: negative start")
                    end = parsed_end

                    if start > 0 or (end != -1):
                        status_code_resp = 206
                    else:
                        status_code_resp = 200

                except ValueError:
                    _logger.warning("Invalid range header format when file size is unknown. Streaming fully from start 0.")
                    start = 0
                    end = -1
                    status_code_resp = 200

    except HTTPException:
        raise
    except FileNotFoundError:
        _logger.warning(f"Audio file not found (stream_audio main try): {filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    except PermissionError:
        _logger.warning(f"Permission denied for audio file (stream_audio main try): {filename}")
        raise HTTPException(status_code=403, detail="Access denied to audio file.")
    except ValueError as ve:
        _logger.warning(f"ValueError during audio stream preparation for {filename}: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        _logger.exception(f"Unexpected error preparing audio stream for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error preparing audio stream: {str(e)}")

    content_length_val: Optional[int] = None
    if file_size is not None and end != -1 and start <= end:
        content_length_val = end - start + 1
    elif file_size is None and end != -1 and start <= end : # Range specified, but total size unknown
        content_length_val = end - start + 1


    headers_resp = {"Accept-Ranges": "bytes"}
    if file_size is not None: # Total size known
        headers_resp["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        if content_length_val is not None: # This should always be true if file_size is not None and range is valid
             headers_resp["Content-Length"] = str(content_length_val)
    elif end != -1 and start <= end : # Range specified (e.g. 0-1023 or 1024-2047), but total size unknown
        headers_resp["Content-Range"] = f"bytes {start}-{end}/*"
        if content_length_val is not None:
             headers_resp["Content-Length"] = str(content_length_val)
    # If file_size is None and no specific end range (end == -1), Content-Length is not set, client figures it out.

    headers_resp["Cache-Control"] = "public, max-age=3600"

    async def file_streamer_gen():
        nonlocal local_file_path_for_streaming
        nonlocal retrieved_content_holder
        nonlocal temp_file_to_clean # Declare nonlocal to modify the outer scope variable

        chunk_size = 64 * 1024

        # This first block handles:
        # 1. Local provider files.
        # 2. Cloud provider files that were 'bytes' and got written to a temporary local file.
        if local_file_path_for_streaming and os.path.exists(local_file_path_for_streaming):
            async with aiofiles.open(local_file_path_for_streaming, "rb") as f:
                await f.seek(start)

                # Determine total bytes to stream for this request
                # If content_length_val is set, that's the exact amount.
                # Otherwise, if file_size is known, stream from 'start' to end of file.
                # If file_size is also None (should not happen if local_file_path_for_streaming is set), stream all.
                to_read_total = content_length_val
                if to_read_total is None and file_size is not None:
                    to_read_total = file_size - start

                bytes_streamed = 0
                while True:
                    # Calculate how much to read in this chunk
                    current_chunk_read_limit = chunk_size
                    if to_read_total is not None: # If we know the total amount to read for this request
                        read_amount = min(chunk_size, to_read_total - bytes_streamed)
                        if read_amount <= 0: break # Nothing more to read for this request
                        current_chunk_read_limit = read_amount

                    chunk = await f.read(current_chunk_read_limit)
                    if not chunk: break # End of file
                    yield chunk
                    bytes_streamed += len(chunk)

            # Clean up temporary file if it was created for cloud 'bytes' content
            if temp_file_to_clean and temp_file_to_clean == local_file_path_for_streaming:
                try:
                    await asyncio.to_thread(os.remove, temp_file_to_clean)
                    _logger.debug(f"Successfully removed temp file: {temp_file_to_clean}")
                    # Reset to prevent re-processing or issues if generator is somehow re-entered (though it shouldn't)
                    local_file_path_for_streaming = None
                    temp_file_to_clean = None # Modified here
                except Exception as e_remove:
                    _logger.warning(f"Could not remove temp file {temp_file_to_clean}: {e_remove}")

        # This block handles cloud provider streams that are not raw bytes (i.e., S3 StreamingBody, Azure AsyncIterator)
        # It's an 'elif' because 'bytes' from cloud would have set 'local_file_path_for_streaming'.
        elif app_config.storage.PROVIDER != "local" and retrieved_content_holder is not None:
            cloud_stream_object = retrieved_content_holder
            try:
                # Case 1: Azure-style AsyncIterator
                if hasattr(cloud_stream_object, '__aiter__'):
                    _logger.debug(f"Streaming from AsyncIterator for {filename}")
                    bytes_to_skip_at_start = start
                    bytes_yielded_count = 0
                    # content_length_val here is the number of bytes to yield *after* skipping 'start' bytes.
                    # 'end' in this context is the original requested end byte offset from file start.

                    # If file_size is unknown and a specific range like "bytes=100-200" is given,
                    # content_length_val will be 101. We need to read 'start' bytes, then 'content_length_val' bytes.
                    # If range is "bytes=100-", content_length_val is None (if file_size unknown).

                    current_pos = 0 # Tracks bytes iterated from the beginning of the actual cloud stream

                    async for chunk_data in cloud_stream_object: # type: ignore
                        chunk_len = len(chunk_data)

                        # If we haven't skipped enough initial bytes yet
                        if current_pos < bytes_to_skip_at_start:
                            if current_pos + chunk_len <= bytes_to_skip_at_start:
                                current_pos += chunk_len
                                continue # Skip this whole chunk
                            else: # This chunk contains the boundary
                                bytes_to_discard_in_chunk = bytes_to_skip_at_start - current_pos
                                chunk_data = chunk_data[bytes_to_discard_in_chunk:]
                                chunk_len = len(chunk_data) # Update chunk_len after slicing
                                current_pos = bytes_to_skip_at_start # We are now at the desired start position
                                if not chunk_data: continue # Entire chunk was part of the skip

                        # Now, current_pos is effectively >= bytes_to_skip_at_start (or 0 if start was 0)
                        # We are in the range of bytes to yield.

                        # If content_length_val is defined (meaning a specific amount of bytes to yield after start)
                        if content_length_val is not None:
                            remaining_bytes_to_yield = content_length_val - bytes_yielded_count
                            if remaining_bytes_to_yield <= 0: break # Yielded enough

                            if chunk_len > remaining_bytes_to_yield:
                                chunk_data = chunk_data[:remaining_bytes_to_yield]
                                chunk_len = len(chunk_data) # Update after slicing

                        yield chunk_data
                        bytes_yielded_count += chunk_len
                        current_pos += chunk_len # Keep track of overall position for next iteration's skip logic (if any)

                        if content_length_val is not None and bytes_yielded_count >= content_length_val:
                            break # Done yielding the requested amount

                # Case 2: S3-style readable synchronous stream
                elif hasattr(cloud_stream_object, 'read') and callable(getattr(cloud_stream_object, 'read')):
                    _logger.debug(f"Streaming from readable sync stream for {filename}")
                    # For sync streams, storage_manager.retrieve_file might have already handled byte range requests
                    # if start_byte/end_byte were passed to it.
                    # However, our current retrieve_file in stream_audio does not pass range, so we handle it here.

                    # Skip 'start' bytes if the stream itself doesn't support seek or ranged reads directly
                    # This is inefficient as it reads and discards.
                    # TODO: Enhance storage_manager.retrieve_file to accept range for S3/Azure to optimize.
                    if start > 0:
                        bytes_skipped = 0
                        while bytes_skipped < start:
                            skip_amount = min(chunk_size, start - bytes_skipped)
                            # Ensure cloud_stream_object is not bytes here
                            if isinstance(cloud_stream_object, (bytes, str)): # Should not happen
                                 _logger.error("Trying to read/skip on raw bytes/str in sync stream logic")
                                 break
                            chunk_to_skip = await asyncio.to_thread(cloud_stream_object.read, skip_amount) # type: ignore
                            if not chunk_to_skip: break # End of stream before skipping enough
                            bytes_skipped += len(chunk_to_skip)
                        if bytes_skipped < start:
                             _logger.warning(f"Could only skip {bytes_skipped}/{start} bytes for {filename}")
                             # Stream ends early, yield nothing more.
                             if hasattr(cloud_stream_object, 'close') and callable(getattr(cloud_stream_object, 'close')):
                                await asyncio.to_thread(cloud_stream_object.close) # type: ignore
                             return


                    bytes_to_yield_total = content_length_val # This is the amount to yield *after* 'start'
                    bytes_yielded_count = 0

                    while True:
                        current_chunk_read_limit = chunk_size
                        if bytes_to_yield_total is not None:
                            read_amount = min(chunk_size, bytes_to_yield_total - bytes_yielded_count)
                            if read_amount <= 0: break
                            current_chunk_read_limit = read_amount

                        # Ensure cloud_stream_object is not bytes here
                        if isinstance(cloud_stream_object, (bytes, str)): # Should not happen
                                _logger.error("Trying to read on raw bytes/str in sync stream logic")
                                break
                        try:
                            chunk = await asyncio.to_thread(cloud_stream_object.read, current_chunk_read_limit) # type: ignore
                        except Exception as read_exc:
                            _logger.error(f"Error reading from sync cloud stream {filename}: {read_exc}")
                            break
                        if not chunk: break
                        yield chunk
                        bytes_yielded_count += len(chunk)

                    if hasattr(cloud_stream_object, 'close') and callable(getattr(cloud_stream_object, 'close')):
                        await asyncio.to_thread(cloud_stream_object.close) # type: ignore
                else:
                    _logger.error(f"Cloud stream for {filename} is of an unhandled type in streamer: {type(cloud_stream_object)}")
                    yield b''
            except Exception as e_cloud_stream:
                _logger.error(f"Error streaming from cloud storage {filename}: {e_cloud_stream}", exc_info=True)
                yield b''
        else:
            _logger.error(f"No valid stream or file path available for streaming {filename}")
            yield b''


    return StreamingResponse(
        file_streamer_gen(),
        media_type="audio/mpeg",
        headers=headers_resp,
        status_code=status_code_resp
    )

@app.get("/story/{filename}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def serve_story_file(filename: str):
    """Serve story text files directly from local storage."""
    # Initial quick check, though storage_manager will also validate
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if app_config.storage.PROVIDER != "local":
        # This endpoint is specifically for direct local serving.
        raise HTTPException(status_code=501,
                            detail=f"Direct file serving via /story/ is only supported for 'local' storage provider. Current provider: {app_config.storage.PROVIDER}.")
    try:
        # storage_manager.serve_file for 'local' provider returns a validated absolute file path.
        # It handles safety checks (is_safe_filename, is_file_in_directory) and existence.
        local_file_path = storage_manager.serve_file(filename=filename, file_type="story")

        if not isinstance(local_file_path, str): # Should be guaranteed by serve_file for local
             _logger.error(f"Storage manager's serve_file (local) did not return a string path for story file {filename}. Got: {type(local_file_path)}")
             raise HTTPException(status_code=500, detail="Internal server error: Storage manager misconfiguration.")

        return FileResponse(
            path=local_file_path,
            media_type="text/plain",
            filename=filename # Original filename for download
        )
    except FileNotFoundError:
        _logger.info(f"Story file not found via storage_manager: {filename}")
        raise HTTPException(status_code=404, detail="Story file not found.")
    except PermissionError:
        _logger.warning(f"Permission denied for story file via storage_manager: {filename}")
        raise HTTPException(status_code=403, detail="Access denied to story file.")
    except ValueError as ve: # Catches validation errors from storage_manager
        _logger.warning(f"ValueError serving story file {filename} via storage_manager: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        _logger.exception(f"Unexpected error serving local story file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while serving story file.")

@app.get("/storyboard/{filename}", tags=["File Serving"], dependencies=[Depends(get_api_key)])
async def serve_storyboard_file(filename: str):
    """Serve storyboard JSON files directly from local storage."""
    # Initial quick check, though storage_manager will also validate
    if not is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if app_config.storage.PROVIDER != "local":
        raise HTTPException(status_code=501,
                            detail=f"Direct file serving via /storyboard/ is only supported for 'local' storage provider. Current provider: {app_config.storage.PROVIDER}.")
    try:
        # storage_manager.serve_file for 'local' provider returns a validated absolute file path.
        local_file_path = storage_manager.serve_file(filename=filename, file_type="storyboard")

        if not isinstance(local_file_path, str): # Should be guaranteed by serve_file for local
             _logger.error(f"Storage manager's serve_file (local) did not return a string path for storyboard file {filename}. Got: {type(local_file_path)}")
             raise HTTPException(status_code=500, detail="Internal server error: Storage manager misconfiguration.")

        return FileResponse(
            path=local_file_path,
            media_type="application/json",
            filename=filename # Original filename for download
        )
    except FileNotFoundError:
        _logger.info(f"Storyboard file not found via storage_manager: {filename}")
        raise HTTPException(status_code=404, detail="Storyboard file not found.")
    except PermissionError:
        _logger.warning(f"Permission denied for storyboard file via storage_manager: {filename}")
        raise HTTPException(status_code=403, detail="Access denied to storyboard file.")
    except ValueError as ve: # Catches validation errors from storage_manager
        _logger.warning(f"ValueError serving storyboard file {filename} via storage_manager: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        _logger.exception(f"Unexpected error serving local storyboard file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while serving storyboard file.")

# Job Management Endpoints
@app.post("/api/stories/generate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Story Management"])
async def generate_story_from_urls(
    request: StoryGenerationRequest,
    background_tasks: BackgroundTasks,
    queue_manager: RedisMessageBroker = Depends(get_queue_manager)
):
    """
    Generate a story from a list of URLs.
    This endpoint accepts a list of URLs and other parameters,
    queues the job, and returns a job ID.
    """
    job_id = str(uuid.uuid4())
    _logger.info(f"Received story generation request with job ID: {job_id}")

    try:
        request_data = {
            "id": job_id,
            "urls": [str(url) for url in request.urls],
            "topic": request.topic,
            "generate_audio": request.generate_audio,
            "temperature": request.temperature or app_config.llm.TEMPERATURE,
            "chunk_size": request.chunk_size or int(os.environ.get("LLM_CHUNK_SIZE", app_config.llm.CHUNK_SIZE)),
            "job_type": "generate_story"
        }

        # The worker expects job_type at the root of the payload, not just inside 'data'.
        # So we must send job_type at the root level in the message payload.
        # The RedisMessageBroker will wrap this in {job_id, data: ...} automatically.
        request_data["job_id"] = job_id  # Ensure job_id is present at the top level
        message_id = await queue_manager.publish_message(request_data, job_id=job_id)
        if not message_id:
            _logger.error(f"Failed to publish message to queue for job ID: {job_id}")
            raise HTTPException(status_code=500, detail="Failed to queue job due to messaging error.")

        return JobResponse(
            job_id=job_id,
            message="Story generation job queued successfully."
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error queueing job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

@app.get("/api/status/{job_id}", response_model=JobStatusResponse, tags=["Job Management"], dependencies=[Depends(get_api_key)])
async def get_job_status( # This is an old endpoint, check_story_status is the current one.
    job_id: str,
    queue_manager: RedisMessageBroker = Depends(get_queue_manager)
):
    """
    Get the status of a story generation job. (DEPRECATED - Use /api/stories/{task_id})
    Returns the result if the job has completed.
    """
    _logger.warning(f"Deprecated endpoint /api/status/{job_id} called. Redirecting or use /api/stories/{job_id}.")
    # Consider redirecting or just forwarding to the new function
    # return await check_story_status(task_id=job_id, queue_manager=queue_manager)
    # For now, keep original logic but mark as deprecated.
    try:
        job_progress = await queue_manager.get_job_progress(job_id)

        status = "pending"
        message = "Job status not yet available or job not found."
        story_details: Optional[List[ContentDetailItem]] = None
        storyboard_details: Optional[List[ContentDetailItem]] = None
        audio_url_resp: Optional[str] = None
        sources_resp: Optional[List[str]] = None
        error_message: Optional[str] = None
        created_at_resp: Optional[str] = None
        updated_at_resp: Optional[str] = None
        processing_time: Optional[float] = None

        if not job_progress:
            _logger.info(f"No job progress found for job_id: {job_id}. Returning pending status.")
        else:
            _logger.info(f"Job progress for {job_id}: {job_progress}")
            status = job_progress.get("status", "unknown")
            data = job_progress.get("data", {})
            created_at_resp = job_progress.get("created_at")
            updated_at_resp = job_progress.get("last_updated")
            processing_time = job_progress.get("processing_time_seconds")

            if status == "completed":
                message = "Job completed successfully."
                if isinstance(data, dict):
                    story_content = data.get("story_script")
                    story_url = data.get("story_url")
                    if story_content or story_url:
                        story_details = [ContentDetailItem(url=story_url, content=story_content)]

                    storyboard_content = data.get("storyboard_details")
                    storyboard_url = data.get("storyboard_url")
                    if storyboard_content or storyboard_url:
                        storyboard_details = [ContentDetailItem(url=storyboard_url, content=storyboard_content)]

                    audio_url_resp = data.get("audio_url")
                    sources_resp = data.get("sources")
                else:
                    _logger.warning(f"Job {job_id} is 'completed' but 'data' field is not a dictionary: {data}")
                    status = "failed"
                    error_message = "Completed job has malformed data."
                    message = "Job data is inconsistent."


            elif status == "processing":
                progress_detail = job_progress.get("progress_detail", "Processing...")
                message = f"Job is currently processing. Detail: {progress_detail}"
            elif status == "failed":
                error_message = job_progress.get("error", "Job failed due to an unknown error.")
                message = f"Job failed: {error_message}"
            elif status == "pending":
                message = "Job is pending and has not started processing yet."
            else:
                message = f"Job is in an '{status}' state."

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            message=message,
            story=story_details,
            storyboard=storyboard_details,
            audio_url=audio_url_resp,
            sources=sources_resp,
            error=error_message,
            created_at=created_at_resp,
            updated_at=updated_at_resp,
            processing_time_seconds=processing_time
        )

    except Exception as e:
        _logger.error(f"Error retrieving status for job {job_id}: {str(e)}", exc_info=True)
        # Ensure all fields of JobStatusResponse are provided or have defaults
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            message="Failed to retrieve job status due to an internal server error.",
            error=str(e),
            story=None, # Default
            storyboard=None, # Default
            audio_url=None, # Default
            sources=None, # Default
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat() if created_at_resp is None else created_at_resp, # Provide default if None
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), # Always update this on error
            processing_time_seconds=None # Default
        )
