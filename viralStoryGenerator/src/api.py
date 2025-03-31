"""
HTTP API backend for ViralStoryGenerator.
This module provides HTTP endpoints that replicate CLI functionality
but accept URLs instead of file paths.
"""
import asyncio
import uuid
import time
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, AnyHttpUrl, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import uvicorn
import redis

from ..utils.redis_manager import RedisManager as RedisQueueManager
from ..utils.storage_manager import storage_manager
from ..utils.crawl4ai_scraper import scrape_urls
from ..utils.config import config
from ..src.llm import process_with_llm
from ..src.source_cleanser import chunkify_and_summarize
from ..src.storyboard import generate_storyboard
from ..src.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.src.api_handlers import (
    create_story_task,
    get_task_status,
    process_story_task,
    process_audio_queue
)

router = APIRouter(
    prefix="",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.VERSION
)

app.include_router(router)

# Mount static file directory for local storage
os.makedirs(config.storage.LOCAL_STORAGE_PATH, exist_ok=True)
app.mount("/static", StaticFiles(directory=config.storage.LOCAL_STORAGE_PATH), name="static")

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Active requests')
QUEUE_SIZE = Gauge('api_queue_size', 'API queue size')
RATE_LIMIT_HIT = Counter('api_rate_limit_hit_total', 'Rate limit exceeded count', ['client_ip'])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.http.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client for rate limiting
redis_client = None
if config.http.RATE_LIMIT_ENABLED and config.redis.ENABLED:
    try:
        redis_client = redis.Redis(
            host=config.redis.HOST,
            port=config.redis.PORT,
            db=config.redis.DB,
            password=config.redis.PASSWORD,
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
        window = config.http.RATE_LIMIT_WINDOW
        limit = config.http.RATE_LIMIT_REQUESTS

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
    if config.http.RATE_LIMIT_ENABLED:
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
            retry_after = config.http.RATE_LIMIT_WINDOW

            # Custom rate limit exceeded response
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Maximum {limit} requests per {config.http.RATE_LIMIT_WINDOW} second window."
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
    if config.http.RATE_LIMIT_ENABLED:
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

# Models for request and response
class StoryGenerationRequest(BaseModel):
    urls: List[AnyHttpUrl] = Field(..., description="List of URLs to scrape for content")
    topic: str = Field(..., description="Topic for the story")
    generate_audio: bool = Field(False, description="Whether to generate audio")
    temperature: Optional[float] = Field(None, description="LLM temperature")
    chunk_size: Optional[int] = Field(None, description="Word chunk size for splitting sources")

class JobResponse(BaseModel):
    job_id: str = Field(..., description="Job ID for tracking progress")
    message: str = Field(..., description="Status message")

class StoryGenerationResult(BaseModel):
    story_script: str = Field(..., description="Generated story script")
    storyboard: Dict[str, Any] = Field(..., description="Generated storyboard")
    audio_url: Optional[str] = Field(None, description="URL to the generated audio file")
    sources: List[str] = Field(..., description="Sources used to generate the story")

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

# Periodic task scheduler
@app.on_event("startup")
async def startup_event():
    _logger.info("API server starting up...")
    # Process any queued audio files at startup
    process_audio_queue()

    # Start the cleanup background task if enabled
    if config.storage.FILE_RETENTION_DAYS > 0:
        asyncio.create_task(scheduled_cleanup())

@app.on_event("shutdown")
async def shutdown_event():
    _logger.info("API server shutting down...")

async def scheduled_cleanup():
    """Run periodic cleanup of old files based on retention policy"""
    while True:
        try:
            # Sleep first to avoid cleanup right at startup
            await asyncio.sleep(24 * 60 * 60)  # Run once per day

            # Get the retention period from config
            retention_days = config.storage.FILE_RETENTION_DAYS

            if retention_days > 0:
                _logger.info(f"Running scheduled cleanup of files older than {retention_days} days")
                deleted_count = storage_manager.cleanup_old_files(retention_days)
                _logger.info(f"Cleanup complete: {deleted_count} files removed")
        except Exception as e:
            _logger.error(f"Error in scheduled cleanup: {e}")

# Setup API security if enabled in config
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Authentication dependency
async def get_api_key(request: Request, api_key: str = Depends(api_key_header)):
    # Skip authentication if API key security is disabled
    if not config.http.API_KEY_ENABLED:
        return None

    if api_key != config.http.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

@app.get("/metrics")
async def get_metrics():
    """
    Endpoint to expose Prometheus metrics.
    This is used by Prometheus to scrape metrics from the application.
    """
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the API server is running.
    """
    return JSONResponse(content={"status": "ok"})

@router.post("/api/stories", dependencies=[Depends(get_api_key)])
async def generate_story(
    topic: str,
    background_tasks: BackgroundTasks,
    sources_folder: Optional[str] = None,
    voice_id: Optional[str] = None
):
    _logger.debug(f"Generate story endpoint called with topic: {topic}")
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
    task = create_story_task(topic, sources_folder, voice_id)
    _logger.debug(f"Story generation task created for topic: {topic}")
    return task

@router.get("/api/stories/{task_id}", dependencies=[Depends(get_api_key)])
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

@router.get("/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serve audio files directly"""
    file_path = os.path.join(config.storage.AUDIO_STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )

@router.get("/api/audio/stream/{filename}")
async def stream_audio(
    filename: str,
    range: str = None
):
    """
    Stream audio file with support for range requests (needed for seeking in audio players)

    Parameters:
    - filename: Name of the audio file to stream
    - range: HTTP Range header for partial content requests

    Returns:
    - Streaming response with audio data and appropriate headers for HTML5 audio players
    """
    # Get file size and check existence
    file_size = None
    file_path = None

    if config.storage.PROVIDER == "local":
        # For local files, check the file system directly
        file_path = os.path.join(config.storage.AUDIO_STORAGE_PATH, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
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

            # Use 206 Partial Content for range requests
            if start > 0 or end < file_size - 1:
                status_code = 206
        except ValueError:
            # If range header is invalid, ignore it
            pass

    # Calculate content length
    content_length = end - start + 1

    # Define headers
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Cache-Control": "public, max-age=3600"  # Allow caching for 1 hour
    }

    # Function to stream file content in chunks
    async def file_streamer():
        if config.storage.PROVIDER == "local" or file_path:
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

@router.get("/story/{filename}")
async def serve_story_file(filename: str):
    """Serve story text files directly"""
    file_path = os.path.join(config.storage.STORY_STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Story file not found")
    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=filename
    )

@router.get("/storyboard/{filename}")
async def serve_storyboard_file(filename: str):
    """Serve storyboard JSON files directly"""
    file_path = os.path.join(config.storage.STORYBOARD_STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Storyboard file not found")
    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=filename
    )

@router.get("/api/stories/{task_id}/download/{file_type}", dependencies=[Depends(get_api_key)])
async def download_story_file(task_id: str, file_type: str):
    """
    Download a generated file from a completed story task

    Parameters:
    - task_id: ID of the task
    - file_type: Type of file to download (story, audio, storyboard)

    Returns:
    - File download response
    """
    task_info = get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not completed")

    file_paths = task_info.get("file_paths", {})
    if file_type not in file_paths or not file_paths[file_type]:
        raise HTTPException(status_code=404, detail=f"No {file_type} file available for task {task_id}")

    file_path = file_paths[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found on server")

    # Set appropriate content type and filename
    if file_type == "audio":
        media_type = "audio/mpeg"
        filename = os.path.basename(file_path)
        return FileResponse(path=file_path, media_type=media_type, filename=filename)
    elif file_type == "storyboard":
        media_type = "application/json"
        filename = os.path.basename(file_path)
        return FileResponse(path=file_path, media_type=media_type, filename=filename)
    else:  # story text
        media_type = "text/plain"
        filename = os.path.basename(file_path)
        return FileResponse(path=file_path, media_type=media_type, filename=filename)

@router.post("/api/generate", response_model=JobResponse)
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
            "temperature": request.temperature or config.llm.TEMPERATURE,
            "chunk_size": request.chunk_size or int(os.environ.get("LLM_CHUNK_SIZE", config.llm.CHUNK_SIZE))
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

@router.get("/api/status/{job_id}")
async def get_job_status(
    job_id: str,
    queue_manager: RedisQueueManager = Depends(get_queue_manager)
):
    """
    Get the status of a story generation job.
    Returns the result if the job has completed.
    """
    try:
        result = queue_manager.get_result(job_id)
        if not result:
            return {"status": "pending", "message": "Job is still processing"}

        return result

    except Exception as e:
        _logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check job status: {str(e)}")

# Function to generate audio from text
def generate_audio(text: str) -> Dict[str, Any]:
    """
    Generate audio from text using ElevenLabs TTS

    Args:
        text: Text to convert to speech

    Returns:
        Dict with file information including path and URL
    """
    try:
        # Check for API key
        api_key = config.elevenLabs.API_KEY
        voice_id = config.elevenLabs.VOICE_ID

        if not api_key:
            _logger.error("No ElevenLabs API key configured. Cannot generate audio.")
            raise ValueError("ElevenLabs API key not configured")

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp3"
        temp_path = os.path.join(tempfile.gettempdir(), filename)

        # Generate the audio file
        success = generate_elevenlabs_audio(
            text=text,
            api_key=api_key,
            output_mp3_path=temp_path,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            stability=0.5,
            similarity_boost=0.75
        )

        if not success:
            raise ValueError("Failed to generate audio with ElevenLabs API")

        # Store the audio file using the storage manager
        with open(temp_path, "rb") as f:
            audio_data = f.read()

        # Store the audio file in the configured storage
        file_info = storage_manager.store_file(
            file_data=audio_data,
            file_type="audio",
            filename=filename,
            content_type="audio/mpeg"
        )

        # Clean up temp file
        os.remove(temp_path)

        return file_info

    except Exception as e:
        _logger.error(f"Error generating audio: {str(e)}")
        raise

async def process_story_generation(job_id: str, request_data: Dict[str, Any]):
    """
    Process a story generation request asynchronously.
    This function is run in the background after a request is received.

    Args:
        job_id: Unique identifier for the job
        request_data: Request data containing URLs and parameters
    """
    _logger.info(f"Starting job {job_id} for topic: {request_data['topic']}")

    try:
        queue_manager = RedisQueueManager(
            queue_name=API_QUEUE_NAME,
            result_prefix=RESULT_PREFIX
        )

        # Update status to processing
        queue_manager.store_result(job_id, {
            "status": "processing",
            "message": "Scraping content from URLs"
        })

        # Scrape content from URLs
        urls = request_data["urls"]
        scraped_content = await scrape_urls(urls)

        # Filter out failed scrapes
        valid_content = [(url, content) for url, content in scraped_content if content]

        if not valid_content:
            queue_manager.store_result(job_id, {
                "status": "failed",
                "message": "Failed to scrape any content from the provided URLs"
            })
            return

        # Update status
        queue_manager.store_result(job_id, {
            "status": "processing",
            "message": "Chunking and summarizing content"
        })

        # Prepare content for processing by combining all scraped content into one text
        combined_content = ""
        for _, content in valid_content:
            combined_content += content + "\n\n"

        # Process through chunking
        temperature = request_data.get("temperature", config.llm.TEMPERATURE)
        endpoint = config.llm.ENDPOINT
        model = config.llm.MODEL
        chunk_size = request_data.get("chunk_size", int(os.environ.get("LLM_CHUNK_SIZE", config.llm.CHUNK_SIZE)))

        # Apply chunking and summarization
        cleansed_content = chunkify_and_summarize(
            raw_sources=combined_content,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            chunk_size=chunk_size
        )

        # Process with LLM
        queue_manager.store_result(job_id, {
            "status": "processing",
            "message": "Generating story script"
        })
        story_script = process_with_llm(request_data["topic"], cleansed_content, temperature)

        # Generate storyboard
        queue_manager.store_result(job_id, {
            "status": "processing",
            "message": "Creating storyboard"
        })
        storyboard = generate_storyboard(story_script)

        result = {
            "status": "completed",
            "story_script": story_script,
            "storyboard": storyboard,
            "sources": [url for url, _ in valid_content]
        }

        # Generate audio if requested
        if request_data.get("generate_audio", False):
            queue_manager.store_result(job_id, {
                "status": "processing",
                "message": "Generating audio"
            })

            try:
                # Generate audio and store it in the configured storage
                file_info = generate_audio(story_script)

                # Add audio URL to result
                if config.storage.PROVIDER == "local":
                    # For local storage, construct URL using server's base URL
                    result["audio_url"] = f"{config.http.BASE_URL}/audio/{os.path.basename(file_info.get('file_path', ''))}"
                else:
                    # For cloud storage providers, use the URL directly
                    result["audio_url"] = file_info.get("url", None)

                # For tracking, add file path details
                result["audio_file"] = {
                    "path": file_info.get("file_path", ""),
                    "filename": file_info.get("filename", ""),
                    "provider": file_info.get("provider", "local"),
                    "storage_id": file_info.get("id", "")
                }

            except Exception as e:
                _logger.error(f"Error generating audio: {str(e)}")
                result["audio_url"] = None
                result["audio_error"] = str(e)

        # Store the final result
        queue_manager.store_result(job_id, result)
        _logger.info(f"Completed job {job_id}")

    except Exception as e:
        _logger.error(f"Error processing job {job_id}: {str(e)}")
        try:
            # Store the error
            queue_manager = RedisQueueManager(
                queue_name=API_QUEUE_NAME,
                result_prefix=RESULT_PREFIX
            )
            queue_manager.store_result(job_id, {
                "status": "failed",
                "message": f"Job failed: {str(e)}"
            })
        except Exception:
            _logger.exception("Failed to store job failure")

def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server with uvicorn"""
    _logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("viralStoryGenerator.src.api:app", host=host, port=port, reload=reload)