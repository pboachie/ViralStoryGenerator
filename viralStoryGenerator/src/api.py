"""
HTTP API backend for ViralStoryGenerator.
This module provides HTTP endpoints that replicate CLI functionality
but accept URLs instead of file paths.
"""
import asyncio
import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyHttpUrl, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

from ..utils.redis_manager import RedisQueueManager
from ..utils.crawl4ai_scraper import scrape_urls
from ..utils.config import config
from ..src.llm import process_with_llm
from ..src.source_cleanser import cleanse_sources
from ..src.storyboard import generate_storyboard
from ..src.elevenlabs_tts import generate_audio

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Viral Story Generator API",
    description="API for generating viral stories from web content",
    version="0.1.2"
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Active requests')
QUEUE_SIZE = Gauge('api_queue_size', 'API queue size')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
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
        logger.error(f"Failed to initialize Redis queue: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis queue service unavailable")

@app.post("/api/generate", response_model=JobResponse)
async def generate_story(
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
            "urls": [str(url) for url in request.urls],
            "topic": request.topic,
            "generate_audio": request.generate_audio,
            "temperature": request.temperature or config.llm.TEMPERATURE
        }

        queue_manager.add_request(request_data)

        # Start processing in the background
        background_tasks.add_task(process_story_generation, job_id, request_data)

        return JobResponse(
            job_id=job_id,
            message="Story generation job queued successfully."
        )

    except Exception as e:
        logger.error(f"Error queueing job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

@app.get("/api/status/{job_id}")
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
        logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check job status: {str(e)}")

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

async def process_story_generation(job_id: str, request_data: Dict[str, Any]):
    """
    Process a story generation request asynchronously.
    This function is run in the background after a request is received.

    Args:
        job_id: Unique identifier for the job
        request_data: Request data containing URLs and parameters
    """
    logger.info(f"Starting job {job_id} for topic: {request_data['topic']}")

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
            "message": "Cleansing and processing content"
        })

        # Prepare content for processing (similar to what source_cleanser would do)
        sources = []
        for url, content in valid_content:
            sources.append({
                "filename": url,  # Using URL as filename for identification
                "content": content
            })

        # Cleanse sources
        cleansed_content = cleanse_sources(sources)

        # Process with LLM
        temperature = request_data.get("temperature", config.llm.TEMPERATURE)
        story_script = process_with_llm(request_data["topic"], cleansed_content, temperature)

        # Generate storyboard
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
                audio_path = generate_audio(story_script)
                # In a real-world scenario, you would host this file and return a URL
                result["audio_url"] = f"/audio/{audio_path.name}"
            except Exception as e:
                logger.error(f"Error generating audio: {str(e)}")
                result["audio_url"] = None

        # Store the final result
        queue_manager.store_result(job_id, result)
        logger.info(f"Completed job {job_id}")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
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
            logger.exception("Failed to store job failure")