# viralStoryGenerator/src/api_worker.py
"""
API Worker for processing ViralStoryGenerator API requests via Redis Stream.
This module consumes requests published by the /api/generate endpoint.
"""
import asyncio
import os
import re
import signal
import sys
import tempfile
import time
import json
from typing import Dict, Any, Optional

from viralStoryGenerator.models import (
    JobStatusResponse
)
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.crawl4ai_scraper import get_scrape_result, queue_scrape_request
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.llm import process_with_llm
from viralStoryGenerator.src.storyboard import generate_storyboard
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.text_processing import split_text_into_chunks
from viralStoryGenerator.utils.vector_db_manager import get_vector_db_client

# Constants
API_QUEUE_NAME = app_config.redis.QUEUE_NAME
API_RESULT_PREFIX = app_config.redis.RESULT_PREFIX
RESULT_TTL = app_config.redis.TTL
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))

# Global module-level connections
_message_broker = None
_vector_db_client = None

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, _):
    """Handle shutdown signals gracefully."""
    _logger.warning(f"Received signal {sig}, initiating shutdown...")
    shutdown_event.set()

    # Give tasks time to complete
    _logger.info("Waiting for current tasks to complete...")
    time.sleep(5)

    _logger.info("Shutdown complete.")
    sys.exit(0)

def preload_components():
    """Preload and initialize key components at startup."""
    global _message_broker

    # Initialize Redis message broker
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name="api_jobs")

    # Create consumer group if it doesn't exist
    try:
        _message_broker.create_consumer_group(group_name="api_worker_group")
    except Exception as e:
        _logger.warning(f"Could not create consumer group: {e}")

    # Ensure stream exists
    _message_broker.ensure_stream_exists("api_jobs")

    _logger.info("Worker components initialized successfully")

def get_message_broker() -> Optional[RedisMessageBroker]:
    """Get the pre-initialized message broker or create a new one if needed."""
    global _message_broker
    if _message_broker is not None:
        return _message_broker

    # Initialize if not already done
    try:
        redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
        _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name="api_jobs")
        return _message_broker
    except Exception as e:
        _logger.error(f"Failed to initialize Redis message broker: {e}")
        return None

def get_vector_db():
    """Get the pre-initialized vector DB client."""
    global _vector_db_client
    return _vector_db_client or get_vector_db_client()

async def process_api_job(job_data: Dict[str, Any]) -> bool:
    """Process a single API job."""
    start_time = time.time()
    job_id = job_data.get("job_id", "unknown")
    message_broker = get_message_broker()

    if not message_broker:
        _logger.error(f"Cannot process job {job_id}: Redis message broker unavailable")
        return False

    # Update job status to processing
    message_broker.track_job_progress(job_id, "processing", {"message": "Job processing started"})

    try:
        # Process the job according to its type
        job_type = job_data.get("job_type", "unknown")

        if job_type == "generate_story":
            # Handle story generation job
            urls = job_data.get("urls", [])
            topic = job_data.get("topic", "")
            include_images = job_data.get("include_images", False)

            # Track progress
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Scraping content from provided URLs", "progress": 10}
            )

            # Process URLs if provided
            scraped_content = []
            if urls:
                scrape_job_id = await queue_scrape_request(urls)
                if scrape_job_id:
                    scraped_content = await get_scrape_result(scrape_job_id)

            # Track progress
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Processing with LLM", "progress": 50}
            )

            # Process with LLM
            llm_result = await process_with_llm(topic, scraped_content)

            # Generate storyboard
            storyboard = None
            if include_images:
                message_broker.track_job_progress(
                    job_id,
                    "processing",
                    {"message": "Generating storyboard with images", "progress": 75}
                )
                storyboard = await generate_storyboard(llm_result, topic)

            # Publish completion message
            message_broker.track_job_progress(
                job_id,
                "completed",
                {
                    "result": llm_result,
                    "storyboard": storyboard,
                    "processing_time": time.time() - start_time
                }
            )

            _logger.info(f"Job {job_id} completed successfully in {time.time() - start_time:.2f}s")
            return True
        else:
            # Unknown job type
            message_broker.track_job_progress(
                job_id,
                "failed",
                {"error": f"Unknown job type: {job_type}"}
            )
            _logger.warning(f"Unknown job type for {job_id}: {job_type}")
            return False

    except Exception as e:
        _logger.exception(f"Error processing job {job_id}: {e}")
        message_broker.track_job_progress(
            job_id,
            "failed",
            {"error": f"Processing error: {str(e)}"}
        )
        return False

async def process_api_jobs():
    """Process API jobs from the Redis stream."""
    while not shutdown_event.is_set():
        try:
            message_broker = get_message_broker()
            if not message_broker:
                _logger.error("Message broker unavailable, waiting before retry...")
                await asyncio.sleep(5)
                continue

            try:
                message_broker.create_consumer_group(group_name="api_worker_group")
            except Exception as e:
                _logger.debug(f"Consumer group 'api_worker_group' already exists: {e}")

            # Consume messages from the stream
            messages = message_broker.consume_messages(
                group_name="api_worker_group",
                consumer_name="api_worker_1",  # Could be dynamic based on worker ID
                count=MAX_CONCURRENT_JOBS,
                block=5000  # Wait up to 5 seconds for new messages
            )

            if not messages:
                await asyncio.sleep(0.1)  # Small sleep if no messages
                continue

            for stream_name, stream_messages in messages:
                if not stream_messages:
                    continue

                # Process messages concurrently with a limit
                tasks = []

                for message_id, message_data in stream_messages:
                    # Convert binary keys and values to strings
                    job_data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in message_data.items()
                    }

                    # Check if this is a system/initialization message
                    if "initialized" in job_data or "purged" in job_data:
                        _logger.debug(f"Skipping system message: {message_id}")
                        message_broker.acknowledge_message("api_worker_group", message_id)
                        continue

                    # Basic validation - check if we have job_id and job_type
                    job_id = job_data.get("job_id")
                    job_type = job_data.get("job_type")

                    if not job_id:
                        job_id = "unknown"
                        _logger.debug(f"Message {message_id} missing job_id")

                    if not job_type:
                        job_type = "unknown"
                        _logger.debug(f"Message {message_id} missing job_type")

                    # Skip invalid messages
                    if job_id == "unknown" and job_type == "unknown":
                        _logger.warning(f"Skipping invalid message {message_id} - missing job_id and job_type")
                        message_broker.acknowledge_message("api_worker_group", message_id)
                        continue

                    _logger.info(f"Processing message {message_id} (job_id: {job_id}, type: {job_type})")

                    # Parse JSON fields if needed
                    for key, value in job_data.items():
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                job_data[key] = json.loads(value)
                            except:
                                pass  # Keep as string if not valid JSON

                    # Create task to process the job
                    task = asyncio.create_task(process_api_job(job_data))
                    tasks.append(task)

                    # Acknowledge message after creating task
                    message_broker.acknowledge_message("api_worker_group", message_id)

                # Wait for all tasks to complete
                if tasks:
                    await asyncio.gather(*tasks)

        except Exception as e:
            _logger.exception(f"Error in process_api_jobs loop: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def run_worker():
    """Run the API worker with graceful shutdown handling."""
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)

    _logger.info(f"Starting API worker (PID: {os.getpid()})...")

    # Preload components
    preload_components()

    # Process jobs indefinitely
    await process_api_jobs()

def main():
    """Entry point for the worker process."""
    if os.name == 'nt':  # Windows
        # Use ProactorEventLoop for Windows compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == "__main__":
    main()