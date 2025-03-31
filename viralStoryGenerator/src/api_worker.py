"""
API Worker for processing ViralStoryGenerator API requests.
This module consumes requests from the API queue and processes them.
"""
import asyncio
import os
import signal
import sys
from typing import Dict, Any

from ..utils.redis_manager import RedisManager as RedisQueueManager
from ..utils.crawl4ai_scraper import scrape_urls
from ..utils.config import config
from .llm import process_with_llm
from .source_cleanser import chunkify_and_summarize
from .storyboard import generate_storyboard
from .elevenlabs_tts import generate_audio
from viralStoryGenerator.src.logger import logger as _logger

# API queue configuration
API_QUEUE_NAME = os.environ.get("API_QUEUE_NAME", "api_requests")
API_RESULT_PREFIX = os.environ.get("API_RESULT_PREFIX", "api_result:")

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    _logger.info(f"Received signal {sig}, shutting down API worker...")
    shutdown_event.set()

async def process_story_generation(job_id: str, request_data: Dict[str, Any], queue_manager: RedisQueueManager):
    _logger.debug(f"Processing story generation for job_id: {job_id}")
    """
    Process a story generation request asynchronously.

    Args:
        job_id: Unique identifier for the job
        request_data: Request data containing URLs and parameters
        queue_manager: Redis queue manager instance
    """
    _logger.info(f"Processing job {job_id} for topic: {request_data['topic']}")

    try:
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

        # Prepare content for processing by combining all scraped content into one text
        combined_content = ""
        for _, content in valid_content:
            combined_content += content + "\n\n"

        # Process through chunking
        temperature = request_data.get("temperature", config.llm.TEMPERATURE)
        endpoint = config.llm.ENDPOINT
        model = config.llm.MODEL
        chunk_size = int(os.environ.get("LLM_CHUNK_SIZE", config.llm.CHUNK_SIZE))

        # Update status to reflect chunking
        queue_manager.store_result(job_id, {
            "status": "processing",
            "message": "Chunking and summarizing content"
        })

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
                audio_path = generate_audio(story_script)
                # In a real-world scenario, you would host this file and return a URL
                result["audio_url"] = f"/audio/{audio_path.name}"
            except Exception as e:
                _logger.error(f"Error generating audio: {str(e)}")
                result["audio_url"] = None

        # Store the final result
        queue_manager.store_result(job_id, result)
        _logger.info(f"Completed job {job_id}")
        _logger.debug(f"Story generation completed for job_id: {job_id}")

    except Exception as e:
        _logger.error(f"Error processing job {job_id}: {str(e)}")
        queue_manager.store_result(job_id, {
            "status": "failed",
            "message": f"Job failed: {str(e)}"
        })

async def run_worker():
    _logger.debug("Starting worker loop...")
    """Run the API queue worker with graceful shutdown handling."""
    # Worker configuration
    batch_size = int(os.environ.get("REDIS_WORKER_BATCH_SIZE",
                                   config.redis.WORKER_BATCH_SIZE))
    sleep_interval = int(os.environ.get("REDIS_WORKER_SLEEP_INTERVAL",
                                       config.redis.WORKER_SLEEP_INTERVAL))
    max_concurrent = int(os.environ.get("REDIS_WORKER_MAX_CONCURRENT",
                                       config.redis.WORKER_MAX_CONCURRENT))

    _logger.info(f"Starting API worker with batch_size={batch_size}, max_concurrent={max_concurrent}")

    try:
        queue_manager = RedisQueueManager(
            queue_name=API_QUEUE_NAME,
            result_prefix=API_RESULT_PREFIX
        )
    except Exception as e:
        _logger.error(f"Failed to initialize Redis queue manager: {str(e)}")
        return

    while not shutdown_event.is_set():
        try:
            # Check queue length
            queue_length = queue_manager.get_queue_length()
            if queue_length == 0:
                # If queue is empty, sleep and check again
                await asyncio.sleep(sleep_interval)
                continue

            # Process up to batch_size requests
            batch = []
            for _ in range(min(batch_size, queue_length)):
                request = queue_manager.get_next_request()
                if request:
                    batch.append(request)

            if not batch:
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"Processing batch of {len(batch)} API requests")

            # Process requests in batches with max_concurrent limit
            tasks = []
            for i in range(0, len(batch), max_concurrent):
                sub_batch = batch[i:i + max_concurrent]
                for request in sub_batch:
                    request_id = request['id']
                    request_data = request['data']
                    tasks.append(process_story_generation(request_id, request_data, queue_manager))

                # Wait for all tasks in sub-batch to complete
                await asyncio.gather(*tasks)
                tasks = []

            # Sleep briefly to avoid spinning too fast
            await asyncio.sleep(0.1)

        except Exception as e:
            _logger.error(f"Error in API worker: {str(e)}")
            await asyncio.sleep(sleep_interval)
    _logger.debug("Worker loop completed.")

def main():
    """Entry point for the API worker process."""
    _logger.debug("Initializing API worker...")
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if sys.platform == 'win32':
        # Windows-specific asyncio setup for signal handling
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _logger.info(f"Starting API Queue Worker (queue={API_QUEUE_NAME}, result_prefix={API_RESULT_PREFIX})")

    # Check if Redis is enabled in configuration
    if not config.redis.ENABLED:
        _logger.error("Redis is disabled in configuration. Enable it to use the API worker.")
        sys.exit(1)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        _logger.info("API worker stopped by keyboard interrupt")
    except Exception as e:
        _logger.exception(f"API worker failed with error: {str(e)}")

    _logger.info("API Queue Worker shutdown complete")
    _logger.debug("API worker initialized.")

if __name__ == "__main__":
    main()