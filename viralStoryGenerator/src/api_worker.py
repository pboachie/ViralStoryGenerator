# viralStoryGenerator/src/api_worker.py
"""
API Worker for processing ViralStoryGenerator API requests via Redis Queue.
This module consumes requests queued by the /api/generate endpoint.
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
from viralStoryGenerator.utils.redis_manager import RedisManager as RedisQueueManager
from viralStoryGenerator.utils.crawl4ai_scraper import get_scrape_result, queue_scrape_request
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.llm import process_with_llm
from viralStoryGenerator.src.source_cleanser import chunkify_and_summarize
from viralStoryGenerator.src.storyboard import generate_storyboard
from viralStoryGenerator.src.elevenlabs_tts import generate_audio
from viralStoryGenerator.src.logger import logger as _logger
# TODO: Import storage manager needed for potential file URL construction? Maybe not needed directly here.
# from viralStoryGenerator.utils.storage_manager import storage_manager

# Use queue configuration from app_config.redis
API_QUEUE_NAME = app_config.redis.QUEUE_NAME
API_RESULT_PREFIX = app_config.redis.RESULT_PREFIX
RESULT_TTL = app_config.redis.TTL

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    _logger.info(f"Received signal {sig}, initiating shutdown for API worker...")
    shutdown_event.set()

async def process_story_request(job_id: str, request_data: Dict[str, Any], queue_manager: RedisQueueManager):
    """
    Processes a single story generation request.

    Args:
        job_id: Unique identifier for the job.
        request_data: Request data containing URLs, topic, and parameters.
        queue_manager: Redis queue manager instance.
    """
    start_time = time.time()
    _logger.info(f"Processing Job {job_id}: Starting story generation for topic '{request_data.get('topic', 'N/A')}'")
    _logger.debug(f"Job {job_id}: Received request data: {request_data}")

    # --- Basic Input Validation ---
    if not all(k in request_data for k in ['urls', 'topic']):
        _logger.error(f"Job {job_id}: Invalid request data received from queue. Missing 'urls' or 'topic'. Data: {request_data}")
        error_result = JobStatusResponse(
            status="failed",
            message="Invalid job data received from queue.",
            error="Missing required fields 'urls' or 'topic'.",
            created_at=request_data.get("request_time", start_time),
            updated_at=time.time(),
            original_request_data=request_data
        ).model_dump(exclude_none=True)
        queue_manager.store_result(job_id, error_result, ttl=RESULT_TTL)
        return

    # --- Processing Steps ---
    urls = request_data["urls"]
    topic = request_data["topic"]
    generate_audio_flag = request_data.get("generate_audio", False)
    temperature = request_data.get("temperature", app_config.llm.TEMPERATURE)
    chunk_size = request_data.get("chunk_size", app_config.llm.CHUNK_SIZE)
    voice_id = request_data.get("voice_id", app_config.elevenLabs.VOICE_ID)

    final_result = {}

    try:
        # 1. Update Status: Processing Start
        status_update = {"status": "processing", "message": "Starting job processing", "updated_at": time.time()}
        queue_manager.store_result(job_id, status_update, merge=True, ttl=RESULT_TTL)

        # 2. Scrape URLs
        _logger.info(f"Job {job_id}: Scraping content from {len(urls)} URL(s)...")
        queue_manager.store_result(job_id, {"message": "Scraping content...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)

        scrape_request_id = await queue_scrape_request(urls)
        if not scrape_request_id:
            _logger.error(f"Job {job_id}: Failed to queue scrape request for URLs: {urls}")
            raise ValueError("Failed to queue scrape request.")

        # Wait for scrape result
        scraped_content_list = await get_scrape_result(scrape_request_id)
        if not scraped_content_list:
            _logger.warning(f"Job {job_id}: Failed to retrieve scrape results for request ID: {scrape_request_id}")
            raise ValueError("Failed to retrieve scrape results.")

        valid_content_list = [(url, content) for url, content in scraped_content_list if content and content.strip()]

        if not valid_content_list:
            _logger.warning(f"Job {job_id}: No valid content scraped from URLs: {urls}")
            raise ValueError("No valid content scraped from the provided URLs.")

        _logger.info(f"Job {job_id}: Successfully scraped content from {len(valid_content_list)} URL(s).")
        scraped_urls = [url for url, _ in valid_content_list]

        # 3. Cleanse Content
        queue_manager.store_result(job_id, {"message": "Cleansing and summarizing content...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
        combined_raw_content = "\n\n".join([content for _, content in valid_content_list])
        _logger.info(f"Job {job_id}: Cleansing content (length: {len(combined_raw_content)} chars)...")
        cleansed_content = chunkify_and_summarize(
            raw_sources=combined_raw_content,
            endpoint=app_config.llm.ENDPOINT,
            model=app_config.llm.MODEL,
            temperature=temperature,
            chunk_size=chunk_size
        )
        if cleansed_content is None:
             _logger.error(f"Job {job_id}: Content cleansing/summarization failed.")
             raise ValueError("Content cleansing and summarization failed.")
        _logger.info(f"Job {job_id}: Content cleansed (new length: {len(cleansed_content)} chars).")


        # 4. Generate Story Script using LLM
        queue_manager.store_result(job_id, {"message": "Generating story script via LLM...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
        _logger.info(f"Job {job_id}: Generating story script...")
        story_script = process_with_llm(topic, cleansed_content, temperature)
        if not story_script or story_script.isspace():
            _logger.error(f"Job {job_id}: LLM failed to generate a valid story script.")
            raise ValueError("LLM generation resulted in empty script.")
        _logger.info(f"Job {job_id}: Story script generated.")


        # 5. Generate Storyboard
        queue_manager.store_result(job_id, {"message": "Generating storyboard...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
        _logger.info(f"Job {job_id}: Generating storyboard...")
        storyboard_data = generate_storyboard(
            story=story_script,
            topic=topic,
            llm_endpoint=app_config.llm.ENDPOINT,
            model=app_config.llm.MODEL,
            temperature=temperature,
            voice_id=voice_id
        )
        if storyboard_data is None or not storyboard_data.get("scenes"):
             _logger.warning(f"Job {job_id}: Storyboard generation failed or produced no scenes. Proceeding without storyboard.")
             storyboard_result = {}
        else:
             _logger.info(f"Job {job_id}: Storyboard generated with {len(storyboard_data.get('scenes', []))} scenes.")
             storyboard_result = storyboard_data


        # 6. Generate Audio (Optional)
        audio_url_result = None
        if generate_audio_flag:
            queue_manager.store_result(job_id, {"message": "Generating audio...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
            _logger.info(f"Job {job_id}: Generating audio (flag is True)...")
            try:
                from viralStoryGenerator.src.elevenlabs_tts import generate_elevenlabs_audio
                from viralStoryGenerator.utils.storage_manager import storage_manager

                # Generate a safe filename
                safe_topic_base = re.sub(r'[\\/*?:"<>|\0]', '_', topic)[:50]
                audio_filename = f"{job_id}_{safe_topic_base}.mp3"

                # Generate audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_f:
                    temp_audio_path = temp_f.name

                success = generate_elevenlabs_audio(
                    text=story_script,
                    api_key=app_config.elevenLabs.API_KEY,
                    output_mp3_path=temp_audio_path,
                    voice_id=voice_id
                )

                if success:
                    _logger.info(f"Job {job_id}: Audio generated to temp file {temp_audio_path}.")
                    with open(temp_audio_path, "rb") as audio_f:
                        store_info = storage_manager.store_file(
                            file_data=audio_f,
                            file_type="audio",
                            filename=audio_filename,
                            content_type="audio/mpeg"
                        )
                    # Clean up temp file
                    os.remove(temp_audio_path)

                    if "error" not in store_info:
                        audio_key = store_info.get("file_path")
                        audio_url_result = storage_manager.get_file_url(audio_key, "audio")
                        _logger.info(f"Job {job_id}: Audio stored successfully. URL: {audio_url_result}")
                    else:
                         _logger.error(f"Job {job_id}: Failed to store generated audio: {store_info.get('error')}")
                else:
                     _logger.warning(f"Job {job_id}: Audio generation failed (elevenlabs).")

            except Exception as audio_err:
                _logger.exception(f"Job {job_id}: Error during audio generation/storage: {audio_err}")
            finally:
                if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                    try: os.remove(temp_audio_path)
                    except OSError: pass
        else:
             _logger.info(f"Job {job_id}: Skipping audio generation (flag is False).")


        # 7. Prepare Final Result
        processing_time = time.time() - start_time
        _logger.info(f"Job {job_id}: Processing successful. Time: {processing_time:.2f}s")
        final_result = JobStatusResponse(
            status="completed",
            message="Story generated successfully.",
            story_script=story_script,
            storyboard=storyboard_result,
            audio_url=audio_url_result,
            sources=scraped_urls,
            created_at=request_data.get("request_time", start_time),
            updated_at=time.time(),
            processing_time_seconds=round(processing_time, 2),
            original_request_data=request_data
        ).model_dump(exclude_none=True)

    except Exception as e:
        # Catch all errors during processing steps
        processing_time = time.time() - start_time
        error_msg = f"Job {job_id}: Processing failed after {processing_time:.2f}s. Error: {str(e)}"
        _logger.exception(error_msg)
        final_result = JobStatusResponse(
            status="failed",
            message="Job processing failed.",
            error=str(e),
            created_at=request_data.get("request_time", start_time),
            updated_at=time.time(),
            original_request_data=request_data
        ).model_dump(exclude_none=True)

    # 8. Store Final Result in Redis
    try:
        success = queue_manager.store_result(job_id, final_result, ttl=RESULT_TTL)
        if not success:
             _logger.error(f"Job {job_id}: CRITICAL - Failed to store final result in Redis!")
        else:
             _logger.debug(f"Job {job_id}: Final result stored in Redis.")
    except Exception as redis_err:
         _logger.exception(f"Job {job_id}: CRITICAL - Exception while storing final result in Redis: {redis_err}")


async def run_worker():
    """Main worker loop to poll Redis queue and process jobs."""
    _logger.info("Starting API Worker process...")

    # Worker configuration
    batch_size = app_config.redis.WORKER_BATCH_SIZE
    sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent = app_config.redis.WORKER_MAX_CONCURRENT

    _logger.info(f"Worker Config: BatchSize={batch_size}, SleepInterval={sleep_interval}s, MaxConcurrent={max_concurrent}")
    _logger.info(f"Listening to Redis queue: '{API_QUEUE_NAME}'")

    queue_manager = None
    while not queue_manager:
        if shutdown_event.is_set():
            _logger.info("Shutdown signal received during startup. Exiting.")
            return
        try:
            queue_manager = RedisQueueManager(
                queue_name=API_QUEUE_NAME,
                result_prefix=API_RESULT_PREFIX
            )
            if not queue_manager.is_available():
                 raise ConnectionError("Initial Redis connection failed.")
            _logger.info("Redis Queue Manager initialized successfully.")
        except Exception as e:
            _logger.error(f"Failed to initialize Redis queue manager: {e}. Retrying in 5 seconds...")
            queue_manager = None
            await asyncio.sleep(5)


    active_tasks = set()
    while not shutdown_event.is_set():
        try:
            # Check Redis connection periodically
            if not queue_manager.is_available():
                 _logger.error("Redis connection lost. Attempting to reconnect...")
                 await asyncio.sleep(5)
                 try:
                      queue_manager = RedisQueueManager(queue_name=API_QUEUE_NAME, result_prefix=API_RESULT_PREFIX)
                      if not queue_manager.is_available(): continue
                      _logger.info("Reconnected to Redis.")
                 except Exception:
                      _logger.error("Reconnect failed. Will retry next cycle.")
                      continue


            # Fetch jobs only if concurrency limit allows
            num_to_fetch = min(batch_size, max_concurrent - len(active_tasks))
            if num_to_fetch <= 0:
                # Max concurrency reached, wait for tasks to complete
                if active_tasks:
                    _logger.debug(f"Concurrency limit ({max_concurrent}) reached. Waiting for tasks to finish.")
                    done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    active_tasks = pending
                else:
                     # Should not happen if num_to_fetch <= 0, but as safeguard
                     await asyncio.sleep(0.1)
                continue

            _logger.debug(f"Checking for up to {num_to_fetch} new jobs...")
            batch = []
            for _ in range(num_to_fetch):
                request = queue_manager.get_next_request()
                if request:
                    if isinstance(request, dict) and 'id' in request and 'data' in request:
                        batch.append(request)
                    else:
                        _logger.error(f"Invalid item received from queue: {str(request)[:100]}...")
                        if isinstance(request, dict) and '_original_data' in request:
                             if hasattr(queue_manager, 'complete_request'):
                                 queue_manager.complete_request(request, success=False)
                else:
                    break

            if not batch:
                _logger.debug(f"No new jobs found. Sleeping for {sleep_interval}s.")
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"Fetched {len(batch)} new job(s). Processing...")

            # Create and manage tasks
            for request in batch:
                job_id = request['id']
                request_data = request['data']
                request_data["request_time"] = request.get("request_time", time.time())

                # Create task and add to active set
                task = asyncio.create_task(process_story_request(job_id, request_data, queue_manager))
                active_tasks.add(task)
                # Remove task from set upon completion (success or failure)
                task.add_done_callback(active_tasks.discard)

            # Wait briefly if batch was full to allow task completion checks
            if len(batch) == num_to_fetch:
                 await asyncio.sleep(0.1)


        except ConnectionError as e:
             _logger.error(f"Redis connection error in main loop: {e}. Worker will pause and retry.")
             await asyncio.sleep(10)
        except Exception as e:
            _logger.exception(f"FATAL: Unexpected error in API worker main loop: {e}. Sleeping before retry...")
            await asyncio.sleep(sleep_interval * 5)

    # --- Shutdown Sequence ---
    _logger.info("Shutdown signal received. Waiting for active tasks to complete...")
    if active_tasks:
        await asyncio.wait(active_tasks)
    _logger.info("All active tasks finished. API Worker exiting.")


def main():
    """Entry point for the API worker process."""
    _logger.info("Initializing API Queue Worker...")
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if sys.platform == 'win32':
        # Required for signal handling on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Check Redis configuration
    if not app_config.redis.ENABLED:
        _logger.error("Redis is disabled in configuration (REDIS_ENABLED=False). API Worker cannot run.")
        sys.exit(1)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        # Should be caught by signal handler, but as fallback
        _logger.info("API worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"API worker failed with unhandled exception: {e}")
        sys.exit(1)

    _logger.info("API Queue Worker shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    # This allows running the worker directly using `python -m viralStoryGenerator.src.api_worker`
    main()