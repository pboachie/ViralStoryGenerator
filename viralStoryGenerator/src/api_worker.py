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
import uuid
import shutil
import datetime
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
from viralStoryGenerator.utils.storage_manager import storage_manager

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

def preload_components(group_name: str):
    """Preload and initialize key components at startup."""
    global _message_broker

    # Initialize Redis message broker
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME) # Use QUEUE_NAME

    # Create consumer group if it doesn't exist using the provided group_name
    try:
        _message_broker.create_consumer_group(group_name=group_name)
        _logger.info(f"Ensured consumer group '{group_name}' exists for stream '{app_config.redis.QUEUE_NAME}'.")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            _logger.warning(f"Could not create consumer group '{group_name}': {e}")
        else:
            _logger.debug(f"Consumer group '{group_name}' already exists.")

    # Ensure stream exists
    _message_broker.ensure_stream_exists(app_config.redis.QUEUE_NAME)

    _logger.info(f"Worker components initialized successfully for stream '{app_config.redis.QUEUE_NAME}'.")

def get_message_broker() -> Optional[RedisMessageBroker]:
    """Get the pre-initialized message broker or create a new one if needed."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    # Initialize if not already done
    try:
        redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
        _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)
        _logger.info(f"Initialized API Worker RedisMessageBroker with stream: '{app_config.redis.QUEUE_NAME}'")
        return _message_broker
    except Exception as e:
        _logger.error(f"Failed to initialize Redis message broker: {e}")
        return None

def get_vector_db():
    """Get the pre-initialized vector DB client."""
    global _vector_db_client
    return _vector_db_client or get_vector_db_client()

async def process_api_job(job_data: Dict[str, Any], consumer_name: str, group_name: str) -> bool:
    """Process a single API job."""
    start_time = time.time()
    job_id = job_data.get("job_id", "unknown")
    message_broker = get_message_broker()
    story_script_info = None
    storyboard_info = None
    final_metadata_info = None

    _logger.debug(f"Processing job {job_id} via consumer '{consumer_name}' in group '{group_name}'")

    if not message_broker:
        _logger.error(f"Cannot process job {job_id}: Redis message broker unavailable")
        return False

    # Update job status to processing
    message_broker.track_job_progress(job_id, "processing", {"message": "Job processing started"})

    try:
        job_type = job_data.get("job_type", "unknown")

        if job_type == "generate_story":
            urls = job_data.get("urls", [])
            if isinstance(urls, str):
                try:
                    urls = json.loads(urls)
                except Exception:
                    urls = [urls]
            if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u.strip() for u in urls):
                _logger.warning(f"Message {job_id} has no valid URLs, acknowledging and skipping")
                message_broker.track_job_progress(job_id, "failed", {"error": "No valid URLs provided"})
                return False
            topic = job_data.get("topic", "")
            # include_images = job_data.get("include_images", False)
            temperature = job_data.get("temperature", app_config.llm.TEMPERATURE)
            voice_id = job_data.get("voice_id")

            # --- Scraping --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Scraping content from provided URLs", "progress": 10}
            )
            scraped_content = []
            scrape_failed = False
            if urls:
                _logger.info(f"Job {job_id}: Queuing scrape request for {len(urls)} URLs.")
                scrape_job_id = await queue_scrape_request(
                    urls,
                    wait_for_result=True,
                    timeout=app_config.httpOptions.TIMEOUT
                )
                if scrape_job_id:
                    _logger.info(f"Job {job_id}: Scrape job {scrape_job_id} queued. Retrieving result...")
                    scrape_result_data = await get_scrape_result(scrape_job_id)
                    if scrape_result_data:
                        scraped_content = [content for url, content in scrape_result_data if content]
                        if not scraped_content:
                            _logger.warning(f"Job {job_id}: Scrape job {scrape_job_id} completed but returned no content.")
                            scrape_failed = True
                        else:
                             _logger.info(f"Job {job_id}: Successfully retrieved scraped content ({len(scraped_content)} items)." )
                    else:
                        _logger.error(f"Job {job_id}: Failed to retrieve result for scrape job {scrape_job_id} after waiting (timed out or failed).")
                        scrape_failed = True
                else:
                    _logger.error(f"Job {job_id}: Scrape request {scrape_job_id or 'unknown'} failed or timed out.")
                    scrape_failed = True

            if scrape_failed:
                message_broker.track_job_progress(job_id, "failed", {"error": "Scraping step failed or timed out"})
                return False
            # < --- End Scraping ---

            # --- LLM Processing --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Generating story script with LLM", "progress": 40}
            )
            llm_result = process_with_llm(
                topic=topic,
                relevant_content="\n\n".join(scraped_content),
                temperature=temperature,
                model=app_config.llm.MODEL # Pass the default model
            )
            if not llm_result:
                 _logger.error(f"Job {job_id}: LLM processing returned empty result.")
                 message_broker.track_job_progress(job_id, "failed", {"error": "LLM processing failed or returned empty."})
                 return False
            # < --- End LLM Processing ---

            # --- Store Story Script --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Storing story script", "progress": 50}
            )
            story_script_filename = f"{job_id}_story.txt"
            try:
                story_script_info = storage_manager.store_file(
                    file_data=llm_result,
                    file_type="story",
                    filename=story_script_filename,
                    content_type="text/plain"
                )
                if "error" in story_script_info:
                    _logger.error(f"Job {job_id}: Failed to store story script: {story_script_info.get('error')}")
                    message_broker.track_job_progress(job_id, "failed", {"error": f"Failed to store story script: {story_script_info.get('error')}"})
                    return False
                else:
                    _logger.info(f"Job {job_id}: Story script stored: {story_script_info.get('file_path')}")
            except Exception as store_err:
                 _logger.exception(f"Job {job_id}: Exception storing story script: {store_err}")
                 message_broker.track_job_progress(job_id, "failed", {"error": f"Exception storing story script: {store_err}"})
                 return False
            # < --- End Store Story Script ---

            # --- Storyboard Generation --- >
            storyboard_data = None
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Generating storyboard (including images/audio if enabled)", "progress": 60}
            )
            try:
                storyboard_data = generate_storyboard(
                    story=llm_result,
                    topic=topic,
                    task_id=job_id,
                    llm_endpoint=app_config.llm.ENDPOINT,
                    temperature=temperature,
                    voice_id=voice_id
                )
                if storyboard_data:
                    _logger.info(f"Job {job_id}: Storyboard generation completed.")
                    storyboard_info = {
                        "file_path": storyboard_data.get("storyboard_file"),
                        "url": storyboard_data.get("storyboard_url"),
                        "provider": storage_manager.provider # Assume same provider
                    }
                else:
                    _logger.warning(f"Job {job_id}: Storyboard generation returned None. Proceeding without storyboard.")
                    message_broker.track_job_progress(job_id, "processing", {"message": "Storyboard generation failed or skipped", "progress": 90})

            except Exception as sb_err:
                 _logger.exception(f"Job {job_id}: Error during storyboard generation: {sb_err}")
                 message_broker.track_job_progress(job_id, "processing", {"message": f"Storyboard generation failed: {sb_err}", "progress": 90})
            # < --- End Storyboard Generation ---

            # --- Final Metadata Aggregation & Storage --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Aggregating and storing final metadata", "progress": 95}
            )
            created_at_ts = job_data.get("request_time") or start_time
            try:
                created_at_float = float(created_at_ts)
            except (ValueError, TypeError):
                _logger.warning(f"Job {job_id}: Could not convert created_at_ts '{created_at_ts}' to float. Using current time as fallback.")
                created_at_float = start_time

            updated_at = time.time()

            final_metadata = {
                "job_id": job_id,
                "topic": topic,
                "status": "completed",
                "message": "Job completed successfully.",
                "created_at": datetime.datetime.fromtimestamp(created_at_float, tz=datetime.timezone.utc).isoformat() if created_at_float else None,
                "updated_at": datetime.datetime.fromtimestamp(updated_at, tz=datetime.timezone.utc).isoformat(),
                "processing_time_seconds": round(updated_at - start_time, 2),
                "story_script_file": story_script_info.get("file_path") if story_script_info else None,
                "story_script_url": story_script_info.get("url") if story_script_info else None,
                "storyboard_file": storyboard_info.get("file_path") if storyboard_info else None,
                "storyboard_url": storyboard_info.get("url") if storyboard_info else None,
                "audio_file": storyboard_data.get("audio_file") if storyboard_data else None,
                "audio_url": storyboard_data.get("audio_url") if storyboard_data else None,
                "sources": urls,
                "llm_model": app_config.llm.MODEL,
                "llm_temperature": temperature,
                "voice_id": voice_id,
            }

            if not storyboard_info and 'sb_err' in locals():
                 final_metadata["message"] = f"Job completed, but storyboard generation failed: {sb_err}"
                 final_metadata["error"] = f"Storyboard generation failed: {sb_err}"
            elif not storyboard_info and not 'sb_err' in locals():
                 final_metadata["message"] = "Job completed, but storyboard generation returned no data."

            metadata_filename = f"{job_id}_metadata.json"
            try:
                metadata_json_str = json.dumps(final_metadata, indent=2)
                final_metadata_info = storage_manager.store_file(
                    file_data=metadata_json_str,
                    file_type="metadata",
                    filename=metadata_filename,
                    content_type="application/json"
                )
                if "error" in final_metadata_info:
                     _logger.error(f"Job {job_id}: Failed to store final metadata: {final_metadata_info.get('error')}")
                     message_broker.track_job_progress(job_id, "failed", {"error": f"Failed to store final metadata: {final_metadata_info.get('error')}"})
                     return False
                else:
                     _logger.info(f"Job {job_id}: Final metadata stored: {final_metadata_info.get('file_path')}")
            except Exception as meta_err:
                 _logger.exception(f"Job {job_id}: Exception storing final metadata: {meta_err}")
                 message_broker.track_job_progress(job_id, "failed", {"error": f"Exception storing final metadata: {meta_err}"})
                 return False
            # < --- End Final Metadata ---

            # --- Final Redis Update --- >
            redis_final_payload = {
                "message": final_metadata["message"],
                "story_script_ref": story_script_info.get("file_path") if story_script_info else None,
                "storyboard_ref": storyboard_info.get("file_path") if storyboard_info else None,
                "metadata_ref": final_metadata_info.get("file_path") if final_metadata_info else None,
                "processing_time": round(updated_at - start_time, 2),
                "created_at": final_metadata["created_at"],
                "updated_at": final_metadata["updated_at"],
            }
            if "error" in final_metadata:
                 redis_final_payload["error"] = final_metadata["error"]

            message_broker.track_job_progress(job_id, "completed", redis_final_payload)
            # < --- End Final Redis Update ---

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
        try:
            message_broker.track_job_progress(
                job_id,
                "failed",
                {
                    "error": f"Processing error: {str(e)}",
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                 }
            )
        except Exception as final_track_err:
             _logger.error(f"Job {job_id}: CRITICAL - Failed to update final status to failed in Redis: {final_track_err}")
        return False

async def process_api_jobs(group_name: str, consumer_name: str):
    """Process API jobs from the Redis stream."""
    while not shutdown_event.is_set():
        try:
            message_broker = get_message_broker()
            if not message_broker:
                _logger.error("Message broker unavailable, waiting before retry...")
                await asyncio.sleep(5)
                continue

            try:
                message_broker.create_consumer_group(group_name=group_name)
            except Exception as e:
                _logger.debug(f"Consumer group '{group_name}' already exists: {e}")

            # Consume messages from the stream using passed arguments
            messages = message_broker.consume_messages(
                group_name=group_name,
                consumer_name=consumer_name,
                count=MAX_CONCURRENT_JOBS,
                block=5000
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
                    job_data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in message_data.items()
                    }

                    # Check if this is a system/initialization message
                    if "initialized" in job_data or "purged" in job_data:
                        _logger.debug(f"Skipping system message: {message_id}")
                        message_broker.acknowledge_message(group_name, message_id)
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
                        message_broker.acknowledge_message(group_name, message_id)
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
                    task = asyncio.create_task(process_api_job(job_data, consumer_name, group_name))
                    tasks.append(task)

                    # Acknowledge message after creating task
                    message_broker.acknowledge_message(group_name, message_id)

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

    group_name = app_config.redis.API_WORKER_GROUP_NAME
    consumer_name = f"api-worker-{uuid.uuid4().hex[:8]}"

    _logger.info(f"Starting API worker (PID: {os.getpid()}, Group: {group_name}, Consumer: {consumer_name})...")

    # Preload components
    preload_components(group_name)

    # Process jobs indefinitely
    await process_api_jobs(group_name, consumer_name)

def main():
    """Entry point for the worker process."""
    if os.name == 'nt':  # Windows
        # Use ProactorEventLoop for Windows compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        _logger.info("Worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"Worker failed unexpectedly: {e}")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError as e:
            if "Cannot run the event loop while another loop is running" in str(e):
                 _logger.warning("Attempted to shutdown asyncgens while loop was closing.")
            else:
                 raise
        finally:
            if loop.is_running():
                 loop.stop()
            if not loop.is_closed():
                 loop.close()
            _logger.info("Event loop closed.")


if __name__ == "__main__":
    main()