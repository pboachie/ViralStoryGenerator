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
from viralStoryGenerator.utils.crawl4ai_scraper import get_scrape_result, queue_scrape_request, redis_manager as scraper_redis_manager
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.llm import process_with_llm
from viralStoryGenerator.src.storyboard import generate_storyboard
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.text_processing import split_text_into_chunks
from viralStoryGenerator.utils.vector_db_manager import add_chunks_to_collection, query_collection, delete_collection, close_client as close_vector_db
from viralStoryGenerator.utils.crawl4ai_scraper import close_scraper_redis_connections

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
    Processes a single story generation request using RAG.
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
    rag_chunk_size = app_config.rag.CHUNK_SIZE
    rag_chunk_overlap = app_config.rag.CHUNK_OVERLAP # TODO: Currently not used by basic split_text_into_chunks
    rag_relevant_chunks_count = app_config.rag.RELEVANT_CHUNKS_COUNT
    voice_id = request_data.get("voice_id", app_config.elevenLabs.VOICE_ID)

    # Define collection name based on job_id for isolation
    collection_name = f"job_{job_id.replace('-', '')}" # ChromaDB names need specific format

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
        _logger.info(f"Job {job_id}: Queued scrape request with ID: {scrape_request_id}")

        # Wait for scrape result with retries
        max_retries = 5
        retry_interval = 5
        scrape_result = None
        for attempt in range(max_retries):
            if scraper_redis_manager:
                 _logger.debug(f"Job {job_id}: Attempt {attempt + 1}: Calling get_scrape_result for ID '{scrape_request_id}'. It will use manager with prefix '{scraper_redis_manager.result_prefix}'.") # DEBUG ADDED
            else:
                 _logger.error(f"Job {job_id}: Attempt {attempt + 1}: Cannot call get_scrape_result, scraper_redis_manager not initialized.")
                 break

            scrape_result = await get_scrape_result(scrape_request_id)
            if scrape_result:
                _logger.debug(f"Job {job_id}: Successfully retrieved scrape result for scrape ID {scrape_request_id}.") # DEBUG ADDED
                break
            scrape_status_data = scraper_redis_manager.get_result(scrape_request_id) if scraper_redis_manager else None
            _logger.warning(f"Job {job_id}: Scrape result for scrape ID {scrape_request_id} not ready (Status: {scrape_status_data.get('status') if scrape_status_data else 'Not Found/Manager Unavailable'}). Retrying in {retry_interval}s (Attempt {attempt + 1}/{max_retries})...") # DEBUG ADDED
            await asyncio.sleep(retry_interval)

        if not scrape_result:
            scrape_status_data = scraper_redis_manager.get_result(scrape_request_id) if scraper_redis_manager else None
            _logger.error(f"Job {job_id}: Failed to retrieve scrape results for scrape ID {scrape_request_id} after {max_retries} attempts. Final status check: {scrape_status_data}") # DEBUG ADDED
            raise ValueError("Failed to retrieve scrape results.")

        valid_content_list = [(url, content) for url, content in scrape_result if content and content.strip()]

        if not valid_content_list:
            _logger.warning(f"Job {job_id}: No valid content scraped from URLs: {urls}")
            raise ValueError("No valid content scraped from the provided URLs.")

        _logger.info(f"Job {job_id}: Successfully scraped content from {len(valid_content_list)} URL(s).")
        scraped_urls = [url for url, _ in valid_content_list]
        combined_raw_content = "\n\n".join([content for _, content in valid_content_list])

        # 3. RAG - Chunk, Embed, and Store Content TODO: Move to separate function
        queue_manager.store_result(job_id, {"message": "Chunking and embedding content...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
        _logger.info(f"Job {job_id}: Chunking content (length: {len(combined_raw_content)} chars) for RAG...")

        # Use the text processing utility to chunk the combined content
        # TODO: Consider using a more sophisticated chunking strategy if needed (e.g., LangChain's text splitters)
        chunks = split_text_into_chunks(combined_raw_content, rag_chunk_size) # Use RAG chunk size
        if not chunks:
             _logger.warning(f"Job {job_id}: No chunks generated from scraped content.")
             # TODO: Decide how to proceed - fail or try generating story without context?
             raise ValueError("Failed to generate text chunks from scraped content.")

        _logger.info(f"Job {job_id}: Generated {len(chunks)} chunks. Storing in vector DB collection '{collection_name}'...")

        # Prepare data for ChromaDB (Considering others)
        doc_ids = [f"{job_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source_url": url, "chunk_index": i} for i, chunk in enumerate(chunks) for url, content in valid_content_list if chunk in content] # Simple metadata, might need refinement
        # Ensure metadatas length matches chunks length (TODO: handle cases where chunk might span multiple sources if not careful)
        if len(metadatas) != len(chunks):
             _logger.warning(f"Job {job_id}: Metadata length mismatch ({len(metadatas)}) vs chunk length ({len(chunks)}). Using basic metadata.")
             metadatas = [{"chunk_index": i} for i in range(len(chunks))]

        # Add chunks to the vector database collection
        add_success = add_chunks_to_collection(collection_name, chunks, metadatas, doc_ids)
        if not add_success:
             raise RuntimeError(f"Failed to add chunks to vector database for job {job_id}.")
        _logger.info(f"Job {job_id}: Successfully stored chunks in vector DB.")

        # 4. RAG - Query Vector DB for Relevant Chunks
        queue_manager.store_result(job_id, {"message": "Retrieving relevant content...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
        _logger.info(f"Job {job_id}: Querying vector DB for chunks relevant to topic: '{topic}'")

        query_results = query_collection(collection_name, query_texts=[topic], n_results=rag_relevant_chunks_count)

        relevant_content = ""
        if query_results and query_results.get('documents') and query_results['documents'][0]:
            relevant_docs = query_results['documents'][0]
            relevant_content = "\n\n".join(relevant_docs)
            _logger.info(f"Job {job_id}: Retrieved {len(relevant_docs)} relevant chunks. Combined length: {len(relevant_content)} chars.")
        else:
            _logger.warning(f"Job {job_id}: Could not retrieve relevant chunks for topic '{topic}'. Proceeding with empty context.")
            # TODO: Decide: fail, or proceed with just the topic? Let's proceed for now.

        # 5. Generate Story Script using LLM with Relevant Content
        queue_manager.store_result(job_id, {"message": "Generating story script via LLM...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
        _logger.info(f"Job {job_id}: Generating story script using relevant content...")
        # Pass the retrieved relevant content
        story_script = process_with_llm(topic, relevant_content, temperature) # llm.py might need prompt adjustment
        if not story_script or story_script.isspace():
            _logger.error(f"Job {job_id}: LLM failed to generate a valid story script from relevant chunks.")
            raise ValueError("LLM generation resulted in empty script.")
        _logger.info(f"Job {job_id}: Story script generated.")

        # 6. Generate Storyboard
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

        # 7. Generate Audio (Optional)
        audio_url_result = None
        if generate_audio_flag and app_config.elevenLabs.ENABLED:
            if not app_config.elevenLabs.API_KEY:
                 _logger.warning(f"Job {job_id}: Audio generation requested, but ElevenLabs API key is missing in config. Skipping.")
            elif not story_script:
                 _logger.warning(f"Job {job_id}: Audio generation requested, but story script is empty. Skipping.")
            else:
                queue_manager.store_result(job_id, {"message": "Generating audio...", "updated_at": time.time()}, merge=True, ttl=RESULT_TTL)
                _logger.info(f"Job {job_id}: Generating audio (request flag is True, global flag is True)...")
                temp_audio_path = None
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
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                            _logger.debug(f"Cleaned up temporary audio file: {temp_audio_path}")
                        except OSError as e:
                             _logger.error(f"Failed to remove temporary audio file {temp_audio_path}: {e}")
        elif not app_config.elevenLabs.ENABLED:
             _logger.info(f"Job {job_id}: Skipping audio generation (globally disabled via ENABLE_AUDIO_GENERATION=False).")
        else: # generate_audio_flag must be False
             _logger.info(f"Job {job_id}: Skipping audio generation (request flag is False).")

        # 8. Prepare Final Result
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
        _logger.debug(f"Job {job_id}: Preparing final result: {final_result}") # DEBUG ADDED

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
        _logger.debug(f"Job {job_id}: Preparing final error result: {final_result}") # DEBUG ADDED

    finally:
        # 9. Cleanup Vector DB Collection for this job
        _logger.info(f"Job {job_id}: Cleaning up vector database collection '{collection_name}'...")
        try:
            delete_success = delete_collection(collection_name)
        except ValueError as ve:
            _logger.debug(f"Collection {collection_name} does not exist; skipping deletion. ({ve})")
            delete_success = True
        if not delete_success:
             _logger.warning(f"Job {job_id}: Failed to cleanup vector database collection '{collection_name}'.")

        # 10. Store Final Result in Redis
        try:
            _logger.debug(f"Job {job_id}: Attempting to store final result in Redis.") # DEBUG ADDED
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

    _logger.info(f"API Worker Config: BatchSize={batch_size}, SleepInterval={sleep_interval}s, MaxConcurrent={max_concurrent}")
    _logger.info(f"API Worker: Listening to Redis queue: '{API_QUEUE_NAME}' with result prefix '{API_RESULT_PREFIX}'")

    queue_manager = None
    while not queue_manager:
        if shutdown_event.is_set():
            _logger.info("Shutdown signal received during startup. Exiting.")
            return
        try:
            queue_manager = RedisQueueManager(
                queue_name=API_QUEUE_NAME,
                result_prefix=API_RESULT_PREFIX,
                ttl=RESULT_TTL
            )
            if not queue_manager.is_available():
                 raise ConnectionError("Initial Redis connection failed.")
            _logger.info(f"API Worker: Redis Queue Manager initialized successfully for queue '{API_QUEUE_NAME}'.")
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
                    _logger.critical(f"Concurrency limit ({max_concurrent}) reached. Waiting for tasks to finish.")
                    done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    active_tasks = pending
                else:
                     # Should not happen if num_to_fetch <= 0, but as safeguard
                     await asyncio.sleep(0.1)
                continue

            batch = []
            # _logger.debug(f"API Worker: Checking queue '{queue_manager.queue_name}' for up to {num_to_fetch} new jobs.") # DEBUG ADDED
            for _ in range(num_to_fetch):
                request = queue_manager.get_next_request()
                if request:
                    if isinstance(request, dict) and 'id' in request and 'data' in request:
                        job_id = request['id'] # DEBUG ADDED
                        job_status_data = queue_manager.get_result(job_id) # DEBUG ADDED
                        current_status = job_status_data.get("status") if job_status_data else "unknown" # DEBUG ADDED
                        _logger.debug(f"API Worker: Checking job {job_id}. Current status: {current_status}") # DEBUG ADDED
                        batch.append(request)
                        _logger.debug(f"API Worker: Added job {job_id} to current batch.") # DEBUG ADDED
                    else:
                        _logger.error(f"Invalid item received from queue: {str(request)[:100]}...")
                        # ... error handling ...
                else:
                    # _logger.debug("API Worker: Queue appears empty.") # DEBUG ADDED
                    break # Queue empty

            if not batch:
                # _logger.debug(f"No new jobs found. Sleeping for {sleep_interval}s.")
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"Fetched {len(batch)} new job(s). Processing...")

            # Create and manage tasks
            for request in batch:
                job_id = request['id']
                request_data = request['data']
                request_data["request_time"] = request.get("request_time", time.time())

                # Check if the job is already completed
                job_status = queue_manager.get_result(job_id)
                if job_status and job_status.get("status") == "completed":
                    _logger.info(f"Job {job_id} is already completed. Skipping.")
                    continue

                _logger.debug(f"API Worker: Creating task for job {job_id}.") # DEBUG ADDED
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
        try:
            await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=10.0)
            _logger.info("Active tasks finished or timed out.")
        except asyncio.TimeoutError:
            _logger.warning("Timeout waiting for active tasks to complete during shutdown.")
        except Exception as e:
            _logger.exception(f"Error waiting for tasks during shutdown: {e}")

    # Close resources
    _logger.info("Closing Vector DB client...")
    close_vector_db()

    _logger.info("API Worker exiting.")


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

    exit_code = 0
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        # Should be caught by signal handler, but as fallback
        _logger.info("API worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"API worker failed with unhandled exception: {e}")
        exit_code = 1
    finally:
        _logger.info("API Worker performing final cleanup...")

        # Ensure proper cleanup sequence
        close_vector_db()
        close_scraper_redis_connections()

        # Add explicit cleanup for multiprocessing resources
        try:
            import multiprocessing as mp
            if hasattr(mp, "resource_tracker") and hasattr(mp.resource_tracker, "_resource_tracker"):
                _logger.debug("Performing explicit multiprocessing resource cleanup")
                rt = mp.resource_tracker._resource_tracker
                if rt is not None:
                    rt.ensure_running()
                    rt.join()
        except Exception as cleanup_err:
            _logger.debug(f"Optional multiprocessing cleanup step failed: {cleanup_err}")

        # Force garbage collection to clean up lingering resources
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        time.sleep(1.0)

    _logger.info("API Queue Worker shutdown complete.")
    sys.exit(exit_code)


if __name__ == "__main__":
    # This allows running the worker directly using `python -m viralStoryGenerator.src.api_worker`
    main()