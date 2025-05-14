# viralStoryGenerator/src/scrape_worker.py
"""
Scraper Worker for processing scraping requests via Redis Streams.
This module consumes requests published to the scraper stream using the efficient scraping method.
"""
import asyncio
import signal
import sys
import time
import json
import os
import uuid
import logging
from typing import List, Optional

from viralStoryGenerator.src.logger import base_app_logger as _logger
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker

try:
    from viralStoryGenerator.utils.crawl4ai_scraper import scrape_urls_efficiently, URLMetadata
    _SCRAPER_UTILS_AVAILABLE = True
except ImportError as e:
    _logger.error(f"Failed to import scraper utilities: {e}. Worker cannot function.")
    _SCRAPER_UTILS_AVAILABLE = False
    class URLMetadata(dict): pass
# --- End import ---

# Global message broker instance
_message_broker = None

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, _frame):
    """Handle shutdown signals gracefully."""
    if not shutdown_event.is_set():
        _logger.warning(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    else:
        _logger.debug(f"Received signal {sig}, but shutdown already in progress.")

async def handle_async_shutdown(sig):
     """Async compatible shutdown handler"""
     if not shutdown_event.is_set():
         _logger.warning(f"Received signal {sig}, initiating async shutdown...")
         shutdown_event.set()
     else:
         _logger.debug(f"Received signal {sig}, but shutdown already in progress.")

def preload_components():
    """Preload and initialize key components at startup."""
    global _message_broker

    if not _SCRAPER_UTILS_AVAILABLE:
        _logger.critical("Scraper utilities (crawl4ai_scraper.py) not available or failed to import. Scraper worker cannot start.")
        sys.exit(1)

    # Initialize Redis message broker
    redis_host = app_config.redis.HOST
    redis_port = str(app_config.redis.PORT)
    redis_url = f"redis://{redis_host}:{redis_port}"

    stream_name = getattr(app_config.redis, 'SCRAPE_QUEUE_NAME', 'vsg_scrape_jobs')
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=stream_name)

    # Create consumer group if it doesn't exist
    group_name = getattr(app_config.redis, 'SCRAPER_CONSUMER_GROUP', 'scraper-workers')
    try:
        _message_broker.create_consumer_group(group_name=group_name)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            _logger.warning(f"Could not create consumer group '{group_name}': {e}")
        else:
            _logger.debug(f"Consumer group '{group_name}' already exists.")

    _message_broker.ensure_stream_exists(stream_name)

    _logger.info(f"Scraper worker components initialized successfully. Stream: '{stream_name}', Group: '{group_name}'")

def get_message_broker() -> RedisMessageBroker:
    """Get the pre-initialized message broker or create a new one if needed."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    _logger.warning("Message broker accessed before preload, initializing now.")
    preload_components()
    if _message_broker is None:
        _logger.critical("Failed to initialize message broker on demand. Exiting.")
        sys.exit(1)
    return _message_broker

async def process_scrape_job(job_data: dict) -> bool:
    """
    Process a single scraping job using the efficient scrape_urls_efficiently function.
    Assumes job_data is a decoded dictionary containing job parameters.
    Returns True on success (job processed, status updated), False on critical failure to update status.
    """
    _logger.debug("PROCESS_SCRAPE_JOB_COROUTINE_STARTED")
    job_id_for_log = "UNKNOWN_JOB_ID"

    try:
        if isinstance(job_data, dict):
            job_id_for_log = job_data.get("job_id", "JOB_ID_NOT_IN_DICT")
            _logger.debug(f"PROCESS_SCRAPE_JOB_DATA_ACCESS_INIT: Job ID ~{job_id_for_log}. Data type: {type(job_data)}. Keys: {list(job_data.keys())}")
        else:
            job_id_for_log = "JOB_DATA_NOT_A_DICT"
            _logger.debug(f"PROCESS_SCRAPE_JOB_DATA_ACCESS_INIT: Job data is not a dict. Type: {type(job_data)}. Value: {str(job_data)[:200]}")
            try:
                mb_early_fail = get_message_broker()
                placeholder_id = job_data.get("message_id_for_tracking", f"unknown-job-{uuid.uuid4().hex[:4]}") if isinstance(job_data, dict) else f"unknown-job-{uuid.uuid4().hex[:4]}"
                mb_early_fail.track_job_progress(placeholder_id, "failed", {"error": f"Critical worker error: job_data was not a dictionary (type: {type(job_data)})"})
            except Exception as e_track_critical_early_fail:
                _logger.error(f"Failed to track critical early failure for non-dict job_data: {e_track_critical_early_fail}")
            return False
    except Exception as e_early_data_access:
        _logger.debug(f"PROCESS_SCRAPE_JOB_EARLY_DATA_ACCESS_ERROR: Job ID was ~{job_id_for_log}. Error: {e_early_data_access}")
        try:
            mb_early_error = get_message_broker()
            error_job_id = job_id_for_log if job_id_for_log not in ["UNKNOWN_JOB_ID", "JOB_ID_NOT_IN_DICT", "JOB_DATA_NOT_A_DICT"] else f"unknown-job-early-err-{uuid.uuid4().hex[:4]}"
            mb_early_error.track_job_progress(error_job_id, "failed", {"error": f"Worker critical error during initial data access: {str(e_early_data_access)}"})
        except Exception as e_track_early_err_fail:
            _logger.error(f"Failed to track critical early data access failure for job {job_id_for_log}: {e_track_early_err_fail}")
        return False

    if not _SCRAPER_UTILS_AVAILABLE:
        _logger.error("scrape_urls_efficiently is not available. Cannot process job.")
        job_id_temp = job_data.get("job_id", f"unknown-job-{uuid.uuid4().hex[:4]}")
        try:
            mb = get_message_broker()
            mb.track_job_progress(job_id_temp, "failed", {"error": "Scraper utilities not available"})
        except Exception as e_track:
            _logger.error(f"Failed to track critical failure for job {job_id_temp}: {e_track}")
        return False

    start_time = time.time()

    job_id = job_data.get("job_id")
    if not job_id:
        job_id = f"temp-job-{uuid.uuid4().hex[:8]}"
        _logger.warning(f"Job received without 'job_id', assigned temporary ID: {job_id}")

    message_broker = get_message_broker()

    try:
        message_broker.track_job_progress(job_id, "processing", {"message": "Efficient scraping job started"})
        _logger.info(f"Processing efficient scrape job {job_id}...")
    except Exception as e:
        _logger.error(f"Failed to update initial processing status for job {job_id}: {e}")

    scraped_results_metadata: List[URLMetadata] = []
    final_status = "failed"
    error_message_summary = "Unknown processing error"
    job_processed_successfully = False

    try:
        urls_raw = job_data.get("urls")
        urls = None

        if isinstance(urls_raw, str):
            try:
                urls = json.loads(urls_raw)
                _logger.debug(f"Job {job_id}: Parsed 'urls' from JSON string to list.")
            except json.JSONDecodeError:
                _logger.error(f"Job {job_id}: Failed to parse 'urls' field from JSON string: {urls_raw[:200]}...")
        elif isinstance(urls_raw, list):
            urls = urls_raw
        else:
            _logger.error(f"Job {job_id}: 'urls' field is of unexpected type: {type(urls_raw)}. Expected string or list. Value: {str(urls_raw)[:200]}...")

        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) and url.strip() for url in urls):
            error_detail = f"Expected a list of non-empty strings. Got type: {type(urls)}, value: {str(urls)[:200]}..."
            error_message_summary = f"Invalid URL format after parsing. {error_detail}"
            _logger.error(f"Invalid input for scrape job {job_id}: {error_message_summary}")
            raise ValueError(error_message_summary)

        # --- Extract configurations from job_data ---
        def _get_config_dict(key: str) -> Optional[dict]:
            val = job_data.get(key)
            if isinstance(val, str):
                try:
                    parsed_val = json.loads(val)
                    if isinstance(parsed_val, dict):
                        return parsed_val
                    _logger.warning(f"Job {job_id}: Config '{key}' was a string but not a JSON dict: '{val[:100]}...'")
                    return None
                except json.JSONDecodeError:
                    _logger.warning(f"Job {job_id}: Could not parse JSON string for config '{key}': '{val[:100]}...'")
                    return None
            elif isinstance(val, dict):
                return val
            elif val is None:
                return None
            else:
                _logger.warning(f"Job {job_id}: Config '{key}' has unexpected type: {type(val)}. Expected dict, string, or None.")
                return None

        browser_config_dict = _get_config_dict("browser_config")
        run_config_dict = _get_config_dict("run_config")
        dispatcher_config_dict = _get_config_dict("dispatcher_config")
        user_query_for_bm25 = job_data.get("user_query_for_bm25")
        if user_query_for_bm25 is not None and not isinstance(user_query_for_bm25, str):
            _logger.warning(f"Job {job_id}: user_query_for_bm25 is not a string, ignoring. Type: {type(user_query_for_bm25)}")
            user_query_for_bm25 = None

        _logger.info(f"Job {job_id}: Calling scrape_urls_efficiently for {len(urls)} URLs.")
        message_broker.track_job_progress(
            job_id,
            "processing",
            {"message": f"Invoking Crawl4AI efficient scrape for {len(urls)} URLs", "progress": 10}
        )

        # --- Perform Actual Scraping ---
        scraped_results_metadata = await scrape_urls_efficiently(
            urls_to_scrape=urls,
            browser_config_dict=browser_config_dict,
            run_config_dict=run_config_dict,
            dispatcher_config_dict=dispatcher_config_dict,
            user_query_for_bm25=user_query_for_bm25
        )

        # --- Evaluate Results ---
        if not scraped_results_metadata:
            error_message_summary = "Efficient scraping function returned no results (empty list)."
            _logger.error(f"Scrape job {job_id} failed: {error_message_summary}")
        else:
            successful_scrapes_count = sum(1 for item in scraped_results_metadata if not item.error and (item.markdown_content or item.title))
            if successful_scrapes_count > 0:
                job_processed_successfully = True
                final_status = "completed"
                error_message_summary = f"Efficient scrape completed. {successful_scrapes_count}/{len(urls)} URLs yielded content."
                _logger.info(f"Scrape job {job_id} completed successfully. URLs with content: {successful_scrapes_count}/{len(urls)}.")
            else:
                # Processed, but no content or all errors
                final_status = "completed"
                error_message_summary = f"Efficient scraping finished, but no usable content extracted from {len(urls)} URLs or all had errors."
                _logger.warning(f"Scrape job {job_id} processed, but: {error_message_summary}")

    except ValueError as ve:
        _logger.error(f"Input validation failed for efficient scrape job {job_id}: {ve}")
        error_message_summary = str(ve)
        final_status = "failed"
    except ImportError as ie:
         _logger.critical(f"Job {job_id} failed due to ImportError (likely Crawl4AI or dependency): {ie}")
         error_message_summary = str(ie)
         final_status = "failed"
    except Exception as e:
        _logger.exception(f"Unexpected error processing efficient scrape job {job_id}: {e}")
        error_message_summary = f"Unexpected scraping error: {str(e)}"
        final_status = "failed"

    # --- Update Final Status in Redis ---
    processing_time = time.time() - start_time
    try:
        results_as_dicts = []
        if scraped_results_metadata:
            for item in scraped_results_metadata:
                if hasattr(item, 'model_dump'):
                    results_as_dicts.append(item.model_dump(mode='json'))
                elif isinstance(item, dict):
                    results_as_dicts.append(item)
                else:
                    _logger.warning(f"Job {job_id}: Cannot serialize item of type {type(item)} for results.")
                    results_as_dicts.append({"url": str(getattr(item, 'url', 'unknown')), "error": "Serialization failed for this item"})

        status_details = {
            "message": str(error_message_summary if final_status != "completed" or not job_processed_successfully else "Efficient scraping completed successfully."),
            "error": str(error_message_summary) if final_status == "failed" or (final_status == "completed" and not job_processed_successfully) else None,
            "processing_time": float(processing_time),
            "urls_processed_count": len(scraped_results_metadata if scraped_results_metadata else []),
            "data": results_as_dicts
        }

        message_broker.track_job_progress(job_id, final_status, status_details)
        _logger.info(f"Finished processing job {job_id} with status '{final_status}' in {processing_time:.2f}s. URLs processed: {len(results_as_dicts)}.")

    except Exception as e_track:
        _logger.error(f"CRITICAL: Failed to update final job status for {job_id}: {e_track}")
        return False

    return job_processed_successfully

async def process_scraper_jobs(consumer_name: str):
    """Process scraper jobs from the Redis stream using the new efficient method."""
    message_broker = get_message_broker()
    loop = asyncio.get_running_loop()

    group_name = getattr(app_config.redis, 'SCRAPER_CONSUMER_GROUP', 'scraper-workers')
    batch_size = app_config.redis.WORKER_BATCH_SIZE
    sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent_jobs = app_config.redis.WORKER_MAX_CONCURRENT or app_config.scraper.WORKER_MAX_CONCURRENT_JOBS
    active_tasks = set()

    _logger.info(f"Scraper worker '{consumer_name}' starting job consumption loop for efficient scrapes.")
    _logger.info(f"Config - Group: {group_name}, BatchSize: {batch_size}, SleepInterval: {sleep_interval}s, MaxConcurrentJobs: {max_concurrent_jobs}")

    while not shutdown_event.is_set():
        # If all concurrent job slots are filled, wait for one task to complete
        if len(active_tasks) >= max_concurrent_jobs and active_tasks:
            _logger.debug(f"Concurrency limit ({max_concurrent_jobs} jobs) reached. Waiting for tasks to complete...")
            done, active_tasks_after_wait = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=sleep_interval # Add timeout to periodically check shutdown_event
            )
            active_tasks = active_tasks_after_wait
            if done:
                 _logger.debug(f"{len(done)} job task(s) completed, freeing slots.")
            if shutdown_event.is_set(): break
            if len(active_tasks) >= max_concurrent_jobs: continue

        if shutdown_event.is_set(): break

        try:
            # Fetch new messages from the stream
            num_to_fetch = min(batch_size, max_concurrent_jobs - len(active_tasks))
            if num_to_fetch <= 0:
                await asyncio.sleep(0.1)
                continue

            messages = await loop.run_in_executor(
                None,
                message_broker.consume_messages,
                group_name,
                consumer_name,
                num_to_fetch,
                2000 # block
            )

            if not messages:
                if not active_tasks:
                    await asyncio.sleep(sleep_interval)
                continue

            for stream_name, stream_messages in messages:
                if shutdown_event.is_set(): break

                for message_id_bytes, message_data_raw_bytes in stream_messages:
                    if shutdown_event.is_set(): break
                    message_id = message_id_bytes.decode('utf-8') if isinstance(message_id_bytes, bytes) else message_id_bytes

                    try:
                        # Decode message data from bytes to string for keys and values
                        job_data = {
                            (k.decode('utf-8') if isinstance(k, bytes) else k):
                            (v.decode('utf-8') if isinstance(v, bytes) else v)
                            for k, v in message_data_raw_bytes.items()
                        }
                    except Exception as decode_err:
                        _logger.error(f"Failed to decode message {message_id} data: {decode_err}. Acknowledging and skipping.")
                        await loop.run_in_executor(None, message_broker.acknowledge_message, group_name, message_id)
                        continue

                    # IMPORTANT: Filter for the new message type
                    message_type = job_data.get("message_type")
                    if message_type != 'scrape_request_efficient':
                        _logger.debug(f"Skipping message {message_id} (Type: {message_type or 'N/A'}). Expected 'scrape_request_efficient'.")
                        await loop.run_in_executor(None, message_broker.acknowledge_message, group_name, message_id)
                        continue

                    job_id = job_data.get("job_id")
                    if not job_id:
                        _logger.error(f"Message {message_id} of type 'scrape_request_efficient' is missing 'job_id'. Acknowledging and skipping.")
                        await loop.run_in_executor(None, message_broker.acknowledge_message, group_name, message_id)
                        continue

                    # Add message_id to job_data if not present, for tracking in case job_id is problematic later
                    if "message_id_for_tracking" not in job_data:
                        job_data["message_id_for_tracking"] = message_id

                    urls_in_job = job_data.get("urls")
                    if not urls_in_job:
                        log_msg = f"Message {message_id} (Job {job_id}) has no 'urls' field or it is empty. Acknowledging and skipping."
                        _logger.warning(log_msg)
                        try:
                            await loop.run_in_executor(
                                None,
                                message_broker.track_job_progress,
                                job_id,
                                "failed",
                                {"error": "Invalid job data: Missing or empty 'urls' field"}
                            )
                            _logger.info(f"Marked job {job_id} (from message {message_id}) as failed due to missing URLs.")
                        except Exception as e_track_fail:
                            _logger.error(f"Failed to track failure for job {job_id} (message {message_id}): {e_track_fail}")
                        await loop.run_in_executor(None, message_broker.acknowledge_message, group_name, message_id)
                        continue

                    if len(active_tasks) >= max_concurrent_jobs:
                         _logger.warning(f"Concurrency limit ({max_concurrent_jobs}) reached unexpectedly before processing {message_id}. Will retry fetching.")
                         break

                    _logger.debug(f"Creating task for efficient scrape message {message_id} (Job ID: {job_id}). Raw job_data being passed: {str(job_data)[:500]}...") # Log data
                    task = asyncio.create_task(process_scrape_job(job_data))
                    task.set_name(f"scrape_job_{job_id}")
                    _logger.debug(f"Task object created: {task} for job {job_id} (message {message_id})") # Log task object

                    async def do_acknowledge_async(mb_instance, current_loop, ack_msg_id, ack_grp_name, ack_j_id):
                        try:
                            await current_loop.run_in_executor(None, mb_instance.acknowledge_message, ack_grp_name, ack_msg_id)
                            _logger.debug(f"Async acknowledged message {ack_msg_id} (Job ID: {ack_j_id}) after task processing.")
                        except Exception as e_ack_async:
                            _logger.error(f"Error during async acknowledgment of message {ack_msg_id} (Job ID: {ack_j_id}): {e_ack_async}")

                    def ack_callback(fut, msg_id=message_id, grp_name=group_name, j_id=job_id):
                        # Enhanced ack_callback logging
                        _logger.debug(f"ACK_CALLBACK_ENTERED for Job ID: {j_id}, Msg ID: {msg_id}. Future state: done={fut.done()}, cancelled={fut.cancelled()}")
                        job_status_updated_successfully = False # Tracks if process_scrape_job returned True
                        try:
                            # This is the critical line that re-raises an exception if the task failed.
                            result_processed_ok = fut.result() # This is the return value of process_scrape_job
                            job_status_updated_successfully = result_processed_ok

                            _logger.debug(f"ACK_CALLBACK: fut.result() obtained: {result_processed_ok} for Job ID: {j_id}")
                            if result_processed_ok:
                                _logger.info(f"Task for message {msg_id} (Job ID: {j_id}) completed, and job status update was successful. Scheduling acknowledgement.")
                            else:
                                 _logger.error(f"Task for message {msg_id} (Job ID: {j_id}) completed, but reported FAILURE TO UPDATE JOB STATUS (process_scrape_job returned False). Scheduling acknowledgement to prevent re-delivery, but data loss for job status might have occurred.")

                            asyncio.create_task(do_acknowledge_async(message_broker, loop, msg_id, grp_name, j_id))

                        except asyncio.CancelledError:
                            _logger.warning(f"Task for message {msg_id} (Job ID: {j_id}) was CANCELLED. Not acknowledging message {msg_id}.")
                        except Exception as e_task_exception:
                            _logger.error(f"Task for message {msg_id} (Job ID: {j_id}) FAILED WITH UNHANDLED EXCEPTION: {type(e_task_exception).__name__} - {str(e_task_exception)}. Not acknowledging message {msg_id}.")
                            # import traceback
                            # _logger.error(f"Traceback for failed task {j_id} (Msg ID: {msg_id}):\\n{traceback.format_exc()}")
                        finally:
                            _logger.debug(f"ACK_CALLBACK_EXITED for Job ID: {j_id}, Msg ID: {msg_id}. Task status updated successfully: {job_status_updated_successfully}")
                    task.add_done_callback(ack_callback)
                    active_tasks.add(task)
                    _logger.debug(f"Task added for job {job_id}. Active tasks: {len(active_tasks)}")

                if shutdown_event.is_set() or (len(active_tasks) >= max_concurrent_jobs and stream_messages):
                    break

        except asyncio.CancelledError:
            _logger.info(f"Scraper job consumption loop for '{consumer_name}' cancelled.")
            break
        except Exception as e_loop:
            _logger.exception(f"Error in main consumption loop for '{consumer_name}': {e_loop}")
            if shutdown_event.is_set(): break
            await asyncio.sleep(sleep_interval * 2)

    # Cleanup during shutdown
    if active_tasks:
        _logger.info(f"Shutdown initiated for '{consumer_name}'. Waiting for {len(active_tasks)} active job tasks to complete...")
        try:
            await asyncio.wait(active_tasks, timeout=app_config.scraper.WORKER_SHUTDOWN_TIMEOUT)
        except asyncio.TimeoutError:
            _logger.warning(f"Some tasks for '{consumer_name}' did not complete within the {app_config.scraper.WORKER_SHUTDOWN_TIMEOUT}s shutdown timeout.")
        except Exception as e_shutdown_wait:
             _logger.error(f"Error during shutdown task waiting for '{consumer_name}': {e_shutdown_wait}")

    _logger.info(f"Scraper job consumption loop for '{consumer_name}' finished.")

async def run_worker():
    """Run the scraper worker with graceful shutdown handling."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sys.platform != 'win32':
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_async_shutdown(s)))
        else:
            signal.signal(sig, handle_shutdown)

    consumer_name = f"scraper-worker-eff-{uuid.uuid4().hex[:8]}"
    _logger.info(f"Starting efficient scraper worker (PID: {os.getpid()}, Consumer: {consumer_name})...")

    preload_components()

    try:
        await process_scraper_jobs(consumer_name)
    except asyncio.CancelledError:
        _logger.info(f"Run_worker for {consumer_name} was cancelled.")
    finally:
        _logger.info(f"Run_worker for {consumer_name} is shutting down.")

def main():
    """Entry point for the worker process."""
    log_level_name = getattr(app_config, 'LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    #todo:  For robustness, especially for a standalone worker script, explicit basicConfig is good.
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [scrape_worker] - %(message)s',
        stream=sys.stdout
    )
    _logger.info(f"Logging configured for Scrape Worker with level: {logging.getLevelName(log_level)}")

    if not app_config.redis.ENABLED:
        _logger.error("Redis is disabled in configuration. Enable REDIS_ENABLED=True to use the scrape worker.")
        sys.exit(1)

    if not _SCRAPER_UTILS_AVAILABLE:
        _logger.critical("Scraper utilities (crawl4ai_scraper.py) not available. Scraper worker cannot run.")
        sys.exit(1)

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        _logger.info("Efficient Scraper Worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"Efficient Scraper Worker failed unexpectedly: {e}")
    finally:
        _logger.info("Shutting down event loop for Efficient Scraper Worker...")
        try:
            # Wait for all async generators to finish
            if hasattr(loop, "shutdown_asyncgens"):
                 loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError as e_gens:
            if ("Cannot run the event loop while another loop is running" not in str(e_gens) and \
               "Event loop is closed" not in str(e_gens)):
                 _logger.warning(f"Error during async generator shutdown: {e_gens}")
        finally:
            if loop.is_running():
                 _logger.info("Stopping event loop...")
                 loop.stop()
            if not loop.is_closed():
                 _logger.info("Closing event loop...")
                 loop.close()
            _logger.info("Event loop closed for Efficient Scraper Worker.")

if __name__ == "__main__":
    main()