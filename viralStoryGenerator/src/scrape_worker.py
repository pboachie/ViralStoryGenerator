# viralStoryGenerator/src/scrape_worker.py
"""
Scraper Worker for processing scraping requests via Redis Streams.
This module consumes requests published to the scraper stream.
"""
import asyncio
import signal
import sys
import time
import json
import os
import uuid

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker

# --- Import scraping function and config models ---
try:
    from crawl4ai import BrowserConfig as Crawl4AI_BrowserConfig, CrawlerRunConfig as Crawl4AI_CrawlerRunConfig
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _logger.error("Crawl4AI library not found. Worker cannot function without it. Run: pip install crawl4ai")
    _CRAWL4AI_AVAILABLE = False
    class Crawl4AI_BrowserConfig: pass
    class Crawl4AI_CrawlerRunConfig: pass
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

def preload_components():
    """Preload and initialize key components at startup."""
    global _message_broker

    if not _CRAWL4AI_AVAILABLE:
        _logger.critical("Crawl4AI is not installed. Scraper worker cannot start.")
        sys.exit(1)

    # Initialize Redis message broker
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name="scraper_jobs")

    # Create consumer group if it doesn't exist
    try:
        _message_broker.create_consumer_group(group_name="scraper_worker_group")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            _logger.warning(f"Could not create consumer group 'scraper_worker_group': {e}")
        else:
            _logger.debug("Consumer group 'scraper_worker_group' already exists.")

    # Ensure stream exists
    _message_broker.ensure_stream_exists("scraper_jobs")

    _logger.info("Scraper worker components initialized successfully")

def get_message_broker() -> RedisMessageBroker:
    """Get the pre-initialized message broker or create a new one if needed."""
    global _message_broker
    if _message_broker is not None:
        return _message_broker

    # Initialize if not already done (should have been done by preload)
    _logger.warning("Message broker accessed before preload, initializing now.")
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name="scraper_jobs")
    return _message_broker

async def process_scrape_job(job_data: dict):
    """
    Process a single scraping job (received as a dictionary) by calling the actual scrape_urls function.
    Assumes job_data is a decoded dictionary containing job parameters.
    Returns True on success, False on handled failure within this function.
    Raises exceptions for unexpected errors.
    """
    from viralStoryGenerator.utils.crawl4ai_scraper import scrape_urls
    if _CRAWL4AI_AVAILABLE:
        from crawl4ai import BrowserConfig, CrawlerRunConfig
    else:
        BrowserConfig = Crawl4AI_BrowserConfig
        CrawlerRunConfig = Crawl4AI_CrawlerRunConfig

    start_time = time.time()

    job_id = job_data.get("job_id")
    if not job_id:
        job_id = f"temp-{uuid.uuid4().hex[:8]}"
        _logger.warning(f"Job received without 'job_id', assigned temporary ID: {job_id}")

    message_broker = get_message_broker()

    try:
        message_broker.track_job_progress(job_id, "processing", {"message": "Scraping job started"})
        _logger.info(f"Processing scrape job {job_id}...")
    except Exception as e:
        _logger.error(f"Failed to update initial processing status for job {job_id}: {e}")

    scrape_result_data = None
    final_status = "failed"
    error_message = "Unknown processing error"
    scrape_successful = False

    try:
        urls = job_data.get("urls")

        if isinstance(urls, str):
            try:
                 parsed_urls = json.loads(urls)
                 if isinstance(parsed_urls, list):
                     urls = parsed_urls
                 else:
                     urls = [urls]
            except json.JSONDecodeError:
                 urls = [urls]

        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) and url.strip() for url in urls):
            error_message = "Invalid URL format: Expected a list of non-empty strings."
            _logger.error(f"Invalid input for scrape job {job_id}: {error_message}")
            raise ValueError(error_message)

        browser_config_dict = job_data.get("browser_config")
        run_config_dict = job_data.get("run_config")

        # Convert config from JSON string if needed
        if isinstance(browser_config_dict, str):
            try:
                browser_config_dict = json.loads(browser_config_dict)
            except json.JSONDecodeError:
                _logger.warning(f"Job {job_id}: Could not parse browser_config JSON string. Using defaults.")
                browser_config_dict = None
        if isinstance(run_config_dict, str):
             try:
                 run_config_dict = json.loads(run_config_dict)
             except json.JSONDecodeError:
                  _logger.warning(f"Job {job_id}: Could not parse run_config JSON string. Using defaults.")
                  run_config_dict = None

        # --- Instantiate Config Objects ---
        browser_config = None
        run_config = None
        if isinstance(browser_config_dict, dict):
            try:
                if _CRAWL4AI_AVAILABLE:
                    browser_config = BrowserConfig(**browser_config_dict)
                else:
                    raise ImportError("Crawl4AI not available")
            except Exception as e:
                 _logger.warning(f"Job {job_id}: Failed to create BrowserConfig from dict: {e}. Using defaults.")
        if isinstance(run_config_dict, dict):
            try:
                 if _CRAWL4AI_AVAILABLE:
                     run_config = CrawlerRunConfig(**run_config_dict)
                 else:
                     raise ImportError("Crawl4AI not available")
            except Exception as e:
                _logger.warning(f"Job {job_id}: Failed to create CrawlerRunConfig from dict: {e}. Using defaults.")

        # --- Perform Actual Scraping ---
        message_broker.track_job_progress(
            job_id,
            "processing",
            {"message": f"Starting Crawl4AI scrape for {len(urls)} URLs", "progress": 10}
        )

        if not _CRAWL4AI_AVAILABLE:
             raise ImportError("Crawl4AI library not available.")

        # Call the scraping function from crawl4ai_scraper
        scrape_result_data = await scrape_urls(
            urls=urls,
            browser_config=browser_config,
            run_config=run_config
        )

        # --- Evaluate Results ---
        if scrape_result_data is None:
             error_message = "Scraping function returned None."
             _logger.error(f"Scrape job {job_id} failed: {error_message}")
        elif isinstance(scrape_result_data, list) and any(content is not None for _, content in scrape_result_data):
             scrape_successful = True
             final_status = "completed"
             error_message = None
             _logger.info(f"Scrape job {job_id} completed successfully. URLs processed: {len(scrape_result_data)}.")
        else:
             error_message = "Scraping finished, but no content was extracted from any URL."
             _logger.error(f"Scrape job {job_id} failed: {error_message}")

    except ValueError as ve:
        _logger.error(f"Input validation failed for scrape job {job_id}: {ve}")
        error_message = str(ve)
        final_status = "failed"
    except ImportError as ie:
         _logger.critical(f"Job {job_id} failed: {ie}")
         error_message = str(ie)
         final_status = "failed"
    except Exception as e:
        _logger.exception(f"Unexpected error processing scrape job {job_id}: {e}")
        error_message = f"Unexpected scraping error: {str(e)}"
        final_status = "failed"
        raise

    # --- Update Final Status in Redis ---
    processing_time = time.time() - start_time
    try:
        status_details = {
            "message": str("Scraping completed successfully" if scrape_successful else error_message or "Processing finished with errors."),
            "error": str(error_message) if error_message else None,
            "processing_time": float(processing_time),
            "urls_processed": 0
        }

        if scrape_result_data is not None:
            try:
                if isinstance(scrape_result_data, list):
                    status_details["urls_processed"] = len(scrape_result_data)
                    try:
                        status_details["data"] = json.dumps(scrape_result_data)
                    except TypeError as json_err:
                        _logger.warning(f"Job {job_id}: Could not serialize scrape result data to JSON: {json_err}")
                        status_details["data"] = "[Data not serializable]"
                else:
                     _logger.warning(f"Job {job_id}: Scrape result was not a list (type: {type(scrape_result_data)}). Storing as string.")
                     status_details["data"] = str(scrape_result_data)

            except Exception as e:
                _logger.error(f"Job {job_id}: Error processing scrape result data for status update: {e}")
                status_details["data"] = "[Error processing result data]"

        # Update the final status
        message_broker.track_job_progress(job_id, final_status, status_details)
        _logger.info(f"Finished processing job {job_id} with status '{final_status}' in {processing_time:.2f}s")

    except Exception as e:
        _logger.error(f"CRITICAL: Failed to update final job status for {job_id}: {e}")
        return False

    return scrape_successful

async def process_scraper_jobs(consumer_name: str):
    """Process scraper jobs from the Redis stream."""
    group_name = "scraper_worker_group"
    batch_size = app_config.redis.WORKER_BATCH_SIZE
    sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent = app_config.redis.WORKER_MAX_CONCURRENT
    active_tasks = set()

    _logger.info(f"Scraper worker '{consumer_name}' starting job consumption loop.")
    _logger.info(f"Config - BatchSize: {batch_size}, SleepInterval: {sleep_interval}s, MaxConcurrent: {max_concurrent}")

    while not shutdown_event.is_set():
        # If all concurrent slots are filled, wait for one task to complete
        if len(active_tasks) >= max_concurrent and active_tasks:
            _logger.debug(f"Concurrency limit ({max_concurrent}) reached. Waiting for tasks to complete...")
            done, active_tasks = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            if done:
                 _logger.debug(f"{len(done)} task(s) completed, freeing slots.")

        # Check if shutdown is requested after waiting
        if shutdown_event.is_set():
            break

        try:
            # Fetch new messages from the stream
            messages = _message_broker.consume_messages(
                group_name=group_name,
                consumer_name=consumer_name,
                count=batch_size,
                block=5000
            )

            if not messages:
                # _logger.debug("No new messages available. Sleeping...")
                await asyncio.sleep(sleep_interval)
                continue

            for stream_name, stream_messages in messages:
                if shutdown_event.is_set():
                    break

                for message_id, message_data_raw in stream_messages:
                    if shutdown_event.is_set():
                        break

                    try:
                        job_data = {
                            k.decode('utf-8') if isinstance(k, bytes) else k:
                            v.decode('utf-8') if isinstance(v, bytes) else v
                            for k, v in message_data_raw.items()
                        }
                    except Exception as decode_err:
                        _logger.error(f"Failed to decode message {message_id} data: {decode_err}. Acknowledging and skipping.")
                        _message_broker.acknowledge_message(group_name, message_id)
                        continue

                    message_type = job_data.get("message_type")
                    if message_type != 'scrape_request':
                        _logger.debug(f"Skipping non-scrape_request message: {message_id} (Type: {message_type or 'N/A'})")
                        _message_broker.acknowledge_message(group_name, message_id)
                        continue

                    urls = job_data.get("urls")
                    job_id = job_data.get("job_id")

                    if not urls:
                        log_msg = f"Message {message_id} has no 'urls' field or it is empty. Acknowledging and skipping."
                        _logger.warning(log_msg)

                        if job_id:
                            try:
                                _message_broker.track_job_progress(
                                    job_id,
                                    "failed",
                                    {"error": "Invalid job data: Missing or empty 'urls' field"}
                                )
                                _logger.info(f"Marked job {job_id} (from message {message_id}) as failed due to missing URLs.")
                            except Exception as e:
                                _logger.error(f"Failed to track failure for job {job_id} (message {message_id}): {e}")

                        _message_broker.acknowledge_message(group_name, message_id)
                        continue

                    if len(active_tasks) >= max_concurrent:
                         _logger.debug(f"Concurrency limit ({max_concurrent}) still reached before processing {message_id}. Waiting...")
                         done, active_tasks = await asyncio.wait(
                             active_tasks, return_when=asyncio.FIRST_COMPLETED
                         )
                         if done:
                             _logger.debug(f"{len(done)} task(s) completed, freeing slots.")
                         if shutdown_event.is_set():
                             break
                         if len(active_tasks) >= max_concurrent:
                              _logger.warning(f"Concurrency limit still reached after waiting? Skipping task creation for {message_id} this cycle.")
                              continue

                    _logger.debug(f"Creating task for message {message_id} (Job ID: {job_id or 'N/A'}).")
                    task = asyncio.create_task(process_scrape_job(job_data))

                    def ack_callback(fut, msg_id=message_id, grp_name=group_name, j_id=job_id):
                        try:
                            result = fut.result()
                            if result is False:
                                 _logger.warning(f"Task for message {msg_id} (Job ID: {j_id or 'N/A'}) indicated failure, but acknowledging as handled.")
                            _message_broker.acknowledge_message(grp_name, msg_id)
                            _logger.debug(f"Acknowledged message {msg_id} after task completion.")
                        except asyncio.CancelledError:
                            _logger.warning(f"Task for message {msg_id} (Job ID: {j_id or 'N/A'}) was cancelled. Not acknowledging.")
                        except Exception as e:
                            _logger.error(f"Task for message {msg_id} (Job ID: {j_id or 'N/A'}) failed with unhandled exception: {e}. Not acknowledging (will likely be redelivered).")

                    task.add_done_callback(ack_callback)
                    active_tasks.add(task)
                    _logger.debug(f"Task added. Active tasks: {len(active_tasks)}")


        except Exception as e:
            _logger.exception(f"Error in main consumption loop: {e}")
            await asyncio.sleep(sleep_interval * 2)

    # Cleanup during shutdown
    if active_tasks:
        _logger.info(f"Shutdown initiated. Waiting for {len(active_tasks)} active tasks to complete...")
        try:
            await asyncio.wait(active_tasks, timeout=30.0)
        except asyncio.TimeoutError:
            _logger.warning("Some tasks did not complete within the 30s shutdown timeout.")
        except Exception as e:
             _logger.error(f"Error during shutdown task waiting: {e}")


    _logger.info("Scraper job consumption loop finished.")

async def run_worker():
    """Run the scraper worker with graceful shutdown handling."""
    # Register signal handlers
    # Use loop.add_signal_handler on non-Windows for better async handling
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sys.platform != 'win32':
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_async_shutdown(s)))
        else:
            # Fallback for Windows
            signal.signal(sig, handle_shutdown)

    consumer_name = f"scraper-worker-{uuid.uuid4().hex[:8]}"
    _logger.info(f"Starting scraper worker (PID: {os.getpid()}, Consumer: {consumer_name})...")

    # Preload components
    preload_components() # Exits if crawl4ai is missing

    # Process jobs indefinitely
    await process_scraper_jobs(consumer_name)

async def handle_async_shutdown(sig):
     """Async compatible shutdown handler"""
     if not shutdown_event.is_set():
         _logger.warning(f"Received signal {sig}, initiating async shutdown...")
         shutdown_event.set()
     else:
         _logger.debug(f"Received signal {sig}, but shutdown already in progress.")

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