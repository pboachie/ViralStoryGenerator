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
from viralStoryGenerator.utils.crawl4ai_scraper import scrape_urls
try:
    from crawl4ai import BrowserConfig, CrawlerRunConfig
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _logger.error("Crawl4AI library not found. Worker cannot function without it. Run: pip install crawl4ai")
    _CRAWL4AI_AVAILABLE = False
    class BrowserConfig: pass
    class CrawlerRunConfig: pass
# --- End import ---

# Global message broker instance
_message_broker = None

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, _frame):
    """Handle shutdown signals gracefully."""
    _logger.warning(f"Received signal {sig}, initiating shutdown...")
    shutdown_event.set()

    # Give tasks time to complete
    _logger.info("Waiting for current tasks to complete...")
    # TODO: Consider a more sophisticated shutdown wait if needed
    # For now, a simple sleep
    # A better approach might involve tracking active tasks and awaiting them.
    time.sleep(5)

    _logger.info("Shutdown complete.")
    sys.exit(0)

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

async def process_scrape_job(job_data):
    """
    Process a single scraping job by calling the actual scrape_urls function.
    """
    start_time = time.time()

    # Unpack the message structure if it is a list
    if isinstance(job_data, list) and len(job_data) == 2:
        stream_name, messages = job_data
        if isinstance(messages, list) and len(messages) > 0:
            job_data = messages[0][1]  # Extract the dictionary from the first message

    # Ensure job_data is a dictionary
    if not isinstance(job_data, dict):
        _logger.error(f"Expected job_data to be a dictionary but got: {job_data}")
        return False

    if isinstance(job_data, list):
        _logger.error(f"Expected job_data to be a dictionary but got a list: {job_data}")
        return False

    job_id = job_data.get("job_id")
    if not job_id:
        job_id = str(uuid.uuid4())
        _logger.warning(f"Job received without ID, assigned: {job_id}")

    message_broker = get_message_broker()

    # Update job status to processing
    message_broker.track_job_progress(job_id, "processing", {"message": "Scraping job started"})
    _logger.info(f"Processing scrape job {job_id}...")

    scrape_result_data = None
    final_status = "failed"
    error_message = "Unknown processing error"
    scrape_successful = False

    try:
        # --- Extract job parameters ---
        urls = job_data.get("urls", [])
        if isinstance(urls, str):
            try:
                 urls = json.loads(urls)
            except json.JSONDecodeError:
                urls = [urls]
        # Validate URLs
        if not urls or not isinstance(urls, list) or not all(isinstance(url, str) and url.strip() for url in urls):
            error_message = "Invalid or empty URLs provided for scraping. Ensure all URLs are non-empty strings."
            _logger.error(f"Invalid input for scrape job {job_id}: {error_message}")
            raise ValueError(error_message)

        browser_config_dict = job_data.get("browser_config")
        run_config_dict = job_data.get("run_config")

        # Convert config from JSON string if needed
        if isinstance(browser_config_dict, str):
            try:
                browser_config_dict = json.loads(browser_config_dict)
            except json.JSONDecodeError:
                _logger.warning(f"Job {job_id}: Could not parse browser_config JSON string.")
                browser_config_dict = None
        if isinstance(run_config_dict, str):
             try:
                 run_config_dict = json.loads(run_config_dict)
             except json.JSONDecodeError:
                  _logger.warning(f"Job {job_id}: Could not parse run_config JSON string.")
                  run_config_dict = None

        # --- Instantiate Config Objects ---
        browser_config = None
        run_config = None
        if isinstance(browser_config_dict, dict):
            try:
                browser_config = BrowserConfig(**browser_config_dict)
            except Exception as e:
                 _logger.warning(f"Job {job_id}: Failed to create BrowserConfig from dict: {e}. Using defaults.")
        if isinstance(run_config_dict, dict):
            try:
                run_config = CrawlerRunConfig(**run_config_dict)
            except Exception as e:
                _logger.warning(f"Job {job_id}: Failed to create CrawlerRunConfig from dict: {e}. Using defaults.")

        # --- Perform Actual Scraping ---
        message_broker.track_job_progress(
            job_id,
            "processing",
            {"message": f"Starting Crawl4AI scrape for {len(urls)} URLs", "progress": 10}
        )

        # Call the scraping function from crawl4ai_scraper
        scrape_result_data = await scrape_urls(
            urls=urls,
            browser_config=browser_config,
            run_config=run_config
        )

        # --- Evaluate Results ---
        if not scrape_result_data:
             error_message = "Scraping returned no results."
             _logger.error(f"Scrape job {job_id} failed: {error_message}")
        elif any(content is not None for _, content in scrape_result_data):
             scrape_successful = True
             final_status = "completed"
             error_message = None
             _logger.info(f"Scrape job {job_id} completed. URLs processed: {len(scrape_result_data)}.")
        else:
             error_message = "Scraping finished, but no content was extracted from any URL."
             _logger.error(f"Scrape job {job_id} failed: {error_message}")

    except ValueError as ve:
        _logger.error(f"Invalid input for scrape job {job_id}: {ve}")
        error_message = str(ve)
        final_status = "failed"
    except ImportError:
         # This case should ideally be caught earlier, but good to handle
         _logger.critical(f"Job {job_id} failed: Crawl4AI library not available.")
         error_message = "Crawl4AI library not installed in worker environment."
         final_status = "failed"
    except Exception as e:
        _logger.exception(f"Error processing scrape job {job_id}: {e}")
        error_message = f"Unexpected scraping error: {str(e)}"
        final_status = "failed"

    # --- Update Final Status in Redis ---
    processing_time = time.time() - start_time

    try:
        status_details = {
            "message": str("Scraping completed successfully" if scrape_successful else error_message),
            "error": str(error_message) if error_message else "",
            "processing_time": float(processing_time)
        }

        if scrape_result_data:
            try:
                if isinstance(scrape_result_data, (list, dict)):
                    status_details["urls_processed"] = len(scrape_result_data)
                    status_details["data"] = json.dumps(scrape_result_data)
                else:
                    status_details["urls_processed"] = 0
                    status_details["data"] = str(scrape_result_data)
            except Exception as e:
                _logger.warning(f"Could not serialize scrape result data: {e}")
                status_details["urls_processed"] = 0
                status_details["data"] = "Data serialization failed"
        else:
            status_details["urls_processed"] = 0

        message_broker.track_job_progress(job_id, final_status, status_details)
        _logger.info(f"Finished processing job {job_id} with status '{final_status}' in {processing_time:.2f}s")
    except Exception as e:
        _logger.error(f"Failed to update job status: {e}")
        # Try one more time with minimal data
        try:
            message_broker.track_job_progress(
                job_id,
                "failed",
                {"message": "Job status update failed", "error": str(e)}
            )
        except Exception as e:
            _logger.error(f"Failed to send even minimal status update: {e}")

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
                active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=None  # Use None for no timeout
            )
            if done:
                _logger.debug(f"{len(done)} task(s) completed.")

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
                # No more messages - sleep before checking again
                _logger.debug("No new messages available. Sleeping...")
                await asyncio.sleep(sleep_interval)
                continue

            for stream_name, stream_messages in messages:
                if shutdown_event.is_set():
                    break

                for message_id, message_data in stream_messages:
                    if isinstance(message_data, dict):
                        job_data = {
                            k.decode() if isinstance(k, bytes) else k:
                            v.decode() if isinstance(v, bytes) else v
                            for k, v in message_data.items()
                        }

                        # Check if this is an initialization or empty message
                        if "initialized" in job_data or "purged" in job_data:
                            _logger.debug(f"Skipping system message: {message_id}")
                            _message_broker.acknowledge_message(group_name, message_id)
                            continue

                        # Basic validation check for URLs
                        urls = job_data.get("urls")
                        if not urls:
                            _logger.warning(f"Message {message_id} has no URLs, acknowledging and skipping")
                            # Still acknowledge to prevent reprocessing
                            _message_broker.acknowledge_message(group_name, message_id)
                            if "job_id" in job_data:
                                _message_broker.track_job_progress(
                                    job_data["job_id"],
                                    "failed",
                                    {"error": "Invalid or empty URLs provided for scraping"}
                                )
                            continue

                    if len(active_tasks) < max_concurrent:
                        task = asyncio.create_task(process_scrape_job([stream_name, [(message_id, message_data)]]))
                        task.add_done_callback(lambda t: _message_broker.acknowledge_message(group_name, message_id)
                                              if not t.exception() else None)
                        active_tasks.add(task)
                        _logger.debug(f"Created new task for message {message_id}. Active tasks: {len(active_tasks)}")
                    else:
                        _logger.debug("All concurrent slots filled. Waiting for a slot to become available...")
                        await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

        except Exception as e:
            _logger.exception(f"Error processing messages: {e}")
            await asyncio.sleep(sleep_interval * 2)  # Retry after longer interval

    # Cleanup during shutdown
    if active_tasks:
        _logger.info("Waiting for any remaining tasks to complete during shutdown...")
        try:
            await asyncio.wait(active_tasks, timeout=15.0)
        except asyncio.TimeoutError:
            _logger.warning("Some tasks did not complete within the timeout period during shutdown.")

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
     _logger.warning(f"Received signal {sig}, initiating async shutdown...")
     shutdown_event.set()
     # Optional: Add further async cleanup here if needed


def main():
    """Entry point for the worker process."""
    if os.name == 'nt':  # Windows
        # Use ProactorEventLoop for Windows compatibility with asyncio subprocesses if needed
        # asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        # For basic Redis/requests, default Selector loop is usually fine
        pass

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        _logger.info("Worker stopped by KeyboardInterrupt.")
    except Exception as e:
         _logger.exception(f"Worker failed unexpectedly: {e}")
    finally:
         _logger.info("Scraper worker process exiting.")


if __name__ == "__main__":
    main()
