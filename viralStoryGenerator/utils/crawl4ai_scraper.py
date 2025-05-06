# viralStoryGenerator/utils/crawl4ai_scraper.py
"""Web scraping utilities using Crawl4AI with Redis Streams message broker."""
import asyncio
import os
from typing import List, Union, Optional, Tuple, Dict, Any
import time
import json
import uuid

# Use Crawl4AI library
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    from viralStoryGenerator.src.logger import logger as _logger
    _logger.error("Crawl4AI library not found. pip install crawl4ai")
    _CRAWL4AI_AVAILABLE = False
    class AsyncWebCrawler: pass
    class BrowserConfig: pass
    class CrawlerRunConfig: pass


from .redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.models.models import ScrapeJobRequest, ScrapeJobResult

_message_broker = None

# Function to get or initialize Redis message broker
def get_message_broker() -> Optional[RedisMessageBroker]:
    """Get or initialize Redis message broker for scraping."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    if not app_config.redis.ENABLED:
        return None

    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.SCRAPE_QUEUE_NAME)
        _logger.info(f"Initialized Scraper RedisMessageBroker with stream: '{app_config.redis.SCRAPE_QUEUE_NAME}'")

        # Create the consumer group if it doesn't exist
        try:
            _message_broker.create_consumer_group("scraper-workers")
        except Exception as e:
            _logger.warning(f"Error creating consumer group (may already exist): {e}")

        return _message_broker
    except Exception as e:
         _logger.exception(f"Failed to initialize RedisMessageBroker for scraper: {e}")
         return None


# --- Main Scraping Function ---
async def scrape_urls(
    urls: Union[str, List[str]],
    browser_config: Optional[BrowserConfig] = None,
    run_config: Optional[CrawlerRunConfig] = None
) -> List[Tuple[str, Optional[str]]]:
    """
    Scrapes URLs using Crawl4AI, returning Markdown content.
    Does NOT use Redis; called directly by worker or API.
    """
    import playwright
    from playwright.sync_api import sync_playwright

    if not _CRAWL4AI_AVAILABLE:
        _logger.error("Cannot scrape URLs: Crawl4AI library is not available.")
        url_list = [urls] if isinstance(urls, str) else urls
        return [(url, None) for url in url_list]

    url_list = [urls] if isinstance(urls, str) else urls
    if not url_list:
        return []

    _logger.info(f"Starting direct Crawl4AI scraping for {len(url_list)} URL(s)...")
    effective_browser_config = browser_config or BrowserConfig(headless=True)
    results_data: List[Tuple[str, Optional[str]]] = []

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with AsyncWebCrawler(config=effective_browser_config) as crawler:
                tasks = [crawler.arun(url, config=run_config) for url in url_list]
                crawl_results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(url_list, crawl_results):
                if isinstance(result, Exception):
                    error_msg = str(result)
                    if "Executable doesn't exist" in error_msg and "playwright" in error_msg.lower():
                        _logger.error(f"Playwright browser not installed for {url}. Run 'playwright install'. Error: {result}")
                    else:
                        _logger.error(f"Error crawling {url}: {result}")
                    results_data.append((url, None))
                elif hasattr(result, 'markdown'):
                    results_data.append((url, result.markdown))
                else:
                    _logger.warning(f"Unexpected result type from crawl4ai for {url}: {type(result)}")
                    results_data.append((url, None))

            break

        except playwright._impl._errors.TargetClosedError as e:
            _logger.warning(f"Attempt {attempt + 1}/{max_retries} failed due to TargetClosedError: {e}")
            if attempt + 1 == max_retries:
                _logger.error("Max retries reached. Failing the scraping process.")
                return [(url, None) for url in url_list]
            await asyncio.sleep(2)

        except Exception as e:
            error_msg = str(e)
            if "Host system is missing dependencies" in error_msg:
                _logger.critical("Playwright browser dependencies are missing. Please run 'playwright install-deps' to install them.")
                return [(url, None) for url in url_list]
            elif "Executable doesn't exist" in error_msg and "playwright" in error_msg.lower():
                _logger.critical("Playwright browser executable not found. Please run 'playwright install' to install the browsers.")
                return [(url, None) for url in url_list]
            else:
                _logger.exception(f"Unexpected error during Crawl4AI execution: {e}")
                return [(url, None) for url in url_list]

        finally:
            _logger.debug("Ensuring browser cleanup after attempt.")
            await asyncio.sleep(1)

    _logger.info(f"Direct Crawl4AI scraping finished for {len(url_list)} URL(s).")
    return results_data

async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    wait_for_result: bool = False,
    timeout: int = 300
) -> Optional[str]:
    """Queues a scraping request via Redis Streams."""
    message_broker = get_message_broker()
    if not message_broker:
        _logger.warning("Redis message broker for scraper not available, cannot queue scrape request.")
        return None

    job_id = str(uuid.uuid4())
    request_payload = {
        'job_id': job_id,
        'urls': [urls] if isinstance(urls, str) else urls,
        'browser_config': browser_config_dict,
        'run_config': run_config_dict,
        'request_time': time.time(),
        'status': 'pending',
        'message_type': 'scrape_request',
        'job_type': 'scrape'
    }

    _logger.debug(f"Scraper: Prepared request payload for job {job_id}")

    try:
        # Ensure stream exists
        message_broker.ensure_stream_exists("scraper_jobs")

        # Publish message to stream
        message_id = message_broker.publish_message(request_payload)
        success = message_id is not None
    except Exception as e:
        _logger.error(f"Failed to publish to Redis Stream: {e}")
        success = False

    if not success:
        _logger.error("Failed to add scrape request to Redis Stream.")
        return None

    _logger.info(f"Scrape request {job_id} queued successfully in Redis Stream.")

    if wait_for_result:
        _logger.warning(f"Waiting for scrape result {job_id} (timeout: {timeout}s) - Blocking operation.")
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            result = message_broker.get_job_status(job_id)
            _logger.debug(f"Scraper: Wait result for {job_id}: {result}")

            if result and result.get("status") in ["completed", "failed"]:
                _logger.info(f"Received result for scrape request {job_id}.")
                return job_id if result.get("status") == "completed" else None

            # Wait before checking again
            await asyncio.sleep(0.5)

        _logger.warning(f"Timed out waiting for scrape result {job_id}.")
        return None

    return job_id


async def get_scrape_result(request_id: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """Gets the result of a previously queued scraping request."""
    message_broker = get_message_broker()
    if not message_broker:
        _logger.error("Redis message broker for scraper not available, cannot get scrape result.")
        return None

    result = message_broker.get_job_status(request_id)
    _logger.debug(f"Scraper: get_scrape_result for {request_id}. Data from Redis Stream: {result}")

    if not result:
        _logger.warning(f"Scrape job {request_id} not found in the stream.")
        return None

    status = result.get("status")
    _logger.debug(f"Scraper: Job {request_id} status: {status}")

    if status == "completed":
        try:
            # Try to extract the data field which might be JSON-encoded
            scrape_output = result.get("data")
            if isinstance(scrape_output, str) and (scrape_output.startswith('[') or scrape_output.startswith('{')):
                scrape_output = json.loads(scrape_output)

            if isinstance(scrape_output, list):
                return scrape_output
            else:
                _logger.error(f"Unexpected data format in completed scrape job {request_id}: {type(scrape_output)}")
                return None
        except Exception as e:
            _logger.error(f"Error parsing scrape result for job {request_id}: {e}")
            return None
    elif status == "failed":
        _logger.error(f"Scrape job {request_id} failed: {result.get('error')}")
        return None
    else:
        _logger.debug(f"Scrape job {request_id} status: {status}. Result not ready.")
        return None


# --- Dedicated Scrape Worker (If using separate worker) ---
async def process_scrape_queue_worker(
    batch_size: int = 5, sleep_interval: int = 1, max_concurrent: int = 3
) -> None:
    """Worker function to process queued scraping requests from Redis Streams."""
    message_broker = get_message_broker()
    if not message_broker:
        _logger.error("Redis message broker not available. Scrape worker cannot start.")
        return
    if not _CRAWL4AI_AVAILABLE:
        _logger.error("Crawl4AI library not available. Scrape worker cannot start.")
        return

    # Create consumer group
    group_name = "scraper-workers"
    consumer_name = f"scraper-{uuid.uuid4().hex[:8]}"
    try:
        message_broker.create_consumer_group(group_name)
        _logger.info(f"Scraper worker initialized with consumer group: {group_name}, consumer: {consumer_name}")
    except Exception as e:
        _logger.warning(f"Error creating consumer group (may already exist): {e}")

    active_tasks = set()

    while True:
        try:
            # Check for space before consuming more messages
            available_slots = max_concurrent - len(active_tasks)

            if (available_slots > 0):
                # Consume messages from the stream
                messages = message_broker.consume_messages(
                    group_name=group_name,
                    consumer_name=consumer_name,
                    count=min(available_slots, batch_size),
                    block=2000  # Block for 2 seconds max
                )

                if messages:
                    _logger.info(f"Scraper worker received {len(messages[0][1])} new messages")

                    # Process each message
                    for stream_name, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            # Convert message data from possible bytes to dict
                            job_data = {}
                            for k, v in message_data.items():
                                key = k.decode() if isinstance(k, bytes) else k
                                value = v.decode() if isinstance(v, bytes) else v
                                job_data[key] = value

                            # Create a task to process this job
                            task = asyncio.create_task(
                                _process_single_scrape_job(
                                    job_id=job_data.get('job_id'),
                                    message_id=message_id,
                                    job_data=job_data,
                                    message_broker=message_broker,
                                    group_name=group_name
                                )
                            )
                            active_tasks.add(task)
                            task.add_done_callback(active_tasks.discard)

            # If we have active tasks or hit our concurrency limit, wait for some to complete
            if active_tasks:
                # Wait for at least one task to complete before fetching more
                done, pending = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=sleep_interval
                )
                active_tasks = pending
            else:
                # No active tasks and no new messages, sleep before polling again
                await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            _logger.info("Scrape worker main loop cancelled.")
            break
        except Exception as e:
            _logger.exception(f"Error in scrape worker main loop: {e}")
            await asyncio.sleep(sleep_interval * 2)

    # Clean up before exiting
    _logger.info("Scrape worker loop finished. Waiting for remaining active tasks...")
    if active_tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=15.0)
            _logger.info("Remaining scrape tasks finished or timed out.")
        except asyncio.TimeoutError:
            _logger.warning("Timeout waiting for remaining scrape tasks during shutdown.")
        except Exception as e:
            _logger.exception(f"Error waiting for remaining scrape tasks: {e}")
    _logger.info("Scrape worker task cleanup complete.")


async def _process_single_scrape_job(job_id: str, message_id: str, job_data: Dict[str, Any],
                                    message_broker: RedisMessageBroker, group_name: str):
    """Helper coroutine to process one scrape job from the Redis Stream."""
    if not job_id:
        job_id = str(uuid.uuid4())
        _logger.warning(f"Received job without ID, assigned new ID: {job_id}")

    _logger.debug(f"Scraper Worker: Processing job {job_id}. Message ID: {message_id}")

    # Extract URLs
    if isinstance(job_data.get('urls'), list):
        urls = job_data.get('urls')
    elif isinstance(job_data.get('urls'), str) and job_data.get('urls').startswith('['):
        # Handle JSON-encoded list
        try:
            urls = json.loads(job_data.get('urls'))
        except:
            urls = [job_data.get('urls')]
    elif 'urls' in job_data:
        urls = [job_data.get('urls')]
    else:
        urls = None

    if not urls:
        _logger.warning(f"Job {job_id} (Message {message_id}): No URLs found in job data. Skipping.")
        message_broker.acknowledge_message(group_name, message_id)
        _logger.debug(f"Acknowledged message {message_id} for job {job_id} due to missing URLs.")
        return

    # Extract config
    browser_config_dict = job_data.get('browser_config')
    run_config_dict = job_data.get('run_config')

    # Convert string JSON to dict if needed
    if isinstance(browser_config_dict, str) and browser_config_dict.startswith('{'):
        try:
            browser_config_dict = json.loads(browser_config_dict)
        except:
            browser_config_dict = None

    if isinstance(run_config_dict, str) and run_config_dict.startswith('{'):
        try:
            run_config_dict = json.loads(run_config_dict)
        except:
            run_config_dict = None

    # Update job status to processing
    message_broker.track_job_progress(job_id, "processing", {
        "message": f"Scraping {len(urls) if urls else 0} URLs..."
    })

    scrape_successful = False
    final_status = "failed"  # Default to failed
    error_message = "Unknown processing error"
    scrape_result_data = None

    try:
        # Convert dicts to Pydantic models if possible
        browser_config = BrowserConfig(**browser_config_dict) if browser_config_dict else None
        run_config = CrawlerRunConfig(**run_config_dict) if run_config_dict else None

        # Perform the scraping
        _logger.info(f"Job {job_id}: Starting scrape for {len(urls)} URLs...")
        scrape_result = await scrape_urls(urls, browser_config, run_config)

        # Check if any URL succeeded
        if scrape_result and any(content is not None for _, content in scrape_result):
            scrape_successful = True
            final_status = "completed"
            scrape_result_data = scrape_result
            error_message = None
            _logger.info(f"Scrape job {job_id} completed successfully.")
        else:
            error_message = "Scraping failed for all URLs or returned no content."
            _logger.error(f"Scrape job {job_id} failed: {error_message}")
            scrape_result_data = scrape_result

    except Exception as e:
        _logger.exception(f"Error processing scrape job {job_id}: {e}")
        error_message = str(e)
        final_status = "failed"
        scrape_successful = False

    finally:
        # Update job status with final result
        message_broker.track_job_progress(job_id, final_status, {
            "message": "Scraping completed successfully" if scrape_successful else error_message,
            "error": None if scrape_successful else error_message,
            "data": scrape_result_data
        })

        # Acknowledge the message to mark it as processed
        message_broker.acknowledge_message(group_name, message_id)
        _logger.debug(f"Acknowledged message {message_id} for job {job_id}")


# --- Cleanup Function ---
def close_redis_connections():
    """Closes Redis connections."""
    global _message_broker
    _message_broker = None
    _logger.info("Redis connections for scraper closed.")


# Export functions to make them properly available
__all__ = ['scrape_urls', 'queue_scrape_request', 'get_scrape_result', 'get_message_broker',
           'process_scrape_queue_worker', 'close_redis_connections']
