# viralStoryGenerator/utils/crawl4ai_scraper.py
"""Web scraping utilities using Crawl4AI with optional Redis queuing."""
import asyncio
import os
from typing import List, Union, Optional, Tuple, Dict, Any
import time
import json
import uuid

import playwright
from playwright.sync_api import sync_playwright

# Use Crawl4AI library
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _logger.error("Crawl4AI library not found. pip install crawl4ai")
    _CRAWL4AI_AVAILABLE = False
    class AsyncWebCrawler: pass
    class BrowserConfig: pass
    class CrawlerRunConfig: pass


# TODO: Use shared RedisManager instance or create one specifically? Shared seems okay.
from .redis_manager import RedisManager as RedisQueueManager
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger

# Initialize Redis queue manager only if Redis is enabled
redis_manager: Optional[RedisQueueManager] = None
if app_config.redis.ENABLED:
    try:
        # Use scrape-specific config values
        scrape_queue_name = getattr(app_config.redis, 'SCRAPE_QUEUE_NAME', 'viralstory_scrape_queue')
        scrape_result_prefix = getattr(app_config.redis, 'SCRAPE_RESULT_PREFIX', 'viralstory_scrape_results:')
        scrape_ttl = getattr(app_config.redis, 'SCRAPE_TTL', app_config.redis.TTL)

        _logger.info(f"Initializing Scraper RedisManager with Queue: '{scrape_queue_name}', Prefix: '{scrape_result_prefix}'") # DEBUG ADDED

        redis_manager = RedisQueueManager(
            queue_name=scrape_queue_name,
            result_prefix=scrape_result_prefix,
            ttl=scrape_ttl
        )
        if not redis_manager.is_available():
             _logger.warning("Scraper RedisManager initialized but Redis is unavailable.")
             redis_manager = None
    except Exception as e:
         _logger.exception(f"Failed to initialize RedisManager for scraper: {e}")
         redis_manager = None


# --- Main Scraping Function ---
async def scrape_urls(
    urls: Union[str, List[str]],
    browser_config: Optional[BrowserConfig] = None,
    run_config: Optional[CrawlerRunConfig] = None
) -> List[Tuple[str, Optional[str]]]:
    """
    Scrapes URLs using Crawl4AI, returning Markdown content.
    Does NOT use Redis queue; called directly by worker or API if Redis disabled.
    """
    if not _CRAWL4AI_AVAILABLE:
        _logger.error("Cannot scrape URLs: Crawl4AI library is not available.")
        url_list = [urls] if isinstance(urls, str) else urls
        return [(url, None) for url in url_list]

    url_list = [urls] if isinstance(urls, str) else urls
    if not url_list: return []

    _logger.info(f"Starting direct Crawl4AI scraping for {len(url_list)} URL(s)...")
    # Add default browser config if none provided (e.g., headless)
    effective_browser_config = browser_config or BrowserConfig(headless=True)
    results_data: List[Tuple[str, Optional[str]]] = []

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Initialize crawler within async context manager
            async with AsyncWebCrawler(config=effective_browser_config) as crawler:
                tasks = [crawler.arun(url, config=run_config) for url in url_list]
                crawl_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
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

            break  # Exit retry loop on success

        except playwright._impl._errors.TargetClosedError as e:
            _logger.warning(f"Attempt {attempt + 1}/{max_retries} failed due to TargetClosedError: {e}")
            if attempt + 1 == max_retries:
                _logger.error("Max retries reached. Failing the scraping process.")
                return [(url, None) for url in url_list]
            await asyncio.sleep(2)  # Wait before retrying

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


    _logger.info(f"Direct Crawl4AI scraping finished for {len(url_list)} URL(s).")
    return results_data

async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    wait_for_result: bool = False,
    timeout: int = 300
) -> Optional[str]:
    """Queues a scraping request via Redis if manager is available."""
    if not redis_manager:
        _logger.warning("Redis manager for scraper not available, cannot queue scrape request.")
        return None
    _logger.debug(f"Scraper: Using Redis Manager for queue '{redis_manager.queue_name}' to queue scrape request.") # DEBUG ADDED

    job_id = str(uuid.uuid4())
    request_payload = {
        'id': job_id,
        'data': {
            'urls': [urls] if isinstance(urls, str) else urls,
            'browser_config': browser_config_dict,
            'run_config': run_config_dict
        },
        'request_time': time.time()
    }
    _logger.debug(f"Scraper: Preparing request payload for job {job_id}: {request_payload}") # DEBUG ADDED

    success = redis_manager.add_request(request_payload)
    if not success:
        _logger.error("Failed to add scrape request to Redis queue.")
        return None

    _logger.info(f"Scrape request {job_id} queued successfully.")

    if wait_for_result:
         _logger.warning(f"Waiting for scrape result {job_id} (timeout: {timeout}s) - Blocking operation.")
         result = redis_manager.wait_for_result(job_id, timeout=timeout)
         _logger.debug(f"Scraper: Wait result for {job_id}: {result}") # DEBUG ADDED
         if result and result.get("status") == "completed":
             _logger.info(f"Received result for scrape request {job_id}.")
             return job_id
         else:
             _logger.warning(f"Timed out or error waiting for scrape result {job_id}. Status: {result.get('status') if result else 'N/A'}")
             return None

    return job_id


async def get_scrape_result(request_id: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """Gets the result of a previously queued scraping request."""
    if not redis_manager:
        _logger.error("Redis manager for scraper not available, cannot get scrape result.")
        return None

    result_data = redis_manager.get_result(request_id)
    _logger.debug(f"Scraper: get_scrape_result for {request_id}. Raw data from Redis: {result_data}") # DEBUG ADDED
    if not result_data:
        if redis_manager.check_key_exists(request_id):
            _logger.debug(f"Scrape job {request_id} is pending/processing.")
        else:
            _logger.warning(f"Scrape job {request_id} not found.")
        return None

    status = result_data.get("status")
    _logger.debug(f"Scraper: Job {request_id} status from Redis: {status}") # DEBUG ADDED
    if status == "completed":
        scrape_output = result_data.get("data")
        if isinstance(scrape_output, list):
             return scrape_output
        else:
             _logger.error(f"Unexpected data format in completed scrape job {request_id}: {type(scrape_output)}")
             return None
    elif status == "failed":
         _logger.error(f"Scrape job {request_id} failed: {result_data.get('error')}")
         return None
    else:
         _logger.debug(f"Scrape job {request_id} status: {status}. Result not ready.")
         return None


# --- Dedicated Scrape Worker (If using separate worker) ---
async def process_scrape_queue_worker(
    batch_size: int = 5, sleep_interval: int = 1, max_concurrent: int = 3
) -> None:
    """Worker function to process queued scraping requests (if Redis queuing is used)."""
    if not redis_manager:
        _logger.error("Scraper Redis manager not available. Scrape worker cannot start.")
        return
    if not _CRAWL4AI_AVAILABLE:
         _logger.error("Crawl4AI library not available. Scrape worker cannot start.")
         return

    active_tasks = set()
    while True:
        try:
            if not redis_manager.is_available():
                 _logger.error("Scraper worker lost Redis connection. Sleeping...")
                 await asyncio.sleep(10)
                 continue

            num_to_fetch = min(batch_size, max_concurrent - len(active_tasks))
            if num_to_fetch <= 0:
                if active_tasks:
                     done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                     active_tasks = pending
                else: await asyncio.sleep(0.1)
                continue

            batch = []
            for _ in range(num_to_fetch):
                request = redis_manager.get_next_request()
                if request:
                     if isinstance(request, dict) and 'id' in request and 'data' in request:
                         batch.append(request)
                     else:
                         _logger.error(f"Invalid item received from scrape queue: {str(request)[:100]}...")
                         if hasattr(redis_manager, 'complete_request') and isinstance(request, dict) and '_original_data' in request:
                             redis_manager.complete_request(request, success=False)
                else: break # Queue empty

            if not batch:
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"Scrape worker processing batch of {len(batch)} requests.")

            for request in batch:
                task = asyncio.create_task(_process_single_scrape_job(request, redis_manager))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

        except Exception as e:
             _logger.exception(f"Error in scrape worker main loop: {e}")
             await asyncio.sleep(sleep_interval * 2)


async def _process_single_scrape_job(request: Dict[str, Any], manager: RedisQueueManager):
    """Helper coroutine to process one scrape job."""
    job_id = request['id']
    _logger.debug(f"Scraper Worker: Starting processing for job {job_id}. Request: {request}") # DEBUG ADDED
    job_data = request.get('data', {})
    urls = job_data.get('urls')
    browser_config_dict = job_data.get('browser_config')
    run_config_dict = job_data.get('run_config')
    start_time = request.get('request_time', time.time())

    scrape_successful = False
    final_status = "failed" # Default to failed
    error_message = "Unknown processing error"
    scrape_result_data = None

    try:
        if not urls:
            raise ValueError("No URLs found in scrape job data.")

        # Convert dicts back to Pydantic models
        browser_config = BrowserConfig(**browser_config_dict) if browser_config_dict else None
        run_config = CrawlerRunConfig(**run_config_dict) if run_config_dict else None

        # Perform the actual scraping
        manager.store_result(job_id, {"status": "processing", "message": f"Scraping {len(urls)} URLs...", "updated_at": time.time()}, merge=True)
        _logger.info(f"Job {job_id}: Starting scrape for {len(urls)} URLs...")
        scrape_result = await scrape_urls(urls, browser_config, run_config)

        # Check if *any* URL succeeded
        if scrape_result and any(content is not None for _, content in scrape_result):
            scrape_successful = True
            final_status = "completed" # <-- Changed from "scraped"
            scrape_result_data = scrape_result
            error_message = None
            _logger.info(f"Scrape job {job_id} completed successfully.")
        else:
             error_message = "Scraping failed for all URLs after retries or returned no content."
             _logger.error(f"Scrape job {job_id} failed: {error_message}")
             scrape_result_data = scrape_result

    except Exception as e:
        _logger.exception(f"Error processing scrape job {job_id}: {e}")
        error_message = str(e)
        final_status = "failed"
        scrape_successful = False

    finally:
        # Store the final result/status
        result_payload = {
            "status": final_status,
            "error": error_message,
            "data": scrape_result_data,
            "created_at": start_time,
            "updated_at": time.time()
        }
        # Make sure error is None if successful
        if scrape_successful:
            result_payload.pop("error", None)

        _logger.debug(f"Scraper Worker: Storing final result for job {job_id}: {result_payload}") # DEBUG ADDED
        store_success = manager.store_result(job_id, result_payload)
        if store_success:
             _logger.debug(f"Updated job {job_id} status to '{final_status}' in Redis.")
        else:
             _logger.error(f"CRITICAL: Failed to store final status '{final_status}' for job {job_id} in Redis.")

        # Complete the request in the processing queue
        completion_success = manager.complete_request(request, success=scrape_successful)
        _logger.debug(f"Scraper Worker: Completion status for job {job_id} in processing queue: {completion_success}") # DEBUG ADDED
        if not completion_success:
             _logger.warning(f"Failed to properly complete/remove job {job_id} from processing queue.")