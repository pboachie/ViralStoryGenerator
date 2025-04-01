#/viralStoryGenerator/utils/crawl4ai_scraper.py
import asyncio
import os
from typing import List, Union, Optional, Tuple, Dict, Any
import time
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from .redis_manager import RedisManager as RedisQueueManager
from viralStoryGenerator.src.logger import logger as _logger

# Initialize Redis queue manager with environment variables if available
def get_redis_manager() -> RedisQueueManager:
    """
    Get or create a Redis queue manager instance using environment variables.

    Returns:
        RedisQueueManager: An instance of the Redis queue manager
    """
    _logger.debug("Initializing Redis queue manager...")
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = int(os.environ.get('REDIS_PORT', 6379))
    db = int(os.environ.get('REDIS_DB', 0))
    password = os.environ.get('REDIS_PASSWORD', None)

    try:
        return RedisQueueManager(host=host, port=port, db=db, password=password)
    except Exception as e:
        _logger.error(f"Failed to initialize Redis queue manager: {str(e)}")
        _logger.warning("Continuing without Redis queue management")
        return None
    finally:
        _logger.debug("Redis queue manager initialized.")

# Queue-based version that uses Redis if available
async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    wait_for_result: bool = False,
    timeout: int = 300
) -> Union[str, None]:
    """
    Queue a scraping request using the Redis queue manager.

    Args:
        urls: A single URL or a list of URLs to scrape
        browser_config: Configuration for the browser
        run_config: Configuration for the crawler run
        wait_for_result: Whether to wait for the result or return immediately
        timeout: Maximum time to wait for a result in seconds

    Returns:
        request_id: The ID of the queued request, or None if queuing failed
    """
    manager = get_redis_manager()
    if not manager:
        _logger.warning("Redis queue manager not available, falling back to direct scraping")
        return None

    # Generate a unique ID for this request
    request_id = str(time.time()) + "_" + str(hash(str(urls)))

    # Prepare request data
    request_data = {
        'id': request_id,
        'data': {
            'urls': urls if isinstance(urls, list) else [urls],
            'browser_config': browser_config,
            'run_config': run_config
        }
    }

    try:
        # Add request to queue
        success = manager.add_request(request_data)

        if not success:
            _logger.warning("Failed to add request to queue")
            return None

        # Wait for result if specified
        if wait_for_result:
            _logger.info(f"Waiting for result of request {request_id} (timeout: {timeout}s)")

            result = manager.wait_for_result(request_id, timeout=timeout)
            if result and result.get("status") == "completed":
                _logger.info(f"Got result for request {request_id}")
                return request_id
            else:
                _logger.warning(f"Timed out or error waiting for result of request {request_id}")
                if result:
                    _logger.debug(f"Result status: {result.get('status')}, error: {result.get('error')}")

        return request_id
    except Exception as e:
        _logger.error(f"Failed to queue scraping request: {str(e)}")
        return None

async def get_scrape_result(request_id: str) -> Union[List[Tuple[str, Optional[str]]], None]:
    """
    Get the result of a previously queued scraping request.

    Args:
        request_id: The ID of the queued request

    Returns:
        result: A list of (url, markdown) tuples, or None if not available
    """
    manager = get_redis_manager()
    if not manager:
        _logger.error("Redis queue manager not available, cannot retrieve result")
        return None

    try:
        result = manager.get_result(request_id)
        if not result:
            _logger.debug(f"No result found for request ID: {request_id}")
            return None

        # Log what we received to help debug
        _logger.debug(f"Received result for {request_id}: {str(result)[:100]}...")

        # Check different possible data structures
        if 'data' in result:
            return result['data']
        elif 'status' in result and result.get('status') != 'completed':
            _logger.debug(f"Request {request_id} not completed: {result.get('status')}")
            return None
        elif isinstance(result, list):
            # Result might be directly a list of tuples
            return result

        _logger.warning(f"Unexpected result format for request {request_id}")
        return None
    except Exception as e:
        _logger.error(f"Failed to get scrape result: {str(e)}")
        return None

async def scrape_urls(
    urls: Union[str, List[str]],
    browser_config: Optional[BrowserConfig] = None,
    run_config: Optional[CrawlerRunConfig] = None,
    direct_only: bool = False
) -> List[Tuple[str, Optional[str]]]:
    """
    Scrape the given URLs using Crawl4AI and return the content as Markdown.
    If Redis queue manager is available, the request will be queued.

    Args:
        urls (Union[str, List[str]]): A single URL or a list of URLs to scrape.
        browser_config (Optional[BrowserConfig]): Configuration for the browser, e.g., headless mode.
        run_config (Optional[CrawlerRunConfig]): Configuration for the crawl run, e.g., extraction strategies.
        direct_only (bool): If True, skip the queue and scrape directly (to avoid recursive queueing)

    Returns:
        List[Tuple[str, Optional[str]]]: A list of tuples, each containing the URL and its corresponding
        Markdown content. If a URL fails to crawl, its Markdown will be None.

    Notes:
        - This function is asynchronous and must be run within an asyncio event loop, e.g., using `asyncio.run()`.
        - Errors during crawling are logged using the `logging` module and do not halt the process.
        - If Redis queue manager is available, the request will be queued to prevent system overload.
        - The function will attempt to use the queue first, then fall back to direct scraping if needed.
    """
    _logger.debug(f"Scraping URLs: {urls}")

    # If direct_only flag is set, skip the queue entirely
    if not direct_only:
        # Check if there's a queue worker running by checking the processing queue
        manager = get_redis_manager()
        queue_active = False
        if manager and manager.is_available():
            try:
                # See if there's a processing queue with items, indicating a worker is running
                processing_queue = f"{manager.queue_name}_processing"
                processing_count = manager.client.llen(processing_queue)
                if processing_count > 0:
                    queue_active = True
                    _logger.debug(f"Queue worker appears active (processing {processing_count} items)")
            except Exception:
                pass

        # Try to use the Redis queue first
        queued_request_id = await queue_scrape_request(urls, browser_config, run_config, wait_for_result=queue_active)
        if queued_request_id:
            # Give the worker a moment to pick up the job if it's active
            if queue_active:
                # Check for result a few times with short delays
                for attempt in range(3):
                    result = await get_scrape_result(queued_request_id)
                    if result:
                        _logger.info(f"Got result from queue for request {queued_request_id}")
                        return result
                    # Brief pause between checks
                    await asyncio.sleep(0.5)

            _logger.info("Falling back to direct scraping after queue attempt")

    # Convert single URL string to a list for uniform processing
    if isinstance(urls, str):
        urls = [urls]

    try:
        # Initialize the AsyncWebCrawler with optional browser configuration
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Create a list of crawl tasks for each URL with optional run configuration
            tasks = [crawler.arun(url, config=run_config) for url in urls]
            # Execute all tasks concurrently, capturing exceptions without stopping
            results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        error_msg = str(e)

        # Check for Playwright installation errors
        if "Executable doesn't exist" in error_msg and "playwright" in error_msg.lower():
            _logger.error("Playwright browser executable not found. Please install browsers by running: playwright install")

            # Return failure for all URLs
            return [(url, None) for url in urls]
        else:
            # Re-raise other exceptions
            raise

    # Process results into a list of (url, markdown) tuples
    output = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            # Log errors and append None for failed crawls
            _logger.error(f"Error crawling {url}: {result}")
            output.append((url, None))
        else:
            # Append successful crawl results as Markdown
            output.append((url, result.markdown))

    # Store the result in the queue for potential future requests
    if not direct_only and 'queued_request_id' in locals() and queued_request_id:
        manager = get_redis_manager()
        if manager and manager.is_available():
            try:
                manager.store_result(queued_request_id, {
                    "status": "completed",
                    "data": output
                })
                _logger.debug(f"Stored direct scraping result in queue for request {queued_request_id}")
            except Exception as e:
                _logger.error(f"Failed to store direct scraping result: {str(e)}")

    _logger.debug(f"Scraping completed for URLs: {urls}")
    return output

# Worker function to process queued requests in a background process
async def process_queue_worker(
    batch_size: int = 5,
    sleep_interval: int = 1,
    max_concurrent: int = 3
) -> None:
    """
    A worker function to process queued scraping requests.

    Args:
        batch_size: Maximum number of requests to process in one batch
        sleep_interval: Time to sleep between polling the queue in seconds
        max_concurrent: Maximum number of concurrent scraping tasks

    Notes:
        This function is meant to be run in a background process or task.
        It will continuously poll the queue and process requests.
    """
    manager = get_redis_manager()
    if not manager:
        _logger.error("Redis queue manager not available, cannot start worker")
        return

    _logger.info(f"Starting queue worker process with batch_size={batch_size}, max_concurrent={max_concurrent}")

    while True:
        try:
            # Check queue length
            queue_length = manager.get_queue_length()
            if queue_length == 0:
                # If queue is empty, sleep and check again
                await asyncio.sleep(sleep_interval)
                continue

            # Process up to batch_size requests
            batch = []
            for _ in range(min(batch_size, queue_length)):
                request = manager.get_next_request()
                if request:
                    batch.append(request)

            if not batch:
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"Processing batch of {len(batch)} requests")

            # Process requests in batches with max_concurrent limit
            for i in range(0, len(batch), max_concurrent):
                sub_batch = batch[i:i + max_concurrent]
                tasks = []

                for request in sub_batch:
                    # Validate request structure
                    if not isinstance(request, dict):
                        _logger.error(f"Invalid request format: {request}")
                        if manager and hasattr(manager, 'complete_request'):
                            manager.complete_request(request, False)
                        continue

                    # Extract request ID and data, with validation
                    request_id = request.get('id')
                    if not request_id:
                        _logger.error("Request is missing ID")
                        if manager and hasattr(manager, 'complete_request'):
                            manager.complete_request(request, False)
                        continue

                    request_data = request.get('data')
                    if not request_data:
                        _logger.error(f"Request {request_id} is missing data")
                        # Store error result and mark request as completed with failure
                        manager.store_result(request_id, {
                            "status": "failed",
                            "error": "Request is missing required data field",
                            "created_at": time.time(),
                            "updated_at": time.time()
                        })
                        if manager and hasattr(manager, 'complete_request'):
                            manager.complete_request(request, False)
                        continue

                    # Wrapped in a function to properly catch and manage exceptions for each task
                    async def process_request(req_id, req_data, req_obj):
                        success = False
                        try:
                            # Validate required fields in req_data
                            if not isinstance(req_data, dict):
                                raise ValueError(f"Request data is not a dictionary: {type(req_data)}")

                            urls = req_data.get('urls')
                            if not urls:
                                raise ValueError("No URLs specified in request data")

                            browser_config = req_data.get('browser_config')
                            run_config = req_data.get('run_config')

                            # Log what we're about to process
                            _logger.debug(f"Processing request {req_id} with URLs: {urls}")

                            # Perform the actual scraping
                            result = await scrape_urls(urls, browser_config, run_config, direct_only=True)

                            # Store result in Redis
                            manager.store_result(req_id, {
                                "status": "completed",
                                "data": result,
                                "updated_at": time.time()
                            })
                            _logger.info(f"Successfully processed request {req_id}")
                            success = True
                        except Exception as e:
                            _logger.error(f"Error processing request {req_id}: {str(e)}")
                            # Store error as result
                            manager.store_result(req_id, {
                                "status": "failed",
                                "error": str(e),
                                "updated_at": time.time()
                            })
                        finally:
                            # Remove from processing queue regardless of outcome
                            # Mark as success (true) if successful, failure (false) if failed
                            try:
                                if manager and hasattr(manager, 'complete_request'):
                                    manager.complete_request(req_obj, success)
                            except Exception as cleanup_error:
                                _logger.error(f"Error completing request {req_id}: {str(cleanup_error)}")

                    # Add the task to our batch
                    tasks.append(process_request(request_id, request_data, request))

                # Wait for all tasks in sub-batch to complete
                if tasks:
                    await asyncio.gather(*tasks)

            # Sleep briefly to avoid spinning too fast
            await asyncio.sleep(0.1)

        except Exception as e:
            _logger.error(f"Error in queue worker: {str(e)}")
            import traceback
            _logger.error(f"Traceback: {traceback.format_exc()}")
            await asyncio.sleep(sleep_interval)
