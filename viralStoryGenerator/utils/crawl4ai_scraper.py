#/viralStoryGenerator/utils/crawl4ai_scraper.py
import asyncio
import logging
import os
from typing import List, Union, Optional, Tuple, Dict, Any
import time
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from .redis_manager import RedisQueueManager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Redis queue manager with environment variables if available
def get_redis_manager() -> RedisQueueManager:
    """
    Get or create a Redis queue manager instance using environment variables.

    Returns:
        RedisQueueManager: An instance of the Redis queue manager
    """
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = int(os.environ.get('REDIS_PORT', 6379))
    db = int(os.environ.get('REDIS_DB', 0))
    password = os.environ.get('REDIS_PASSWORD', None)

    try:
        return RedisQueueManager(host=host, port=port, db=db, password=password)
    except Exception as e:
        logger.error(f"Failed to initialize Redis queue manager: {str(e)}")
        logger.warning("Continuing without Redis queue management")
        return None

# Queue-based version that uses Redis if available
async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    wait_for_result: bool = True,
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
        logger.warning("Redis queue manager not available, falling back to direct scraping")
        return None

    # Prepare request data
    request_data = {
        'urls': urls if isinstance(urls, list) else [urls],
        'browser_config': browser_config,
        'run_config': run_config
    }

    try:
        # Add request to queue
        request_id = manager.add_request(request_data)

        # Wait for result if specified
        if wait_for_result:
            result = manager.wait_for_result(request_id, timeout=timeout)
            if result:
                return result
            else:
                logger.warning(f"Timed out waiting for result of request {request_id}")
                return request_id

        return request_id
    except Exception as e:
        logger.error(f"Failed to queue scraping request: {str(e)}")
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
        logger.error("Redis queue manager not available, cannot retrieve result")
        return None

    try:
        result = manager.get_result(request_id)
        if result and 'data' in result:
            return result['data']
        return None
    except Exception as e:
        logger.error(f"Failed to get scrape result: {str(e)}")
        return None

async def scrape_urls(
    urls: Union[str, List[str]],
    browser_config: Optional[BrowserConfig] = None,
    run_config: Optional[CrawlerRunConfig] = None
) -> List[Tuple[str, Optional[str]]]:
    """
    Scrape the given URLs using Crawl4AI and return the content as Markdown.
    If Redis queue manager is available, the request will be queued.

    Args:
        urls (Union[str, List[str]]): A single URL or a list of URLs to scrape.
        browser_config (Optional[BrowserConfig]): Configuration for the browser, e.g., headless mode.
        run_config (Optional[CrawlerRunConfig]): Configuration for the crawl run, e.g., extraction strategies.

    Returns:
        List[Tuple[str, Optional[str]]]: A list of tuples, each containing the URL and its corresponding
        Markdown content. If a URL fails to crawl, its Markdown will be None.

    Notes:
        - This function is asynchronous and must be run within an asyncio event loop, e.g., using `asyncio.run()`.
        - Errors during crawling are logged using the `logging` module and do not halt the process.
        - If Redis queue manager is available, the request will be queued to prevent system overload.
        - The function will attempt to use the queue first, then fall back to direct scraping if needed.

    Example:
        ```python
        import asyncio
        from crawl4ai_scraper import scrape_urls

        async def main():
            urls = ["https://example.com", "https://another.com"]
            results = await scrape_urls(urls)
            for url, markdown in results:
                if markdown:
                    print(f"Markdown for {url}:\n{markdown[:100]}...")
                else:
                    print(f"Failed to crawl {url}")

        if __name__ == "__main__":
            asyncio.run(main())
        ```
    """
    # Try to use the Redis queue first
    queued_request_id = await queue_scrape_request(urls, browser_config, run_config)
    if queued_request_id:
        result = await get_scrape_result(queued_request_id)
        if result:
            return result
        logger.info("Falling back to direct scraping after queue attempt")

    # Convert single URL string to a list for uniform processing
    if isinstance(urls, str):
        urls = [urls]

    # Initialize the AsyncWebCrawler with optional browser configuration
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Create a list of crawl tasks for each URL with optional run configuration
        tasks = [crawler.arun(url, config=run_config) for url in urls]
        # Execute all tasks concurrently, capturing exceptions without stopping
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results into a list of (url, markdown) tuples
    output = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            # Log errors and append None for failed crawls
            logging.error(f"Error crawling {url}: {result}")
            output.append((url, None))
        else:
            # Append successful crawl results as Markdown
            output.append((url, result.markdown))

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
        logger.error("Redis queue manager not available, cannot start worker")
        return

    logger.info(f"Starting queue worker process with batch_size={batch_size}, max_concurrent={max_concurrent}")

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

            logger.info(f"Processing batch of {len(batch)} requests")

            # Process requests in batches with max_concurrent limit
            for i in range(0, len(batch), max_concurrent):
                sub_batch = batch[i:i + max_concurrent]
                tasks = []

                for request in sub_batch:
                    request_id = request['id']
                    request_data = request['data']

                    # Create task to scrape URLs and store result
                    async def process_request(req_id, req_data):
                        try:
                            urls = req_data.get('urls', [])
                            browser_config = req_data.get('browser_config')
                            run_config = req_data.get('run_config')

                            # Perform the actual scraping
                            result = await scrape_urls(urls, browser_config, run_config)

                            # Store result in Redis
                            manager.store_result(req_id, result)
                            logger.info(f"Successfully processed request {req_id}")
                        except Exception as e:
                            logger.error(f"Error processing request {req_id}: {str(e)}")
                            # Store error as result
                            manager.store_result(req_id, {'error': str(e)})

                    tasks.append(process_request(request_id, request_data))

                # Wait for all tasks in sub-batch to complete
                await asyncio.gather(*tasks)

            # Sleep briefly to avoid spinning too fast
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in queue worker: {str(e)}")
            await asyncio.sleep(sleep_interval)
