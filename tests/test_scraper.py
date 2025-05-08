"""
Simple test script to verify the scraper functionality
"""
import asyncio
import sys
import time
from viralStoryGenerator.utils.crawl4ai_scraper import queue_scrape_request, get_scrape_result, get_redis_manager
from viralStoryGenerator.src.logger import logger as _logger

async def test_scraper():
    """Test the scraper functionality directly"""
    # Get the Redis manager
    scraper_redis_manager = get_redis_manager()
    if not scraper_redis_manager:
        _logger.error("Failed to initialize scraper Redis manager")
        return False

    # Test URL
    test_url = "https://example.com"
    _logger.info(f"Queuing scrape request for {test_url}...")

    # Queue the request
    scrape_request_id = await queue_scrape_request(test_url)
    if not scrape_request_id:
        _logger.error("Failed to queue scrape request")
        return False

    _logger.info(f"Scrape request queued with ID: {scrape_request_id}")

    # Poll for result
    max_retries = 10
    retry_interval = 2
    for attempt in range(max_retries):
        _logger.info(f"Attempt {attempt + 1}/{max_retries}: Checking for scrape result...")

        # Get status directly from Redis
        scrape_status_data = scraper_redis_manager.get_result(scrape_request_id)
        if scrape_status_data:
            status = scrape_status_data.get("status")
            _logger.info(f"Status: {status}")

            if status == "failed":
                _logger.error(f"Scrape job failed: {scrape_status_data.get('error')}")
                return False

        # Try to get the actual result
        scrape_result = await get_scrape_result(scrape_request_id)
        if scrape_result:
            _logger.info(f"Scrape result: {scrape_result}")
            return True

        _logger.info(f"No result yet. Waiting {retry_interval}s...")
        await asyncio.sleep(retry_interval)

    _logger.error("Failed to get scrape result after maximum retries")
    return False

if __name__ == "__main__":
    result = asyncio.run(test_scraper())
    sys.exit(0 if result else 1)
