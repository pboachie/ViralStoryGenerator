# viralStoryGenerator/src/scrape_worker.py
"""
Scraper Worker for processing scraping requests via Redis Queue.
This module consumes requests queued by the scraping system.
"""
import asyncio
import signal
import sys
import time

from viralStoryGenerator.utils.crawl4ai_scraper import process_scrape_queue_worker, redis_manager as scraper_redis_manager
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    _logger.info(f"Received signal {sig}, initiating shutdown for Scraper worker...")
    shutdown_event.set()

async def run_scrape_worker_main():
    """Runs the main scrape worker loop and handles shutdown."""
    _logger.info("Starting Scraper Worker process...")

    # Explicitly log the queue name being used by the scraper's redis_manager
    if scraper_redis_manager:
        _logger.info(f"Scraper Worker: Configured to listen on queue: '{scraper_redis_manager.queue_name}' with result prefix: '{scraper_redis_manager.result_prefix}'")
    else:
        _logger.error("Scraper Worker: Redis manager is not initialized!")
        return # Exit if manager isn't ready

    active_tasks = set()
    while not shutdown_event.is_set():
        try:
            if not scraper_redis_manager.is_available():
                _logger.error("Scraper worker lost Redis connection. Sleeping...")
                await asyncio.sleep(10)
                continue

            # Process the scrape queue
            task = asyncio.create_task(process_scrape_queue_worker())
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

            # Wait briefly to allow task completion checks
            await asyncio.sleep(0.1)

        except Exception as e:
            _logger.exception(f"Error in scrape worker main loop: {e}")
            await asyncio.sleep(5)

    _logger.info("Shutdown signal received. Waiting for active tasks to complete...")
    if active_tasks:
        await asyncio.wait(active_tasks)
    _logger.info("All active tasks finished. Scraper Worker exiting.")

def main():
    """Entry point for the Scraper worker process."""
    _logger.info("Initializing Scraper Queue Worker...")
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if sys.platform == 'win32':
        # Required for signal handling on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Check Redis configuration
    if not app_config.redis.ENABLED:
        _logger.error("Redis is disabled in configuration (REDIS_ENABLED=False). Scraper Worker cannot run.")
        sys.exit(1)
    # Use the imported manager for the check
    if not scraper_redis_manager or not scraper_redis_manager.is_available():
        _logger.error("Scraper Redis manager failed to initialize or connect. Scraper Worker cannot run.")
        sys.exit(1)

    try:
        asyncio.run(run_scrape_worker_main())
    except KeyboardInterrupt:
        _logger.info("Scraper worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"Scraper worker failed with unhandled exception: {e}")
        sys.exit(1)

    _logger.info("Scraper Queue Worker shutdown complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()