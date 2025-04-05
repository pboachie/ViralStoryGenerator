# viralStoryGenerator/src/scrape_worker.py
"""
Scraper Worker for processing scraping requests via Redis Queue.
This module consumes requests queued by the scraping system.
"""
import asyncio
import signal
import sys
import time
import json

from viralStoryGenerator.utils.crawl4ai_scraper import process_scrape_queue_worker, redis_manager as scraper_redis_manager, close_scraper_redis_connections # Import cleanup
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.vector_db_manager import close_client as close_vector_db

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
        return

    if not scraper_redis_manager.is_available():
        _logger.error("Scraper Worker: Redis is not available. Check connection settings.")
        return

    await clear_stalled_processing_jobs()

    worker_task = None
    try:
        # Worker configuration from config
        batch_size = app_config.redis.WORKER_BATCH_SIZE
        sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
        max_concurrent = app_config.redis.WORKER_MAX_CONCURRENT

        _logger.info(f"Scraper Worker Config: BatchSize={batch_size}, SleepInterval={sleep_interval}s, MaxConcurrent={max_concurrent}")

        # Start the worker task
        worker_task = asyncio.create_task(
            process_scrape_queue_worker(
                batch_size=batch_size,
                sleep_interval=sleep_interval,
                max_concurrent=max_concurrent
            )
        )

        # Wait for shutdown signal
        await shutdown_event.wait()
        _logger.info("Shutdown signal received by scrape worker main loop.")
    except Exception as e:
        _logger.exception(f"Error in scrape worker main loop: {e}")
        worker_task = None
    finally:
        # --- Shutdown Sequence ---
        if worker_task and not worker_task.done():
            _logger.info("Cancelling main scrape worker task...")
            worker_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(worker_task, return_exceptions=True), timeout=15.0)
            except asyncio.TimeoutError:
                 _logger.warning("Timeout waiting for scrape worker task cancellation.")
            except asyncio.CancelledError:
                _logger.info("Scrape worker task cancelled successfully.")
            except Exception as e:
                _logger.exception(f"Exception during scrape worker task cancellation/cleanup: {e}")

        # Close resources
        _logger.info("Closing Vector DB client (from scrape worker)...")
        close_vector_db()
        _logger.info("Closing Scraper Redis connection pool...")
        close_scraper_redis_connections()

        _logger.info("Scraper Worker exiting.")

async def clear_stalled_processing_jobs(max_age_seconds: int = 300):
    """Cleans up any jobs stuck in the processing queue from previous runs."""
    if not scraper_redis_manager or not scraper_redis_manager.is_available():
        return

    _logger.info("Checking for stalled jobs in processing queue...")
    processing_queue_name = f"{scraper_redis_manager.queue_name}_processing"

    try:
        # Get all items from processing queue
        stalled_items = scraper_redis_manager.client.lrange(processing_queue_name, 0, -1)
        stalled_count = 0

        for item_str in stalled_items:
            try:
                item = json.loads(item_str)
                job_id = item.get('id') or item.get('job_id')

                if job_id:
                    # Mark job as failed
                    scraper_redis_manager.store_result(job_id, {
                        "status": "failed",
                        "error": "Job was found stalled in processing queue",
                        "updated_at": time.time()
                    })

                    # Remove from processing queue
                    scraper_redis_manager.client.lrem(processing_queue_name, 0, item_str)
                    stalled_count += 1
                    _logger.warning(f"Marked stalled job {job_id} as failed and removed from queue")
            except (json.JSONDecodeError, TypeError):
                # For badly formatted items, just try to remove them
                scraper_redis_manager.client.lrem(processing_queue_name, 0, item_str)

        if stalled_count > 0:
            _logger.info(f"Cleared {stalled_count} stalled jobs from processing queue")
    except Exception as e:
        _logger.error(f"Error clearing stalled jobs: {e}")

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

    exit_code = 0
    try:
        asyncio.run(run_scrape_worker_main())
    except KeyboardInterrupt:
        _logger.info("Scraper worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"Scraper worker failed with unhandled exception: {e}")
        sys.exit(1)

    _logger.info("Scraper Queue Worker shutdown complete.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()