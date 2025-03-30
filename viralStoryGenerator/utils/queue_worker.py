"""
Queue Worker for processing crawl4ai requests from the Redis queue.
Run this script separately to start a background worker process.
"""
import asyncio
import logging
import signal
import sys
import os
from .config import config
from .crawl4ai_scraper import process_queue_worker

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crawl4ai_worker.log')
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down worker...")
    shutdown_event.set()

async def run_worker():
    """Run the queue worker with graceful shutdown handling."""
    # Get worker configuration from config
    batch_size = config.redis.WORKER_BATCH_SIZE
    sleep_interval = config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent = config.redis.WORKER_MAX_CONCURRENT

    logger.info(f"Starting Crawl4AI queue worker with batch_size={batch_size}, "
                f"max_concurrent={max_concurrent}")

    # Create worker task
    worker_task = asyncio.create_task(
        process_queue_worker(
            batch_size=batch_size,
            sleep_interval=sleep_interval,
            max_concurrent=max_concurrent
        )
    )

    # Wait for shutdown signal or worker completion
    await shutdown_event.wait()

    # Cancel worker task gracefully
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Worker task cancelled successfully")

def main():
    """Entry point for the worker process."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if sys.platform == 'win32':
        # Windows-specific asyncio setup for signal handling
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logger.info("Starting Crawl4AI Queue Worker")

    # Check if Redis is enabled in configuration
    if not config.redis.ENABLED:
        logger.error("Redis is disabled in configuration. Enable it to use the queue worker.")
        sys.exit(1)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker stopped by keyboard interrupt")
    except Exception as e:
        logger.exception(f"Worker failed with error: {str(e)}")

    logger.info("Crawl4AI Queue Worker shutdown complete")

if __name__ == "__main__":
    main()