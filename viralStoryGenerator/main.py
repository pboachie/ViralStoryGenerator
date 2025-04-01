# viralStoryGenerator/main.py
import os
import argparse
import asyncio
import multiprocessing
import uvicorn
import threading
import time
from fastapi.staticfiles import StaticFiles

from viralStoryGenerator.src.logger import logger as _logger, log_startup
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.api import app as api_router
from viralStoryGenerator.utils.scheduled_cleanup import cleanup_task
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.src.api_handlers import process_audio_queue

def main():
    """
    Main entry point for ViralStoryGenerator API.
    """
    _logger.debug("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(
        description="ViralStoryGenerator API - Generate viral stories via HTTP"
    )

    # API server command line arguments
    parser.add_argument("--host", type=str, default=os.environ.get("API_HOST", "0.0.0.0"),
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8000")),
                        help="Port to bind the server to")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("API_WORKERS", str(multiprocessing.cpu_count()))),
                        help="Number of worker processes")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--log-level", type=str,
                        default=os.environ.get("LOG_LEVEL", "info").lower(),
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")

    # Parse arguments
    args = parser.parse_args()

    # Log startup information using the new colored logger
    log_startup(
        environment=app_config.ENVIRONMENT,
        version=app_config.VERSION,
        storage_provider=app_config.storage.PROVIDER
    )

    # Start scheduled cleanup in separate thread
    cleanup_thread = threading.Thread(target=scheduled_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    _logger.info("Cleanup scheduler thread started")

    # Start uvicorn server
    uvicorn.run(
        "viralStoryGenerator.src.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


async def scheduled_cleanup():
    """
    Scheduled cleanup task to remove old files.
    Runs on a predefined interval.
    """
    _logger.info("Scheduled cleanup task started. Interval: {0} hours, Retention: {1} days".format(
        app_config.storage.CLEANUP_INTERVAL_HOURS,
        app_config.storage.FILE_RETENTION_DAYS
    ))

    while True:
        try:
            _logger.info("Running scheduled file cleanup")

            # Run the cleanup task
            cleanup_task._run_cleanup()

            # Process any queued audio generation
            process_audio_queue()

            # Sleep for the specified interval
            _logger.debug(f"Next cleanup scheduled in {app_config.storage.CLEANUP_INTERVAL_HOURS} hours")
            await asyncio.sleep(app_config.storage.CLEANUP_INTERVAL_HOURS * 3600)

        except Exception as e:
            _logger.error(f"Error in scheduled cleanup: {str(e)}")
            # Sleep for a shorter time on error
            await asyncio.sleep(3600)  # 1 hour


# Create necessary directories
os.makedirs(app_config.storage.AUDIO_STORAGE_PATH, exist_ok=True)
os.makedirs(app_config.storage.STORY_STORAGE_PATH, exist_ok=True)
os.makedirs(app_config.storage.STORYBOARD_STORAGE_PATH, exist_ok=True)

# Mount static directories for serving files directly
api_router.mount("/static/audio", StaticFiles(directory=app_config.storage.AUDIO_STORAGE_PATH), name="audio")
api_router.mount("/static/stories", StaticFiles(directory=app_config.storage.STORY_STORAGE_PATH), name="stories")
api_router.mount("/static/storyboards", StaticFiles(directory=app_config.storage.STORYBOARD_STORAGE_PATH), name="storyboards")

# Startup event handler
@api_router.on_event("startup")
async def startup_event():
    """Handle application startup tasks"""
    _logger.debug("Startup event triggered.")

    # Start scheduled cleanup task
    asyncio.create_task(scheduled_cleanup())

    # Process any queued audio generation tasks
    process_audio_queue()


# Shutdown event handler
@api_router.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown tasks"""
    _logger.debug("Shutdown event triggered.")

    # Perform any necessary cleanup
    if storage_manager:
        try:
            await storage_manager.close()
            _logger.debug("Storage manager closed successfully.")
        except Exception as e:
            _logger.error(f"Error closing storage manager: {str(e)}")


# Root endpoint
@api_router.get("/")
async def root():
    """Root endpoint that returns basic application information"""
    return {
        "app": "ViralStoryGenerator API",
        "version": app_config.VERSION,
        "environment": app_config.ENVIRONMENT,
        "docs": "/docs",
    }


if __name__ == "__main__":
    main()
