# viralStoryGenerator/main.py
import argparse
import asyncio
import os
import multiprocessing
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config
from fastapi.staticfiles import StaticFiles
from viralStoryGenerator.src.api import app as api_router
from viralStoryGenerator.utils.scheduled_cleanup import cleanup_task
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.src.api_handlers import (
    process_audio_queue
)
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
    _logger.debug(f"Arguments parsed: {args}")

    # Configure startup based on environment
    is_development = app_config.ENVIRONMENT.lower() == "development"

    # Set arguments for API server
    # Run API server via the start_api_server function in api.py
    _logger.info(f"Starting ViralStoryGenerator API with {args.workers} workers on {args.host}:{args.port}...")
    _logger.debug("Starting API server...")

    from viralStoryGenerator.src.api import start_api_server
    start_api_server(host=args.host, port=args.port, reload=args.reload or is_development)
    _logger.debug("API server started successfully.")

async def scheduled_cleanup():
    """Run periodic cleanup of old files based on retention policy"""
    while True:
        try:
            # Sleep first to avoid cleanup right at startup
            await asyncio.sleep(24 * 60 * 60)  # Run once per day

            # Get the retention period from config
            retention_days = app_config.storage.FILE_RETENTION_DAYS

            if retention_days > 0:
                _logger.info(f"Running scheduled cleanup of files older than {retention_days} days")
                deleted_count = storage_manager.cleanup_old_files(retention_days)
                _logger.info(f"Cleanup complete: {deleted_count} files removed")
        except Exception as e:
            _logger.error(f"Error in scheduled cleanup: {e}")


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
    """Perform tasks when the application starts"""
    _logger.debug("Startup event triggered.")
    _logger.info(f"Starting {app_config.APP_TITLE} v{app_config.VERSION}")
    _logger.info(f"Environment: {app_config.ENVIRONMENT}")
    _logger.info(f"Storage provider: {app_config.storage.PROVIDER}")
        # Process any queued audio files at startup
    process_audio_queue()

        # Start the cleanup background task if enabled
    if app_config.storage.FILE_RETENTION_DAYS > 0:
        asyncio.create_task(scheduled_cleanup())

    # Start the scheduled cleanup task
    if app_config.storage.FILE_RETENTION_DAYS > 0:
        cleanup_started = cleanup_task.start()
        if cleanup_started:
            _logger.info(f"Scheduled file cleanup enabled: Every {app_config.storage.CLEANUP_INTERVAL_HOURS} hours, {app_config.storage.FILE_RETENTION_DAYS} days retention")
        else:
            _logger.warning("Failed to start scheduled file cleanup")
    _logger.debug("Startup event completed.")

# Shutdown event handler
@api_router.on_event("shutdown")
async def shutdown_event():
    """Perform tasks when the application shuts down"""
    _logger.debug("Shutdown event triggered.")
    _logger.info("Application shutting down")

    # Stop the scheduled cleanup task
    cleanup_task.stop()
    _logger.debug("Shutdown event completed.")


# Root endpoint
@api_router.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": app_config.APP_TITLE,
        "version": app_config.VERSION,
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    _logger.debug("Starting ViralStoryGenerator API...")
    main()
