# viralStoryGenerator/main.py
import datetime
import os
import argparse
import asyncio
import multiprocessing
import uvicorn
import time
import sys
from fastapi.staticfiles import StaticFiles
import logging

from viralStoryGenerator.utils.config import app_config, validate_config_on_startup, ConfigError
from viralStoryGenerator.src.logger import log_startup
from viralStoryGenerator.src.api import app as api_router
from viralStoryGenerator.utils.scheduled_cleanup import cleanup_task
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.src.api_handlers import process_audio_queue


_logger = logging.getLogger(__name__)

# --- Configuration Validation ---
try:
    validate_config_on_startup(app_config)
except ConfigError as e:
    # Use print/stderr as logger might not be fully configured if run directly initially
    print(f"CRITICAL CONFIGURATION ERROR: {e}", file=sys.stderr)
    print("Application cannot start.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
     print(f"Unexpected error during configuration validation: {e}", file=sys.stderr)
     sys.exit(1)

async def scheduled_cleanup_runner():
    """
    Asynchronous task to periodically run cleanup and audio queue processing.
    """
    interval_hours = app_config.storage.CLEANUP_INTERVAL_HOURS
    retention_days = app_config.storage.FILE_RETENTION_DAYS
    is_enabled = retention_days > 0 or os.path.exists(app_config.AUDIO_QUEUE_DIR)

    if not is_enabled:
        _logger.info("Scheduled cleanup and audio queue processing disabled (retention=0 and no audio queue dir).")
        return

    _logger.info(f"Scheduled task started. Interval: {interval_hours} hours. Retention: {retention_days} days.")

    while True:
        next_run_time = time.time() + (interval_hours * 3600)
        next_run_dt = datetime.datetime.fromtimestamp(next_run_time)
        _logger.info(f"Running scheduled tasks (Cleanup & Audio Queue)... Next run approx: {next_run_dt.isoformat()}")

        try:
            # Run the storage cleanup task (if retention enabled)
            if retention_days > 0:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, cleanup_task._run_cleanup_job)
            else:
                 _logger.debug("Skipping file cleanup as retention_days <= 0.")

            # TODO: Make process_audio_queue async or run in executor if it's blocking
            _logger.debug("Running audio queue processing...")
            # If process_audio_queue is potentially blocking:
            # await loop.run_in_executor(None, process_audio_queue)
            # If it's quick/non-blocking:
            process_audio_queue()

            # Sleep until the next interval
            sleep_duration = interval_hours * 3600
            _logger.debug(f"Scheduled tasks finished. Sleeping for {sleep_duration:.0f} seconds.")
            await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
             _logger.info("Scheduled cleanup task cancelled.")
             break
        except Exception as e:
            _logger.exception(f"Error in scheduled cleanup/audio processing task: {e}")
            _logger.info("Sleeping for 1 hour before retrying scheduled tasks...")
            await asyncio.sleep(3600)

# --- FastAPI App Setup ---

try:
    os.makedirs(app_config.storage.AUDIO_STORAGE_PATH, exist_ok=True)
    os.makedirs(app_config.storage.STORY_STORAGE_PATH, exist_ok=True)
    os.makedirs(app_config.storage.STORYBOARD_STORAGE_PATH, exist_ok=True)
    os.makedirs(app_config.AUDIO_QUEUE_DIR, exist_ok=True)
except OSError as e:
     _logger.critical(f"Failed to create necessary storage directories: {e}")
     sys.exit(1)

if app_config.storage.PROVIDER == "local":
    api_router.mount("/static/audio", StaticFiles(directory=app_config.storage.AUDIO_STORAGE_PATH), name="static_audio")
    api_router.mount("/static/stories", StaticFiles(directory=app_config.storage.STORY_STORAGE_PATH), name="static_stories")
    api_router.mount("/static/storyboards", StaticFiles(directory=app_config.storage.STORYBOARD_STORAGE_PATH), name="static_storyboards")
    _logger.info("Mounted static directories for local file serving.")

# Startup event handler
@api_router.on_event("startup")
async def startup_event():
    """Handle application startup tasks."""
    _logger.debug("Application startup event triggered.")
    _logger.info("Creating scheduled cleanup/audio processing task...")
    asyncio.create_task(scheduled_cleanup_runner())

    # Run audio queue processing once immediately on startup
    _logger.info("Running initial audio queue processing on startup...")
    await asyncio.get_running_loop().run_in_executor(None, process_audio_queue) # If blocking
    # process_audio_queue() # If quick


# Shutdown event handler
@api_router.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown tasks."""
    _logger.debug("Application shutdown event triggered.")
    # Perform any necessary cleanup
    if storage_manager:
        try:
            closed = await storage_manager.close()
            if closed:
                 _logger.info("Storage manager closed successfully.")
            else:
                 _logger.warning("Storage manager close method indicated no action or failure.")
        except Exception as e:
            _logger.error(f"Error closing storage manager during shutdown: {e}")
    _logger.info("Application shutdown sequence completed.")

# --- Main Function (Entry point for running the server) ---
def main():
    """
    Parses arguments and starts the Uvicorn server for the FastAPI application.
    """
    _logger.debug("Parsing command-line arguments for API server...")
    parser = argparse.ArgumentParser(
        description="ViralStoryGenerator API Server"
    )

    # API server command line arguments using defaults from config
    parser.add_argument("--host", type=str, default=app_config.http.HOST,
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=app_config.http.PORT,
                        help="Port to bind the server to")
    parser.add_argument("--workers", type=int, default=app_config.http.WORKERS,
                        help="Number of worker processes (ignored if --reload is used)")
    parser.add_argument("--reload", action="store_true", default=False,
                        help="Enable auto-reload for development (forces workers=1)")
    parser.add_argument("--log-level", type=str, default=app_config.LOG_LEVEL.lower(),
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level for Uvicorn")

    args = parser.parse_args()

    # Log startup information using the colored logger
    log_startup(
        environment=app_config.ENVIRONMENT,
        version=app_config.VERSION,
        storage_provider=app_config.storage.PROVIDER
    )


    _logger.info(f"Starting Uvicorn server on {args.host}:{args.port}...")
    uvicorn.run(
        "viralStoryGenerator.src.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
        access_log=False,
        ssl_keyfile=app_config.http.SSL_KEY_FILE if app_config.http.SSL_ENABLED else None,
        ssl_certfile=app_config.http.SSL_CERT_FILE if app_config.http.SSL_ENABLED else None,
    )

# --- Script Execution Guard ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        sys.argv.pop(1)
        _logger.info("Starting API server via main entry point...")
    else:
         _logger.info("Starting API server...")

    # Run the main server function
    main()