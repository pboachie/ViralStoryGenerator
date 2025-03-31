# viralStoryGenerator/main.py
import argparse
import sys
import os
import multiprocessing
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils import config
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from viralStoryGenerator.src.api import app as api_router
from viralStoryGenerator.utils.scheduled_cleanup import cleanup_task

def main():
    """
    Main entry point for ViralStoryGenerator API.
    """
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

    # Configure startup based on environment
    is_development = config.config.ENVIRONMENT.lower() == "development"

    # Set arguments for API server
    uvicorn_args = [
        "viralStoryGenerator.src.api:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.log_level
    ]

    # Use reload in development mode if specified
    if args.reload or is_development:
        uvicorn_args.append("--reload")

    # Use multiple workers in production mode
    if not is_development:
        uvicorn_args.extend(["--workers", str(args.workers)])

    # Run API server via the start_api_server function in api.py
    _logger.info(f"Starting ViralStoryGenerator API with {args.workers} workers on {args.host}:{args.port}...")

    from viralStoryGenerator.src.api import start_api_server
    start_api_server(host=args.host, port=args.port, reload=args.reload or is_development)

# Set up the FastAPI application
app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.http.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router)

# Create necessary directories
os.makedirs(config.storage.AUDIO_STORAGE_PATH, exist_ok=True)
os.makedirs(config.storage.STORY_STORAGE_PATH, exist_ok=True)
os.makedirs(config.storage.STORYBOARD_STORAGE_PATH, exist_ok=True)

# Mount static directories for serving files directly
app.mount("/static/audio", StaticFiles(directory=config.storage.AUDIO_STORAGE_PATH), name="audio")
app.mount("/static/stories", StaticFiles(directory=config.storage.STORY_STORAGE_PATH), name="stories")
app.mount("/static/storyboards", StaticFiles(directory=config.storage.STORYBOARD_STORAGE_PATH), name="storyboards")

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Perform tasks when the application starts"""
    _logger.info(f"Starting {config.APP_TITLE} v{config.VERSION}")
    _logger.info(f"Environment: {config.ENVIRONMENT}")
    _logger.info(f"Storage provider: {config.storage.PROVIDER}")
    
    # Start the scheduled cleanup task
    if config.storage.FILE_RETENTION_DAYS > 0:
        cleanup_started = cleanup_task.start()
        if cleanup_started:
            _logger.info(f"Scheduled file cleanup enabled: Every {config.storage.CLEANUP_INTERVAL_HOURS} hours, {config.storage.FILE_RETENTION_DAYS} days retention")
        else:
            _logger.warning("Failed to start scheduled file cleanup")

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Perform tasks when the application shuts down"""
    _logger.info("Application shutting down")
    
    # Stop the scheduled cleanup task
    cleanup_task.stop()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    # Include cleanup task status if it's running
    cleanup_status = cleanup_task.status() if cleanup_task.is_running else None
    
    return {
        "status": "healthy",
        "version": config.VERSION,
        "environment": config.ENVIRONMENT,
        "storage_provider": config.storage.PROVIDER,
        "cleanup": cleanup_status
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": config.APP_TITLE,
        "version": config.VERSION,
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    _logger.debug("Starting ViralStoryGenerator API...")
    main()
