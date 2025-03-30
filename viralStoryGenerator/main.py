# viralStoryGenerator/main.py
import argparse
import sys
import os
import multiprocessing
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils import config

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

if __name__ == "__main__":
    _logger.debug("Starting ViralStoryGenerator API...")
    main()
