# viralStoryGenerator/main.py
import argparse
import sys
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.bin.run_api_server import main as api_main

def main():
    """
    Main entry point for ViralStoryGenerator API.
    """
    parser = argparse.ArgumentParser(
        description="ViralStoryGenerator API - Generate viral stories from web content"
    )

    # API server command line arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Parse arguments
    args = parser.parse_args()

    # Set arguments for API server
    sys.argv = [
        sys.argv[0],
        "--host", args.host,
        "--port", str(args.port)
    ]
    if args.reload:
        sys.argv.append("--reload")

    # Run API server
    api_main()

if __name__ == "__main__":
    # Configure basic logging
    _logger.debug("Starting ViralStoryGenerator API...")
    main()
