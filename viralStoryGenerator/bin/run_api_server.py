#!/usr/bin/env python
"""
Script to run the ViralStoryGenerator API server.

Usage:
  python run_api_server.py [--host HOST] [--port PORT]
"""
import argparse
import os
import sys
import uvicorn
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from viralStoryGenerator.utils.config import config as app_config

def main():
    parser = argparse.ArgumentParser(description="Run the ViralStoryGenerator API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, app_config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api_server.log')
        ]
    )

    print(f"Starting ViralStoryGenerator API server on {args.host}:{args.port}")
    print(f"Documentation available at http://{args.host}:{args.port}/docs")

    # Run the server
    uvicorn.run(
        "viralStoryGenerator.src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=app_config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()