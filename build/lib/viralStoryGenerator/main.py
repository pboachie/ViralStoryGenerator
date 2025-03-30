# viralStoryGenerator/main.py
import argparse
import sys
import os
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.src.cli import cli_main
from viralStoryGenerator.bin.run_api_server import main as api_main

def main():
    """
    Main entry point for ViralStoryGenerator.
    Provides options to run in CLI mode or API server mode.
    """
    parser = argparse.ArgumentParser(
        description="ViralStoryGenerator - Generate viral stories from web content or local files"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # CLI command (default)
    cli_parser = subparsers.add_parser("cli", help="Run in CLI mode (default)")
    cli_parser.add_argument("--topic", type=str, help="Topic for the story")
    cli_parser.add_argument("--sources", type=str, help="Path to the sources folder")

    # API server command
    api_parser = subparsers.add_parser("api", help="Run as API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Parse arguments
    args = parser.parse_args()

    # Default to CLI mode if no command is specified
    if not args.command:
        # If run with no arguments, show help
        if len(sys.argv) == 1:
            parser.print_help()
            return
        args.command = "cli"

    # Dispatch based on command
    if args.command == "api":
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
    else:
        # Run CLI
        cli_args = []
        if hasattr(args, "topic") and args.topic:
            cli_args.extend(["--topic", args.topic])
        if hasattr(args, "sources") and args.sources:
            cli_args.extend(["--sources", args.sources])

        # Replace sys.argv with our custom arguments
        sys.argv = [sys.argv[0]] + cli_args
        cli_main()

if __name__ == "__main__":
    # Configure basic logging
    _logger.debug("Starting ViralStoryGenerator...")
    main()
