# viralStoryGenerator/src/worker_runner.py
"""
Unified worker runner script.
Launches the specified worker type (api, queue, or scrape) based on command-line arguments.
"""

import argparse
import sys
import os
import uvicorn
import logging
import importlib

from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger

try:
    from watchfiles import run_process
except ImportError:
    run_process = None

def worker_entry(module_path):
    mod = importlib.import_module(module_path)
    mod.main()

def run_api_server(args, uvicorn_extra_args=None):
    """Starts the FastAPI API server using Uvicorn."""
    log_level_name = app_config.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('api_server.log', mode='a')
        ]
    )
    # Get the uvicorn logger and set its level
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.setLevel(log_level)
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(log_level)


    _logger.info(f"Starting API server on {args.host}:{args.port} (Reload: {args.reload})")
    _logger.info(f"API Documentation available at http://{args.host}:{args.port}/docs")

    uvicorn_args = dict(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level_name.lower(),
        factory=False
    )
    # Remove None values
    uvicorn_args = {k: v for k, v in uvicorn_args.items() if v is not None}

    reload_dirs = [os.path.dirname(os.path.abspath(__file__))] if args.reload else []

    if uvicorn_extra_args:
        cmd = [sys.executable, "-m", "uvicorn"]
        for k, v in uvicorn_args.items():
            if k == "factory":
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k.replace('_', '-')}")
            else:
                cmd.append(f"--{k.replace('_', '-')}")
                cmd.append(str(v))
        for d in reload_dirs:
            cmd.append("--reload-dir")
            cmd.append(d)
        cmd += uvicorn_extra_args
        cmd.append("viralStoryGenerator.src.api:app")
        os.execvp(cmd[0], cmd)
    else:
        uvicorn.run("viralStoryGenerator.src.api:app", reload_dirs=reload_dirs if args.reload else None, **uvicorn_args)

def main():
    parser = argparse.ArgumentParser(description="Run a specific ViralStoryGenerator worker or the API server.")
    subparsers = parser.add_subparsers(dest="command", required=True, help='Select command to run')

    # Subparser for the API server
    parser_api = subparsers.add_parser('api', help='Run the FastAPI API server')
    parser_api.add_argument("--host", type=str, default=app_config.http.HOST, help="Host to bind the server to")
    parser_api.add_argument("--port", type=int, default=app_config.http.PORT, help="Port to bind the server to")
    parser_api.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser_api.set_defaults(func=run_api_server)

    # Subparser for background workers
    parser_worker = subparsers.add_parser('worker', help='Run a background worker process')
    parser_worker.add_argument(
        "--worker-type",
        type=str,
        required=True,
        choices=["queue", "scrape"],
        help="The type of background worker to run (queue or scrape)."
    )
    parser_worker.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    # Add worker-specific arguments here if needed in the future
    # parser_worker.set_defaults(func=run_background_worker) # We'll call the specific main directly

    args, unknown = parser.parse_known_args()

    if args.command == "api":
        args.func(args, uvicorn_extra_args=unknown)
    elif args.command == "worker":
        worker_type = args.worker_type.lower()
        reload_enabled = getattr(args, 'reload', False)
        _logger.info(f"Attempting to start background worker of type: {worker_type}")
        try:
            if reload_enabled:
                if not run_process:
                    _logger.error("watchfiles is not installed. Please install it to use --reload with workers.")
                    sys.exit(1)
                # Determine the module to run
                if worker_type == "queue":
                    module_path = "viralStoryGenerator.src.queue_worker"
                elif worker_type == "scrape":
                    module_path = "viralStoryGenerator.src.scrape_worker"
                else:
                    _logger.error(f"Unknown worker type: {worker_type}")
                    sys.exit(1)
                watch_dir = os.path.join(os.path.dirname(__file__))
                _logger.info(f"[RELOAD] Watching {watch_dir} for changes...")
                from functools import partial
                run_process(watch_dir, target=partial(worker_entry, module_path))
            else:
                if worker_type == "queue":
                    from viralStoryGenerator.src.queue_worker import main as queue_main
                    _logger.info("Launching Queue worker (for crawl4ai)...")
                    queue_main()
                elif worker_type == "scrape":
                    from viralStoryGenerator.src.scrape_worker import main as scrape_main
                    _logger.info("Launching Scrape worker...")
                    scrape_main()
        except ImportError as e:
            _logger.critical(f"Failed to import necessary modules for worker type '{worker_type}': {e}")
            _logger.critical("Ensure the worker script and its dependencies exist and are correctly referenced.")
            sys.exit(1)
        except Exception as e:
            _logger.exception(f"An unexpected error occurred while running the {worker_type} worker: {e}")
            sys.exit(1)
    else:
        _logger.error(f"Invalid command specified: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
