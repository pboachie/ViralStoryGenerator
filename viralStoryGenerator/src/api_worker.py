# viralStoryGenerator/src/api_worker.py
"""
API Worker for processing ViralStoryGenerator API requests via Redis Stream.
This module consumes requests published by the /api/generate endpoint.
"""
import asyncio
import os
import re
import signal
import sys
import tempfile
import time
import json
import uuid
import shutil
import datetime
from typing import Dict, Any, Optional

import viralStoryGenerator.src.logger

from viralStoryGenerator.models import (
    JobStatusResponse
)
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.crawl4ai_scraper import get_scrape_result, queue_scrape_request
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.llm import process_with_llm, clean_markdown_with_llm
from viralStoryGenerator.src.storyboard import generate_storyboard
import logging
from viralStoryGenerator.utils.text_processing import split_text_into_chunks
from viralStoryGenerator.utils.vector_db_manager import get_vector_db_client
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt
from viralStoryGenerator.utils.api_job_processor import process_api_job

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

# Constants
API_QUEUE_NAME = app_config.redis.QUEUE_NAME
API_RESULT_PREFIX = app_config.redis.RESULT_PREFIX
RESULT_TTL = app_config.redis.TTL
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))

# Global module-level connections
_message_broker = None
_vector_db_client = None

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, _):
    """Handle shutdown signals gracefully."""
    _logger.warning(f"Received signal {sig}, initiating shutdown...")
    shutdown_event.set()

    # Give tasks time to complete
    _logger.info("Waiting for current tasks to complete...")
    time.sleep(5)

    _logger.info("Shutdown complete.")
    sys.exit(0)

def preload_components(group_name: str):
    """Preload and initialize key components at startup."""
    global _message_broker

    # Initialize Redis message broker
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME) # Use QUEUE_NAME

    # Create consumer group if it doesn't exist using the provided group_name
    try:
        _message_broker.create_consumer_group(group_name=group_name)
        _logger.info(f"Ensured consumer group '{group_name}' exists for stream '{app_config.redis.QUEUE_NAME}'.")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            _logger.warning(f"Could not create consumer group '{group_name}': {e}")
        else:
            _logger.debug(f"Consumer group '{group_name}' already exists.")

    _logger.info(f"Worker components initialized successfully for stream '{app_config.redis.QUEUE_NAME}'.")

def get_message_broker() -> Optional[RedisMessageBroker]:
    """Get the pre-initialized message broker or create a new one if needed."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    # Initialize if not already done
    try:
        redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
        _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)
        _logger.info(f"Initialized API Worker RedisMessageBroker with stream: '{app_config.redis.QUEUE_NAME}'")
        return _message_broker
    except Exception as e:
        _logger.error(f"Failed to initialize Redis message broker: {e}")
        return None

def get_vector_db():
    """Get the pre-initialized vector DB client."""
    global _vector_db_client
    return _vector_db_client or get_vector_db_client()

async def process_api_jobs(group_name: str, consumer_name: str):
    """Process API jobs from the Redis stream."""
    while not shutdown_event.is_set():
        try:
            message_broker = get_message_broker()
            if not message_broker:
                _logger.error("Message broker unavailable, waiting before retry...")
                await asyncio.sleep(5)
                continue

            try:
                message_broker.create_consumer_group(group_name=group_name)
            except Exception as e:
                _logger.debug(f"Consumer group '{group_name}' already exists: {e}")

            # Consume messages from the stream using passed arguments
            messages = message_broker.consume_messages(
                group_name=group_name,
                consumer_name=consumer_name,
                count=MAX_CONCURRENT_JOBS,
                block=5000
            )

            if not messages:
                await asyncio.sleep(0.1)  # Small sleep if no messages
                continue

            for stream_name, stream_messages in messages:
                if not stream_messages:
                    continue

                # Process messages concurrently with a limit
                tasks = []

                for message_id, message_data in stream_messages:
                    if not isinstance(message_data, dict):
                        _logger.warning(f"Skipping message {message_id}: message_data is not a dict ({type(message_data)})")
                        message_broker.acknowledge_message(group_name, message_id)
                        continue

                    job_data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in message_data.items()
                    }

                    # Check if this is a system/initialization message
                    if "initialized" in job_data or "purged" in job_data:
                        _logger.debug(f"Skipping system message: {message_id}")
                        message_broker.acknowledge_message(group_name, message_id)
                        continue

                    job_id = job_data.get("job_id")
                    job_type = job_data.get("job_type")

                    if not job_id:
                        job_id = "unknown"
                        _logger.debug(f"Message {message_id} missing job_id")

                    if not job_type:
                        job_type = "unknown"
                        _logger.debug(f"Message {message_id} missing job_type")

                    # Skip invalid messages
                    if job_id == "unknown" and job_type == "unknown":
                        _logger.warning(f"Skipping invalid message {message_id} - missing job_id and job_type")
                        message_broker.acknowledge_message(group_name, message_id)
                        continue

                    _logger.info(f"Processing message {message_id} (job_id: {job_id}, type: {job_type})")

                    # Parse JSON fields if needed
                    for key, value in job_data.items():
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                job_data[key] = json.loads(value)
                            except:
                                pass

                    # Create task to process the job
                    task = asyncio.create_task(process_api_job(job_data, consumer_name, group_name, message_broker))
                    tasks.append(task)

                    # Acknowledge message after creating task
                    message_broker.acknowledge_message(group_name, message_id)

                # Wait for all tasks to complete
                if tasks:
                    await asyncio.gather(*tasks)

        except Exception as e:
            _logger.exception(f"Error in process_api_jobs loop: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def run_worker():
    """Run the API worker with graceful shutdown handling."""
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)

    group_name = app_config.redis.API_WORKER_GROUP_NAME
    consumer_name = f"api-worker-{uuid.uuid4().hex[:8]}"

    _logger.info(f"Starting API worker (PID: {os.getpid()}, Group: {group_name}, Consumer: {consumer_name})...")

    # Preload components
    preload_components(group_name)

    # Process jobs indefinitely
    await process_api_jobs(group_name, consumer_name)

def main():
    """Entry point for the worker process."""
    if os.name == 'nt':  # Windows
        # Use ProactorEventLoop for Windows compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        _logger.info("Worker stopped by KeyboardInterrupt.")
    except Exception as e:
        _logger.exception(f"Worker failed unexpectedly: {e}")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError as e:
            if "Cannot run the event loop while another loop is running" in str(e):
                 _logger.warning("Attempted to shutdown asyncgens while loop was closing.")
            else:
                 raise
        finally:
            if loop.is_running():
                 loop.stop()
            if not loop.is_closed():
                 loop.close()
            _logger.info("Event loop closed.")


if __name__ == "__main__":
    main()