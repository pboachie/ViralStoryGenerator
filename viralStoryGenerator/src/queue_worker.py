"""
Queue Worker for processing main generation jobs from the Redis queue (api_jobs).
Run this script separately to start a background worker process.
"""
import asyncio
import signal
import sys
import os
import json
import uuid
import time
from typing import Dict

from ..utils.config import config as app_config
from ..utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.src.logger import logger as _logger
from ..utils.api_job_processor import process_api_job

_message_broker = None

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    if not shutdown_event.is_set():
        _logger.info(f"Received signal {sig}, shutting down API job worker...")
        shutdown_event.set()

def preload_components():
    """Preload and initialize key components at startup."""
    global _message_broker

    redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME) # Use QUEUE_NAME

    group_name = app_config.redis.API_WORKER_GROUP_NAME
    try:
        _message_broker.create_consumer_group(group_name=group_name)
        _logger.info(f"Ensured consumer group '{group_name}' exists for stream '{app_config.redis.QUEUE_NAME}'.")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            _logger.warning(f"Could not create consumer group '{group_name}': {e}")
        else:
            _logger.debug(f"Consumer group '{group_name}' already exists.")

    _message_broker.ensure_stream_exists(app_config.redis.QUEUE_NAME)

    _logger.info(f"API Job worker components initialized successfully for stream '{app_config.redis.QUEUE_NAME}'.")


def get_message_broker() -> RedisMessageBroker:
    """Get the pre-initialized message broker or create a new one if needed."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    _logger.warning("Message broker accessed before preload, initializing now.")
    redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)
    return _message_broker


async def process_single_api_job(job_data_raw: Dict[bytes, bytes], message_id: str, group_name: str, consumer_name: str):
    """Processes a single job received from the api_jobs stream."""
    job_id = None
    message_broker = get_message_broker()
    try:
        # Decode job data from bytes
        job_data = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in job_data_raw.items()
        }
        job_id = job_data.get("job_id")
        job_type = job_data.get("job_type")

        if not job_id:
            job_id = str(uuid.uuid4())
            _logger.warning(f"Job message {message_id} missing job_id, assigned: {job_id}")
            job_data["job_id"] = job_id

        _logger.info(f"Processing API job {job_id} (Type: {job_type or 'Unknown'}). Message ID: {message_id}")

        await process_api_job(job_data, consumer_name, group_name, message_broker)

    except Exception as e:
        _logger.exception(f"Error processing API job {job_id or message_id}: {e}")
        if job_id and message_broker:
            try:
                message_broker.track_job_progress(job_id, "failed", {"error": f"Worker failed: {str(e)}"})
            except Exception as track_e:
                _logger.error(f"Failed to update status to failed for job {job_id}: {track_e}")
    finally:
        if message_broker:
            try:
                message_broker.acknowledge_message(group_name, message_id)
                _logger.debug(f"Acknowledged message {message_id} for job {job_id or 'unknown'}")
            except Exception as ack_e:
                 _logger.error(f"Failed to acknowledge message {message_id}: {ack_e}")


async def run_api_job_consumer(consumer_name: str):
    """Continuously consumes and processes jobs from the api_jobs stream."""
    message_broker = get_message_broker()
    group_name = app_config.redis.API_WORKER_GROUP_NAME
    batch_size = app_config.redis.WORKER_BATCH_SIZE
    sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent = app_config.redis.WORKER_MAX_CONCURRENT
    active_tasks = set()

    _logger.info(f"API Job worker '{consumer_name}' starting consumption loop for stream '{app_config.redis.QUEUE_NAME}'.")
    _logger.info(f"Config - BatchSize: {batch_size}, SleepInterval: {sleep_interval}s, MaxConcurrent: {max_concurrent}")

    while not shutdown_event.is_set():
        # Manage concurrency
        if len(active_tasks) >= max_concurrent and active_tasks:
            _logger.debug(f"Concurrency limit ({max_concurrent}) reached. Waiting...")
            done, active_tasks = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
            )
            if done:
                 _logger.debug(f"{len(done)} API job task(s) completed.")
            continue

        # Check if shutdown is requested
        if shutdown_event.is_set():
            break

        try:
            # Fetch new messages
            available_slots = max_concurrent - len(active_tasks)
            if available_slots <= 0:
                await asyncio.sleep(0.1)
                continue

            messages = message_broker.consume_messages(
                group_name=group_name,
                consumer_name=consumer_name,
                count=min(available_slots, batch_size),
                block=2000
            )

            if not messages:
                # _logger.debug("No new API jobs. Sleeping...")
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"Received {len(messages[0][1])} new API job message(s).")
            for stream_name, stream_messages in messages:
                for message_id_bytes, message_data_bytes in stream_messages:
                    if shutdown_event.is_set(): break

                    message_id = message_id_bytes.decode()

                    job_data_temp = {}
                    try:
                        job_data_temp = {
                            k.decode() if isinstance(k, bytes) else k:
                            v.decode() if isinstance(v, bytes) else v
                            for k, v in message_data_bytes.items()
                        }
                    except Exception as decode_e:
                        _logger.error(f"Failed to decode message {message_id}: {decode_e}. Skipping.")
                        message_broker.acknowledge_message(group_name, message_id)
                        continue

                    # --- Job Type Filtering --- >
                    job_type = job_data_temp.get("job_type")
                    if job_type != "generate_story":
                        if job_type is None or job_type in ["pending", "processing", "completed", "failed", "initialized", "purged"]:
                            #  _logger.debug(f"Skipping non-actionable message {message_id} (Type: {job_type}).")
                            pass
                        else:
                             _logger.warning(f"Skipping message {message_id} with unexpected job_type: {job_type}.")
                        try:
                            message_broker.acknowledge_message(group_name, message_id)
                        except Exception as ack_e:
                            _logger.error(f"Failed to acknowledge message {message_id}: {ack_e}")
                        continue
                    # --- End Job Type Filtering ---

                    task = asyncio.create_task(
                        process_single_api_job(message_data_bytes, message_id, group_name, consumer_name)
                    )
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
                    _logger.debug(f"Created task for API job message {message_id}. Active tasks: {len(active_tasks)}")

                if shutdown_event.is_set(): break

        except asyncio.CancelledError:
             _logger.info("API job consumption loop cancelled.")
             break
        except Exception as e:
            _logger.exception(f"Error in API job consumption loop: {e}")
            await asyncio.sleep(sleep_interval * 2)

    # Cleanup during shutdown
    if active_tasks:
        _logger.info(f"Waiting for {len(active_tasks)} remaining API job tasks to complete...")
        try:
            await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=30.0)
            _logger.info("Remaining API job tasks finished.")
        except asyncio.TimeoutError:
            _logger.warning("Timeout waiting for remaining API job tasks during shutdown.")
        except Exception as e:
             _logger.exception(f"Error during API job task cleanup: {e}")

    _logger.info("API job consumption loop finished.")


async def run_worker():
    """Run the API job worker with graceful shutdown handling."""
    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sys.platform != 'win32':
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_async_shutdown(s)))
        else:
            signal.signal(sig, handle_shutdown)

    consumer_name = f"api-worker-{uuid.uuid4().hex[:8]}"
    _logger.info(f"Starting API Job worker (PID: {os.getpid()}, Consumer: {consumer_name})...")

    # Preload components
    preload_components()

    # Start consuming jobs
    await run_api_job_consumer(consumer_name)

async def handle_async_shutdown(sig):
     """Async compatible shutdown handler"""
     if not shutdown_event.is_set():
         _logger.warning(f"Received signal {sig}, initiating async shutdown...")
         shutdown_event.set()

def main():
    """Entry point for the worker process."""
    if os.name == 'nt': # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    _logger.info("Starting API Job Queue Worker")

    # Check if Redis is enabled
    if not app_config.redis.ENABLED:
        _logger.error("Redis is disabled in configuration. Enable REDIS_ENABLED=True to use the queue worker.")
        sys.exit(1)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        _logger.info("API Job worker stopped by keyboard interrupt.")
    except Exception as e:
        _logger.exception(f"API Job worker failed unexpectedly: {e}")
    finally:
        # Cleanup asyncio loop resources
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError as e:
             if "Cannot run the event loop while another loop is running" not in str(e): raise
        finally:
            if loop.is_running(): loop.stop()
            if not loop.is_closed(): loop.close()
            _logger.info("Event loop closed.")

    _logger.info("API Job Queue Worker shutdown complete.")

if __name__ == "__main__":
    main()