import viralStoryGenerator.src.logger
import asyncio
import signal
import sys
import os
import json
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple


from ..utils.config import config as app_config
from ..utils.redis_manager import RedisMessageBroker
import logging
from ..utils.api_job_processor import process_api_job

_logger = logging.getLogger(__name__)

_message_broker = None

shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    if not shutdown_event.is_set():
        _logger.info(f"Received signal {sig}, shutting down API job worker...")
        shutdown_event.set()

async def preload_components():
    global _message_broker

    redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)
    await _message_broker.initialize()

    _logger.debug(f"API Job worker components initialized for stream '{app_config.redis.QUEUE_NAME}'")

def get_message_broker() -> RedisMessageBroker:
    """Get the pre-initialized message broker or create a new one if needed. Ensures initialization."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    _logger.warning("Message broker accessed before preload, initializing now.")
    redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=app_config.redis.QUEUE_NAME)
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if not loop.is_running():
        loop.run_until_complete(_message_broker.initialize())
    return _message_broker


async def process_single_api_job(job_data_raw: Dict[bytes, bytes], message_id: str, group_name: str, consumer_name: str):
    """Processes a single job received from the api_jobs stream."""
    job_id = None
    message_broker = get_message_broker()
    try:
        job_data_decoded: Dict[str, Any] = {}
        for k, v in job_data_raw.items():
            key_decoded = k.decode() if isinstance(k, bytes) else str(k)
            value_decoded = v.decode() if isinstance(v, bytes) else v
            job_data_decoded[key_decoded] = value_decoded

        job_data = job_data_decoded

        job_id_any = job_data.get("job_id")
        job_id = str(job_id_any) if job_id_any is not None else str(uuid.uuid4())
        if job_id_any is None:
            _logger.warning(f"Job message {message_id} missing job_id, assigned: {job_id}")
            job_data["job_id"] = job_id

        job_type = job_data.get("job_type")

        _logger.debug(f"Processing API job {job_id} (Type: {job_type or 'Unknown'}). Message ID: {message_id}")

        await process_api_job(job_data, consumer_name, group_name, message_broker)

    except Exception as e:
        _logger.exception(f"Error processing API job {job_id or message_id}: {e}")
        if job_id and message_broker:
            try:
                await message_broker.track_job_progress(job_id, "failed", {"error": f"Worker failed: {str(e)}"})
            except Exception as track_e:
                _logger.error(f"Failed to update status to failed for job {job_id}: {track_e}")
    finally:
        if message_broker:
            try:
                await message_broker.acknowledge_message(message_id)
                _logger.debug(f"Acknowledged message {message_id} for job {job_id or 'unknown'}")
            except Exception as ack_e:
                 _logger.error(f"Failed to acknowledge message {message_id}: {ack_e}")


async def run_api_job_consumer(consumer_name: str):
    """Continuously consumes and processes jobs from the api_jobs stream."""
    message_broker = get_message_broker()
    batch_size = app_config.redis.WORKER_BATCH_SIZE
    sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent = app_config.redis.WORKER_MAX_CONCURRENT
    active_tasks = set()

    _logger.debug(f"API Job worker '{consumer_name}' starting consumption loop on '{app_config.redis.QUEUE_NAME}'")
    _logger.debug(f"Config - BatchSize: {batch_size}, SleepInterval: {sleep_interval}s, MaxConcurrent: {max_concurrent}")

    while not shutdown_event.is_set():
        if len(active_tasks) >= max_concurrent and active_tasks:
            _logger.debug(f"Concurrency limit ({max_concurrent}) reached. Waiting...")
            done, active_tasks = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
            )
            if done:
                 _logger.debug(f"{len(done)} API job task(s) completed.")
            continue

        if shutdown_event.is_set():
            break

        try:
            available_slots = max_concurrent - len(active_tasks)
            if (available_slots <= 0):
                await asyncio.sleep(0.1)
                continue

            messages: List[Tuple[str, Dict[bytes, bytes]]] = await message_broker.consume_messages(
                count=min(available_slots, batch_size),
                block_ms=2000
            )

            if not messages:
                await asyncio.sleep(sleep_interval)
                continue

            _logger.debug(f"Received {len(messages)} new API job message(s).")
            for stream_id, raw_fields_payload in messages:
                if shutdown_event.is_set(): break

                if not stream_id or not raw_fields_payload or not isinstance(raw_fields_payload, dict):
                    _logger.error(f"Invalid message structure received. Stream ID: {stream_id}, Raw Fields: {raw_fields_payload}. Skipping.")
                    if stream_id:
                        try:
                            await message_broker.acknowledge_message(stream_id)
                        except Exception as ack_e:
                            _logger.error(f"Failed to acknowledge malformed message {stream_id}: {ack_e}")
                    continue

                
                payload_json_str = raw_fields_payload.get(b"payload")

                parsed_payload_dict: Optional[Dict[str, Any]] = None
                job_type: Optional[str] = None
                actual_job_data_for_processing: Optional[Dict[str, Any]] = None

                if payload_json_str and isinstance(payload_json_str, str):
                    try:
                        parsed_payload_dict = json.loads(payload_json_str)

                        if isinstance(parsed_payload_dict, dict):
                            job_type = parsed_payload_dict.get("job_type")
                            actual_job_data_for_processing = parsed_payload_dict.get("data")

                            
                            if job_type is None and isinstance(actual_job_data_for_processing, dict):
                                job_type = actual_job_data_for_processing.get("job_type")

                            
                            if not isinstance(actual_job_data_for_processing, dict):
                                _logger.warning(f"Message {stream_id}: 'data' field in parsed payload is missing or not a dict. Parsed payload: {parsed_payload_dict}. Will attempt to use root as job data if job_type is present.")
                                if job_type and "job_id" in parsed_payload_dict:
                                    actual_job_data_for_processing = parsed_payload_dict
                                else:
                                    actual_job_data_for_processing = None
                        else:
                            _logger.error(f"Parsed payload for message {stream_id} is not a dictionary. Payload string: {payload_json_str}")
                    except json.JSONDecodeError as e:
                        _logger.error(f"Could not parse payload JSON for message {stream_id}. Error: {e}. Payload string: '{payload_json_str}'")
                else:
                    _logger.warning(f"Message {stream_id} is missing 'payload' field, it's not a string, or it's empty. Raw Fields: {raw_fields_payload}")

                
                if job_type != "generate_story":
                    log_message = f"Message {stream_id} with job_type '{job_type}' (from JSON payload)"
                    if job_type is None:
                        log_message = f"Message {stream_id} has missing, undecodable, or non-string job_type in its JSON payload"

                    if job_type is None or job_type not in ["pending", "processing", "completed", "failed", "initialized", "purged"]:
                        _logger.warning(f"{log_message}. Acknowledging and skipping. Full raw payload: {raw_fields_payload}")
                    else:
                        _logger.debug(f"{log_message}. Skipping non-generate_story job.")

                    try:
                        await message_broker.acknowledge_message(stream_id)
                    except Exception as e_ack:
                        _logger.error(f"Failed to acknowledge message {stream_id} after skipping: {e_ack}")
                    continue

                if not isinstance(actual_job_data_for_processing, dict):
                    _logger.error(f"Skipping message {stream_id} due to 'actual_job_data_for_processing' not being a valid dictionary after payload processing. Value: {actual_job_data_for_processing}")
                    try:
                        await message_broker.acknowledge_message(stream_id)
                    except Exception as e_ack:
                        _logger.error(f"Failed to acknowledge message {stream_id} after 'actual_job_data_for_processing' was invalid: {e_ack}")
                    continue

                if len(active_tasks) >= max_concurrent:
                    await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    active_tasks = {t for t in active_tasks if not t.done()}

                task = asyncio.create_task(
                    process_api_job(
                        job_data=actual_job_data_for_processing,
                        consumer_name=consumer_name,
                        group_name=message_broker.consumer_group_name,
                        message_broker=message_broker
                    )
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            if shutdown_event.is_set(): break

        except asyncio.CancelledError:
             _logger.debug("API job consumption loop cancelled")
             break
        except Exception as e:
            _logger.exception(f"Error in API job consumption loop: {e}")
            await asyncio.sleep(sleep_interval * 2)

    if active_tasks:
        _logger.debug(f"Waiting for {len(active_tasks)} remaining API job tasks to complete...")
        try:
            await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=30.0)
            _logger.debug("Remaining API job tasks finished.")
        except asyncio.TimeoutError:
            _logger.warning("Timeout waiting for remaining API job tasks during shutdown.")
        except Exception as e:
             _logger.exception(f"Error during API job task cleanup: {e}")

    _logger.debug("API job consumption loop finished.")


async def run_worker():
    """Run the API job worker with graceful shutdown handling."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sys.platform != 'win32':
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_async_shutdown(s)))
        else:
            signal.signal(sig, handle_shutdown)

    consumer_name = f"api-worker-{uuid.uuid4().hex[:8]}"
    _logger.debug(f"Starting API Job worker (PID: {os.getpid()}, Consumer: {consumer_name})...")

    await preload_components()

    await run_api_job_consumer(consumer_name)

async def handle_async_shutdown(sig):
     """Async compatible shutdown handler"""
     if not shutdown_event.is_set():
         _logger.warning(f"Received signal {sig}, initiating async shutdown...")
         shutdown_event.set()

def main():
    """Entry point for the worker process."""
    log_level = getattr(logging, app_config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    _logger.debug(f"Logging configured with level: {logging.getLevelName(log_level)}")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    _logger.info("Starting API Job Queue Worker")

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
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError as e:
             if "Cannot run the event loop while another loop is running" not in str(e): raise
        finally:
            if loop.is_running(): loop.stop()
            if not loop.is_closed(): loop.close()
            _logger.info("Event loop closed.")

    _logger.debug("API Job Queue Worker shutdown complete.")

if __name__ == "__main__":
    main()