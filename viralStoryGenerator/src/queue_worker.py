import logging
import asyncio
import os
import signal
import sys
import uuid
import json

from typing import Dict, Any, List, Optional, Tuple

from ..utils.config import config as app_config
from ..utils.redis_manager import RedisMessageBroker
from ..utils.api_job_processor import process_api_job
from viralStoryGenerator.src.api import API_QUEUE_NAME

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
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=API_QUEUE_NAME)
    await _message_broker.initialize()

    _logger.debug(f"API Job worker components initialized for stream '{API_QUEUE_NAME}'")


def get_message_broker() -> RedisMessageBroker:
    """Get the pre-initialized message broker or create a new one if needed. Ensures initialization."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    _logger.warning("Message broker accessed before preload, initializing now.")
    redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
    _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=API_QUEUE_NAME)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if not loop.is_running():
        loop.run_until_complete(_message_broker.initialize())
    return _message_broker


# This function (process_single_api_job) seems to be an alternative processing path
# or an older version. The main logic being debugged is in run_api_job_consumer.
# For clarity, I'm leaving it as is but focusing fixes on run_api_job_consumer.
# async def process_single_api_job(job_data_raw: Dict[bytes, bytes], message_id: str, group_name: str, consumer_name: str):
#     """Processes a single job received from the api_jobs stream."""
#     job_id = None
#     message_broker = get_message_broker()
#     try:
#         job_data_decoded: Dict[str, Any] = {}
#         for k, v in job_data_raw.items():
#             key_decoded = k.decode('utf-8') if isinstance(k, bytes) else str(k)
#             value_decoded = v.decode('utf-8') if isinstance(v, bytes) else v
#             job_data_decoded[key_decoded] = value_decoded

#         payload_json_str = job_data_decoded.get("payload")

#         if not payload_json_str or not isinstance(payload_json_str, str):
#             _logger.error(f"Message {message_id} is missing 'payload' field or it's not a string. Fields: {job_data_decoded}")
#             # Acknowledge malformed message if possible
#             await message_broker.acknowledge_message(message_id)
#             return

#         try:
#             job_data_from_payload = json.loads(payload_json_str)
#         except json.JSONDecodeError as e:
#             _logger.error(f"Failed to parse JSON from 'payload' field for message {message_id}. Error: {e}. Payload string: '{payload_json_str}'")
#             await message_broker.acknowledge_message(message_id)
#             return

#         job_id_any = job_data_from_payload.get("job_id")
#         job_id = str(job_id_any) if job_id_any is not None else str(uuid.uuid4())
#         if job_id_any is None:
#             _logger.warning(f"Job message {message_id} (from parsed payload) missing job_id, assigned: {job_id}")
#             job_data_from_payload["job_id"] = job_id

#         inner_data = job_data_from_payload.get("data")
#         if not isinstance(inner_data, dict):
#             _logger.error(f"Message {message_id}: 'data' field in parsed payload is missing or not a dict. Parsed payload: {job_data_from_payload}")
#             await message_broker.acknowledge_message(message_id)
#             return

#         job_type = inner_data.get("job_type")
#         # Ensure job_id is in inner_data for process_api_job
#         if "job_id" not in inner_data:
#             inner_data["job_id"] = job_id


#         _logger.debug(f"Processing API job {job_id} (Type: {job_type or 'Unknown'}). Message ID: {message_id} via process_single_api_job")

#         await process_api_job(inner_data, consumer_name, group_name, message_broker) # Removed job_id_override

#     except Exception as e:
#         _logger.exception(f"Error processing API job {job_id or message_id} in process_single_api_job: {e}")
#         if job_id and message_broker:
#             try:
#                 await message_broker.track_job_progress(job_id, "failed", {"error": f"Worker failed (process_single_api_job): {str(e)}"})
#             except Exception as track_e:
#                 _logger.error(f"Failed to update status to failed for job {job_id}: {track_e}")
#     finally:
#         if message_broker and message_id: # Ensure message_id is valid before ack
#             try:
#                 await message_broker.acknowledge_message(message_id)
#                 _logger.debug(f"Acknowledged message {message_id} for job {job_id or 'unknown'} (in process_single_api_job)")
#             except Exception as ack_e:
#                  _logger.error(f"Failed to acknowledge message {message_id} in process_single_api_job: {ack_e}")

async def run_api_job_consumer(consumer_name: str):
    """Continuously consumes and processes jobs from the api_jobs stream."""
    message_broker = get_message_broker()
    batch_size = app_config.redis.WORKER_BATCH_SIZE
    sleep_interval = app_config.redis.WORKER_SLEEP_INTERVAL
    max_concurrent = app_config.redis.WORKER_MAX_CONCURRENT
    active_tasks = set()

    _logger.debug(f"API Job worker '{consumer_name}' starting consumption loop on '{API_QUEUE_NAME}'")
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

            # messages: List[Tuple[str, Dict[bytes, bytes]]]
            # message_id_from_stream, raw_redis_fields_bytes
            messages = await message_broker.consume_messages(
                count=min(available_slots, batch_size),
                block_ms=2000
            )

            if not messages:
                await asyncio.sleep(sleep_interval)
                continue

            _logger.debug(f"Received {len(messages)} new API job message(s).")
            for message_id_from_stream, raw_redis_fields_bytes in messages:
                if shutdown_event.is_set(): break

                if not message_id_from_stream or not raw_redis_fields_bytes or not isinstance(raw_redis_fields_bytes, dict):
                    _logger.error(f"Invalid message structure received. Stream ID: {message_id_from_stream}, Raw Fields: {raw_redis_fields_bytes}. Skipping.")
                    if message_id_from_stream:
                        try:
                            await message_broker.acknowledge_message(message_id_from_stream)
                        except Exception as ack_e:
                            _logger.error(f"Failed to acknowledge malformed message {message_id_from_stream}: {ack_e}")
                    continue

                # Decode all keys and values from bytes to string, similar to process_single_api_job
                decoded_fields: Dict[str, str] = {}
                try:
                    for k_bytes, v_bytes in raw_redis_fields_bytes.items():
                        key_str = k_bytes.decode('utf-8') if isinstance(k_bytes, bytes) else str(k_bytes)
                        val_str = v_bytes.decode('utf-8') if isinstance(v_bytes, bytes) else str(v_bytes)
                        decoded_fields[key_str] = val_str
                except UnicodeDecodeError as ude:
                    _logger.error(f"Message {message_id_from_stream} contained non-UTF8 field names/values: {ude}. Raw: {raw_redis_fields_bytes}. Skipping.")
                    await message_broker.acknowledge_message(message_id_from_stream)
                    continue

                payload_json_str = decoded_fields.get("payload")

                job_data_for_processing: Optional[Dict[str, Any]] = None
                job_type: Optional[str] = None
                job_id_from_payload: Optional[str] = None

                if not payload_json_str: # Already a string due to decoding above
                    _logger.error(f"Message {message_id_from_stream} is missing 'payload' field after decoding. Decoded Fields: {decoded_fields}")
                else:
                    try:
                        # This is the dict like: {"job_id": "...", "data": {"id": ..., "job_type": ...}}
                        parsed_outer_payload = json.loads(payload_json_str)

                        if not isinstance(parsed_outer_payload, dict):
                            _logger.error(f"Message {message_id_from_stream}: Parsed JSON payload is not a dict. Payload string: '{payload_json_str}'")
                        else:
                            job_id_from_payload = parsed_outer_payload.get("job_id")
                            job_data_for_processing = parsed_outer_payload.get("data")

                            if not isinstance(job_data_for_processing, dict):
                                _logger.error(f"Message {message_id_from_stream}: 'data' field in parsed JSON payload is missing or not a dict. Parsed outer payload: {parsed_outer_payload}")
                                job_data_for_processing = None
                            else:
                                job_type = job_data_for_processing.get("job_type")
                                # Ensure job_id is part of the data passed to process_api_job
                                if "job_id" not in job_data_for_processing and job_id_from_payload:
                                    job_data_for_processing["job_id"] = job_id_from_payload
                                elif "job_id" not in job_data_for_processing:
                                    new_job_id = str(uuid.uuid4())
                                    job_data_for_processing["job_id"] = new_job_id
                                    _logger.warning(f"Message {message_id_from_stream} had no job_id in outer payload or inner data. Assigned new: {new_job_id} to job_data_for_processing.")
                                elif not job_data_for_processing.get("job_id") and job_id_from_payload : # job_id key exists but is None/empty
                                     job_data_for_processing["job_id"] = job_id_from_payload
                                     _logger.info(f"Populated empty job_id in job_data_for_processing with {job_id_from_payload} for message {message_id_from_stream}")


                    except json.JSONDecodeError as e:
                        _logger.error(f"Message {message_id_from_stream}: Could not parse JSON from payload string. Error: {e}. Payload string: '{payload_json_str}'")

                if job_type != "generate_story":
                    log_detail = f"job_type '{job_type}'" if job_type else "missing or invalid job_type"
                    _logger.warning(f"Message {message_id_from_stream} has {log_detail} or is not 'generate_story'. Skipping. Decoded fields: {decoded_fields}")
                    try:
                        await message_broker.acknowledge_message(message_id_from_stream)
                    except Exception as e_ack:
                        _logger.error(f"Failed to acknowledge message {message_id_from_stream} after skipping ({log_detail}): {e_ack}")
                    continue

                if not job_data_for_processing:
                    _logger.error(f"Skipping message {message_id_from_stream} due to invalid or missing 'data' field in payload after processing. Decoded fields: {decoded_fields}")
                    try:
                        await message_broker.acknowledge_message(message_id_from_stream)
                    except Exception as e_ack:
                        _logger.error(f"Failed to acknowledge message {message_id_from_stream} after invalid 'data': {e_ack}")
                    continue

                # Final check for job_id in the data to be processed
                if "job_id" not in job_data_for_processing or not job_data_for_processing.get("job_id"):
                    final_job_id = job_id_from_payload or str(uuid.uuid4())
                    job_data_for_processing["job_id"] = final_job_id
                    _logger.info(f"Ensured job_id '{final_job_id}' is in job_data_for_processing for message {message_id_from_stream} before task creation.")


                if len(active_tasks) >= max_concurrent:
                    _logger.debug(f"Max concurrent tasks ({max_concurrent}) reached before dispatching new task. Waiting for a slot...")
                    done_interim, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    _logger.debug(f"{len(done_interim)} task(s) completed, making space.")

                current_job_id_for_log = job_data_for_processing.get('job_id', 'ID_MISSING_IN_LOG')
                _logger.info(f"Dispatching task for job {current_job_id_for_log} (Type: {job_type}) from message {message_id_from_stream}")

                # Create a closure to capture message_id_from_stream for acknowledgement
                async def task_wrapper(p_job_data, p_consumer_name, p_group_name, p_broker, p_message_id_to_ack):
                    try:
                        await process_api_job(
                            job_data=p_job_data,
                            consumer_name=p_consumer_name,
                            group_name=p_group_name,
                            message_broker=p_broker
                        )
                    finally:
                        # Always acknowledge the message from the stream after processing (success or failure)
                        try:
                            await p_broker.acknowledge_message(p_message_id_to_ack)
                            _logger.debug(f"Acknowledged stream message {p_message_id_to_ack} for job {p_job_data.get('job_id', 'UNKNOWN')}")
                        except Exception as e_ack_task:
                            _logger.error(f"Failed to acknowledge message {p_message_id_to_ack} from task_wrapper: {e_ack_task}")

                task = asyncio.create_task(
                    task_wrapper(
                        p_job_data=job_data_for_processing.copy(), # Pass a copy to avoid issues if modified elsewhere
                        p_consumer_name=consumer_name,
                        p_group_name=message_broker.consumer_group_name,
                        p_broker=message_broker,
                        p_message_id_to_ack=message_id_from_stream
                    )
                )
                active_tasks.add(task)
                task.add_done_callback(lambda t: active_tasks.discard(t))


            if shutdown_event.is_set(): break

        except asyncio.CancelledError:
             _logger.debug("API job consumption loop cancelled")
             break
        except Exception as e:
            _logger.exception(f"Error in API job consumption loop: {e}")
            await asyncio.sleep(sleep_interval * 2) # Backoff on error to avoid rapid retries

    if active_tasks:
        _logger.debug(f"Waiting for {len(active_tasks)} remaining API job tasks to complete...")
        try:
            await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=app_config.scraper.WORKER_SHUTDOWN_TIMEOUT)
            _logger.debug("Remaining API job tasks finished.")
        except asyncio.TimeoutError:
            _logger.warning(f"Timeout waiting for {len(active_tasks)} remaining API job tasks during shutdown.")
            for task_to_cancel in list(active_tasks): # Iterate over a copy
                if not task_to_cancel.done():
                    task_to_cancel.cancel() # Attempt to cancel lingering tasks
        except Exception as e:
             _logger.exception(f"Error during API job task cleanup: {e}")

    _logger.debug("API job consumption loop finished.")


async def run_worker():
    """Run the API job worker with graceful shutdown handling."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sys.platform != 'win32': # signal.SIGHUP, signal.SIGQUIT are not on windows
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_async_shutdown(s)))
        else: # Fallback for Windows
            signal.signal(sig, handle_shutdown) # handle_shutdown sets the asyncio.Event

    consumer_name = f"api-worker-{uuid.uuid4().hex[:8]}"
    _logger.debug(f"Starting API Job worker (PID: {os.getpid()}, Consumer: {consumer_name})...")

    await preload_components() # Ensure broker is ready

    await run_api_job_consumer(consumer_name)

async def handle_async_shutdown(sig):
     """Async compatible shutdown handler"""
     if not shutdown_event.is_set():
         _logger.warning(f"Received signal {sig}, initiating async shutdown...")
         shutdown_event.set()

def main():
    """Entry point for the worker process."""
    log_level_str = app_config.LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        stream=sys.stdout,
        force=True
    )
    _logger.setLevel(log_level) # Ensure this module's logger also respects the level
    logging.getLogger('viralStoryGenerator').setLevel(log_level) # And other project loggers


    _logger.debug(f"Logging configured. Root effective level: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}, Module '{__name__}' effective level: {logging.getLevelName(_logger.getEffectiveLevel())}")


    if os.name == 'nt': # Required for Windows if using ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    _logger.info(f"Starting API Job Queue Worker (Version: {app_config.VERSION})")

    if not app_config.redis.ENABLED:
        _logger.error("Redis is disabled in configuration. Enable REDIS_ENABLED=True to use the queue worker.")
        sys.exit(1)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): # If a previous run closed it
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError: # No current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        _logger.info("API Job worker stopped by keyboard interrupt.")
        if not shutdown_event.is_set(): # Ensure event is set if KeyboardInterrupt is outside run_worker
            shutdown_event.set()
            # Re-run run_worker briefly to allow it to see the event and shutdown gracefully
            loop.run_until_complete(run_worker())
    except Exception as e:
        _logger.exception(f"API Job worker failed unexpectedly: {e}")
    finally:
        _logger.info("Shutting down API Job worker...")
        # Ensure shutdown_event is set, then run loop for cleanup tasks
        if not shutdown_event.is_set():
            shutdown_event.set()

        # Give run_worker a chance to finish processing active_tasks if it hasn't exited yet
        # This might be redundant if run_worker already completed due to shutdown_event
        # but ensures cleanup if shutdown was triggered externally.
        if not loop.is_closed():
            # Run pending tasks, including cleanup in run_api_job_consumer
            loop.run_until_complete(asyncio.sleep(0.1)) # Allow any final tasks to register/run

            # Shutdown async generators
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except RuntimeError as e: # pragma: no cover
                if "Cannot run the event loop while another loop is running" not in str(e) and "Event loop is closed" not in str(e):
                    _logger.error(f"Error during asyncgen shutdown: {e}")
            except Exception as e: # pragma: no cover
                 _logger.error(f"Unexpected error during asyncgen shutdown: {e}")
            finally: # pragma: no cover
                if loop.is_running():
                    loop.stop() # Request loop to stop
                if not loop.is_closed():
                    loop.close() # Close the loop
                    _logger.info("Event loop closed.")
        else: # pragma: no cover
            _logger.info("Event loop was already closed.")

    _logger.info("API Job Queue Worker shutdown complete.")

if __name__ == "__main__":
    main()