import pytest
import asyncio
import signal
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming the module is viralStoryGenerator.src.queue_worker
from viralStoryGenerator.src import queue_worker as queue_worker_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For type hinting and spec for mocks

# --- Mocks & Fixtures ---

@pytest.fixture
def mock_app_config_qworker(monkeypatch):
    """Fixture to provide a mock app_config for queue_worker tests."""
    monkeypatch.setattr(app_config.redis, 'HOST', "mock_redis_host")
    monkeypatch.setattr(app_config.redis, 'PORT', 6379)
    monkeypatch.setattr(app_config.redis, 'QUEUE_NAME', "test_job_queue") # Assuming this is the primary queue name used
    monkeypatch.setattr(app_config.redis, 'WORKER_BATCH_SIZE', 5)
    monkeypatch.setattr(app_config.redis, 'WORKER_SLEEP_INTERVAL', 0.01) # Short sleep for tests
    monkeypatch.setattr(app_config.redis, 'WORKER_MAX_CONCURRENT', 3)
    monkeypatch.setattr(app_config.worker, 'DEFAULT_JOB_TYPE', "default_job_type_from_config") # If used
    return app_config

@pytest.fixture
def mock_broker_instance_qworker():
    """Provides a fresh MagicMock for RedisMessageBroker for each test."""
    broker = MagicMock(spec=RedisMessageBroker)
    broker.initialize = AsyncMock()
    broker.consume_messages = AsyncMock()
    broker.acknowledge_message = AsyncMock()
    broker.track_job_progress = AsyncMock() # For tracking failures
    broker.close = AsyncMock()
    return broker

@pytest.fixture
def mock_process_api_job_func():
    """Provides a fresh AsyncMock for process_api_job for each test."""
    return AsyncMock()

@pytest.fixture(autouse=True)
def reset_queue_worker_globals(monkeypatch):
    """Reset global broker instance in queue_worker.py before each test if it uses one."""
    if hasattr(queue_worker_module, '_message_broker'): # Check if it uses a global
        monkeypatch.setattr(queue_worker_module, '_message_broker', None)
    # Reset shutdown event if it's global in queue_worker
    if hasattr(queue_worker_module, 'shutdown_event'):
         queue_worker_module.shutdown_event.clear()


# --- Tests for process_single_api_job ---

# Scenario 1.1: Successful job processing
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker._logger') # Patch logger in queue_worker.py
async def test_process_single_api_job_successful(
    mock_logger, mock_broker_instance_qworker, mock_process_api_job_func, mock_app_config_qworker
):
    message_id = "msg_success_001"
    job_id = "job_success_001"
    job_type = "story_generation"
    job_data_raw = { # Simulate raw data from Redis (bytes)
        b"job_id": job_id.encode('utf-8'),
        b"job_type": job_type.encode('utf-8'),
        b"topic": b"A great success story"
    }
    # Decoded job data as expected by process_api_job
    expected_decoded_job_data = {
        "job_id": job_id,
        "job_type": job_type,
        "topic": "A great success story"
    }

    mock_process_api_job_func.return_value = True # Simulate successful processing

    await queue_worker_module.process_single_api_job(
        mock_broker_instance_qworker, message_id, job_data_raw, "test_group", "test_consumer"
    )

    # Verify process_api_job call
    mock_process_api_job_func.assert_called_once_with(
        job_id, job_type, expected_decoded_job_data, mock_broker_instance_qworker, "test_consumer"
    )
    
    # Verify acknowledge_message call
    mock_broker_instance_qworker.acknowledge_message.assert_called_once_with("test_group", message_id)
    
    # Verify logging
    mock_logger.info.assert_any_call(
        f"Consumer test_consumer: Successfully processed job {job_id} (message {message_id})."
    )
    # Ensure track_job_progress was NOT called for success (unless it's also used for success tracking)
    mock_broker_instance_qworker.track_job_progress.assert_not_called()


# Scenario 1.2: process_api_job raises an exception
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker._logger')
async def test_process_single_api_job_process_exception(
    mock_logger, mock_broker_instance_qworker, mock_process_api_job_func, mock_app_config_qworker
):
    message_id = "msg_exception_002"
    job_id = "job_exception_002"
    job_type = "video_rendering"
    job_data_raw = {b"job_id": job_id.encode('utf-8'), b"job_type": job_type.encode('utf-8')}
    expected_decoded_job_data = {"job_id": job_id, "job_type": job_type}
    
    exception_message = "Simulated processing error"
    mock_process_api_job_func.side_effect = Exception(exception_message)

    await queue_worker_module.process_single_api_job(
        mock_broker_instance_qworker, message_id, job_data_raw, "test_group", "test_consumer"
    )

    mock_process_api_job_func.assert_called_once_with(
        job_id, job_type, expected_decoded_job_data, mock_broker_instance_qworker, "test_consumer"
    )
    
    # Verify track_job_progress was called with "failed" status
    mock_broker_instance_qworker.track_job_progress.assert_called_once()
    args, _ = mock_broker_instance_qworker.track_job_progress.call_args
    assert args[0] == job_id
    status_update = args[1]
    assert status_update["status"] == "failed"
    assert exception_message in status_update["error_message"]
    assert args[2] is True # publish_to_live should be True

    # Verify acknowledge_message is still called
    mock_broker_instance_qworker.acknowledge_message.assert_called_once_with("test_group", message_id)
    
    mock_logger.error.assert_any_call(
        f"Consumer test_consumer: Exception processing job {job_id} (message {message_id}): {exception_message}"
    )


# Scenario 1.3: Error during acknowledge_message
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker._logger')
async def test_process_single_api_job_ack_exception(
    mock_logger, mock_broker_instance_qworker, mock_process_api_job_func, mock_app_config_qworker
):
    message_id = "msg_ack_fail_003"
    job_id = "job_ack_fail_003"
    job_type = "cleanup_task"
    job_data_raw = {b"job_id": job_id.encode('utf-8'), b"job_type": job_type.encode('utf-8')}
    
    mock_process_api_job_func.return_value = True # Processing succeeds
    
    ack_exception_message = "Simulated ACK error"
    mock_broker_instance_qworker.acknowledge_message.side_effect = Exception(ack_exception_message)

    await queue_worker_module.process_single_api_job(
        mock_broker_instance_qworker, message_id, job_data_raw, "test_group", "test_consumer"
    )

    mock_process_api_job_func.assert_called_once() # Processing was attempted
    mock_broker_instance_qworker.acknowledge_message.assert_called_once_with("test_group", message_id)
    
    mock_logger.error.assert_any_call(
        f"Consumer test_consumer: Failed to acknowledge message {message_id} for job {job_id}. Error: {ack_exception_message}"
    )
    # track_job_progress should not be called for an ACK failure if processing itself was successful
    mock_broker_instance_qworker.track_job_progress.assert_not_called()


# --- Tests for run_api_job_consumer ---

# Scenario 2.1: Consumes and processes a message
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker.get_message_broker')
@patch('viralStoryGenerator.src.queue_worker.process_single_api_job', new_callable=AsyncMock) # Mock the async function directly
@patch('viralStoryGenerator.src.queue_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock) # To control loop delays
async def test_run_api_job_consumer_consumes_and_processes(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_qworker, mock_app_config_qworker # Use fixtures
):
    # Setup mocks
    mock_get_broker.return_value = mock_broker_instance_qworker
    
    message_id = "msg_consume_001"
    job_data_raw = {b"job_id": b"job_consume_001"} # Minimal raw data
    
    # consume_messages will return one message, then empty to allow shutdown
    consume_call_count = 0
    async def consume_side_effect(batch_size, block_ms=None): # block_ms might be used
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1:
            return [(message_id, job_data_raw)]
        # After the first batch, signal shutdown to stop the loop
        queue_worker_module.shutdown_event.set()
        return []
    mock_broker_instance_qworker.consume_messages.side_effect = consume_side_effect
    
    # process_single_api_job is mocked directly
    mock_process_single_job.return_value = None # Doesn't matter for this test

    # Ensure shutdown_event is clear at the start
    queue_worker_module.shutdown_event.clear()

    consumer_name = "test_consumer_01"
    await queue_worker_module.run_api_job_consumer(consumer_name)

    # Verifications
    mock_broker_instance_qworker.initialize.assert_called_once()
    assert mock_broker_instance_qworker.consume_messages.call_count >= 1 # Called at least once
    
    # process_single_api_job should have been called with the consumed message
    mock_process_single_job.assert_called_once_with(
        mock_broker_instance_qworker,
        message_id,
        job_data_raw,
        mock_app_config_qworker.redis.QUEUE_NAME + "_group", # Default group name construction
        consumer_name
    )
    
    # asyncio.sleep should not have been called if a message was processed immediately
    # unless the loop structure forces it. Given the loop:
    # if not messages: await asyncio.sleep(...)
    # it shouldn't be called here after the first successful consumption.
    # However, the loop runs again after processing, gets an empty batch, then shutdown is set.
    # This means after the empty batch, sleep *would* be called if not for shutdown.
    # The current consume_side_effect sets shutdown on the second call (which returns empty).
    # So, the loop might run: 1. Get message, process. 2. Get empty, shutdown. No sleep.
    # Or: 1. Get message, process. 2. Get empty. 3. Loop check shutdown (true), exit. No sleep.
    # Let's check the log instead for "No messages..." which implies it tried to sleep or would have.
    
    # If the loop structure is `while not shutdown: messages = consume(); if messages: process else: sleep`,
    # then after processing, it loops, consumes (gets empty), then if shutdown is set there, it exits.
    # If shutdown is set by consume_messages itself when returning empty, then sleep is not hit.
    
    # Based on the provided code structure for run_api_job_consumer,
    # it processes tasks in asyncio.gather.
    # The main loop `while not shutdown_event.is_set()` calls `consume_messages`.
    # If `messages` is empty, it logs and sleeps.
    # If `messages` has items, it creates tasks.
    # Our side effect for consume_messages returns a message first, then sets shutdown and returns empty.
    # So:
    # 1. Loop starts. consume_messages -> returns 1 message.
    # 2. Task for process_single_api_job is created.
    # 3. Loop continues. consume_messages -> sets shutdown, returns []. (consume_messages call count is 2)
    # 4. `if not messages:` is true. Logs "No messages". `await asyncio.sleep` is called.
    # 5. Loop continues. `shutdown_event.is_set()` is true. Loop exits.
    # So, sleep *should* be called once.
    if consume_call_count > 1: # If consume_messages was called more than once (meaning it returned empty at least once before shutdown)
        # Check if the last call to consume_messages returned an empty list
        # This is a bit indirect. A better way might be to check logger.debug("No messages...")
        # For now, let's assume if consume_call_count > 1, the second call returned [] and set shutdown.
        # The sleep happens if messages is empty AND shutdown is NOT set.
        # In this specific side_effect, shutdown is set when messages are empty.
        # So, the condition `if not messages:` is true, then `if shutdown_event.is_set(): break` is true.
        # Thus, sleep is NOT called.
        mock_asyncio_sleep.assert_not_called()
    
    mock_broker_instance_qworker.close.assert_called_once() # Called at the end of run_api_job_consumer


# Scenario 2.2: Handles empty message batch
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker.get_message_broker')
@patch('viralStoryGenerator.src.queue_worker.process_single_api_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.queue_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_run_api_job_consumer_empty_batch_then_shutdown(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_qworker, mock_app_config_qworker
):
    mock_get_broker.return_value = mock_broker_instance_qworker
    
    consume_call_count = 0
    async def consume_empty_side_effect(batch_size, block_ms=None):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count >= 2: # Return empty a few times then shutdown
            queue_worker_module.shutdown_event.set()
        return [] # Always empty
    mock_broker_instance_qworker.consume_messages.side_effect = consume_empty_side_effect
    
    queue_worker_module.shutdown_event.clear()
    consumer_name = "test_consumer_empty"
    await queue_worker_module.run_api_job_consumer(consumer_name)

    assert mock_broker_instance_qworker.consume_messages.call_count >= 1
    mock_process_single_job.assert_not_called() # No messages to process
    # asyncio.sleep should be called because messages were empty and shutdown was not initially set
    mock_asyncio_sleep.assert_any_call(mock_app_config_qworker.redis.WORKER_SLEEP_INTERVAL)
    mock_logger.debug.assert_any_call(f"Consumer {consumer_name}: No messages received, sleeping for {mock_app_config_qworker.redis.WORKER_SLEEP_INTERVAL}s...")
    mock_broker_instance_qworker.close.assert_called_once()


# Scenario 2.3: Concurrency management (WORKER_MAX_CONCURRENT)
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker.get_message_broker')
@patch('viralStoryGenerator.src.queue_worker.process_single_api_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.queue_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock) # Mock sleep to control loop timing
async def test_run_api_job_consumer_concurrency_limit(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_qworker, mock_app_config_qworker, monkeypatch
):
    max_concurrent = 2
    monkeypatch.setattr(mock_app_config_qworker.redis, 'WORKER_MAX_CONCURRENT', max_concurrent)
    
    mock_get_broker.return_value = mock_broker_instance_qworker

    # Create asyncio Events to control when process_single_api_job "completes"
    job_events = [asyncio.Event() for _ in range(max_concurrent + 1)]
    
    processed_job_ids = []
    async def process_job_side_effect(broker, msg_id, data, group, consumer):
        job_num = len(processed_job_ids)
        processed_job_ids.append(data[b'job_id'].decode())
        mock_logger.info(f"Started processing job_num {job_num}, job_id {data[b'job_id'].decode()}")
        await job_events[job_num].wait() # Wait until event is set
        mock_logger.info(f"Finished processing job_num {job_num}, job_id {data[b'job_id'].decode()}")
        return True # Simulate success
    mock_process_single_job.side_effect = process_job_side_effect

    messages_to_consume = [
        ("msg1", {b"job_id": b"job1"}),
        ("msg2", {b"job_id": b"job2"}),
        ("msg3", {b"job_id": b"job3"}), # This one should wait
    ]

    consume_state = {"consumed_all": False}
    async def consume_concurrent_side_effect(batch_size, block_ms=None):
        if not consume_state["consumed_all"]:
            consume_state["consumed_all"] = True
            return messages_to_consume
        # After first batch, keep returning empty until shutdown
        await asyncio.sleep(0.001) # Allow other tasks to run
        queue_worker_module.shutdown_event.set() # Then shutdown
        return []
    mock_broker_instance_qworker.consume_messages.side_effect = consume_concurrent_side_effect
    
    queue_worker_module.shutdown_event.clear()
    worker_task = asyncio.create_task(queue_worker_module.run_api_job_consumer("test_consumer_concurrent"))

    await asyncio.sleep(0.05) # Give time for worker to start and pick up initial tasks

    # At this point, max_concurrent jobs should have started processing
    assert mock_process_single_job.call_count == max_concurrent
    assert len(processed_job_ids) == max_concurrent
    assert "job1" in processed_job_ids
    assert "job2" in processed_job_ids
    
    # The third job should not have started yet
    assert "job3" not in processed_job_ids

    # Allow one job to complete
    job_events[0].set() 
    await asyncio.sleep(0.05) # Give time for the next job to be picked up

    # Now the third job should have started
    assert mock_process_single_job.call_count == max_concurrent + 1
    assert len(processed_job_ids) == max_concurrent + 1
    assert "job3" in processed_job_ids

    # Allow remaining jobs to complete
    for i in range(1, max_concurrent + 1):
        job_events[i].set()
    
    await worker_task # Wait for the worker to finish (it will due to shutdown_event)

    mock_broker_instance_qworker.close.assert_called_once()


# Scenario 2.4: Graceful shutdown (covered by previous tests using shutdown_event.set())
# Adding a specific test to ensure active tasks are awaited if applicable (depends on asyncio.gather/wait usage)
# The current structure uses asyncio.gather(tasks) which implicitly awaits.

# Scenario 2.5: Invalid message structure from consume_messages
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.queue_worker.get_message_broker')
@patch('viralStoryGenerator.src.queue_worker.process_single_api_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.queue_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_run_api_job_consumer_invalid_message_structure(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_qworker, mock_app_config_qworker
):
    mock_get_broker.return_value = mock_broker_instance_qworker
    
    message_id_invalid = "msg_invalid_struct_001"
    # job_data_raw is not a dict, or missing bytes keys, or values not bytes
    # The worker's decode_job_data expects bytes keys/values.
    # Here, process_single_api_job expects raw_job_data to be a dict[bytes, bytes]
    # Let's test with job_data_raw not being a dict.
    invalid_job_data_raw = "This is not a dict" 
    
    consume_call_count = 0
    async def consume_invalid_side_effect(batch_size, block_ms=None):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1:
            return [(message_id_invalid, invalid_job_data_raw)]
        queue_worker_module.shutdown_event.set()
        return []
    mock_broker_instance_qworker.consume_messages.side_effect = consume_invalid_side_effect
    
    queue_worker_module.shutdown_event.clear()
    consumer_name = "test_consumer_invalid_msg"
    await queue_worker_module.run_api_job_consumer(consumer_name)

    # process_single_api_job should have been called, and it should handle the error internally.
    mock_process_single_job.assert_called_once_with(
        mock_broker_instance_qworker,
        message_id_invalid,
        invalid_job_data_raw, # Passes the raw data
        mock_app_config_qworker.redis.QUEUE_NAME + "_group",
        consumer_name
    )
    # The error logging for invalid structure (e.g. not a dict, missing keys after decode)
    # would happen inside process_single_api_job.
    # We need to check that process_single_api_job was called, and its internal logic
    # (tested separately) handles logging and acking.
    # Here, we just ensure run_api_job_consumer correctly dispatches it.
    
    mock_broker_instance_qworker.close.assert_called_once()

# --- Tests for preload_components and get_message_broker (queue_worker context) ---

@patch('viralStoryGenerator.src.queue_worker.RedisMessageBroker') # Patch where it's imported/used in queue_worker
@patch('viralStoryGenerator.src.queue_worker._logger')
def test_qw_get_message_broker_initializes_and_returns(
    mock_logger, MockRedisMessageBroker, mock_app_config_qworker, monkeypatch
):
    # Ensure global _message_broker is None at the start of this specific test
    monkeypatch.setattr(queue_worker_module, '_message_broker', None)
    
    mock_broker_internal_instance = MagicMock(spec=RedisMessageBroker)
    mock_broker_internal_instance.initialize = AsyncMock()
    MockRedisMessageBroker.return_value = mock_broker_internal_instance

    # get_message_broker in queue_worker is synchronous
    broker = queue_worker_module.get_message_broker()

    assert broker is mock_broker_internal_instance
    expected_redis_url = f"redis://{mock_app_config_qworker.redis.HOST}:{mock_app_config_qworker.redis.PORT}"
    MockRedisMessageBroker.assert_called_once_with(
        redis_url=expected_redis_url,
        stream_name=mock_app_config_qworker.redis.QUEUE_NAME, # Uses QUEUE_NAME
        consumer_group_name=mock_app_config_qworker.redis.QUEUE_NAME + "_group" # Default group name
    )
    # In queue_worker's get_message_broker, initialize() is NOT called. It's called in run_api_job_consumer.
    mock_broker_internal_instance.initialize.assert_not_called() 
    mock_logger.info.assert_any_call(f"Message broker initialized for stream: {mock_app_config_qworker.redis.QUEUE_NAME}")


def test_qw_get_message_broker_returns_existing(mock_app_config_qworker, monkeypatch):
    # Pre-set the global broker
    mock_existing_broker_global = MagicMock(spec=RedisMessageBroker)
    monkeypatch.setattr(queue_worker_module, '_message_broker', mock_existing_broker_global)
    
    with patch('viralStoryGenerator.src.queue_worker.RedisMessageBroker') as MockRedisMessageBroker_new:
        broker = queue_worker_module.get_message_broker()
        assert broker is mock_existing_broker_global
        MockRedisMessageBroker_new.assert_not_called() # Should not create new if one exists


@patch('viralStoryGenerator.src.queue_worker.get_message_broker') # Mocks the factory
@patch('viralStoryGenerator.src.queue_worker._logger')
async def test_qw_preload_components(mock_logger, mock_get_broker_func, mock_broker_instance_qworker):
    # get_message_broker should be called by preload_components
    mock_get_broker_func.return_value = mock_broker_instance_qworker
    
    # preload_components is async
    await queue_worker_module.preload_components()

    mock_get_broker_func.assert_called_once()
    mock_broker_instance_qworker.initialize.assert_called_once() # Initialize is called here
    mock_broker_instance_qworker.ensure_stream_exists.assert_called_once()
    mock_broker_instance_qworker.create_consumer_group.assert_called_once() # With default group name
    mock_logger.info.assert_any_call("Preloaded Redis components (stream, group).")


@patch('viralStoryGenerator.src.queue_worker.get_message_broker')
@patch('viralStoryGenerator.src.queue_worker._logger')
async def test_qw_preload_components_group_exists(mock_logger, mock_get_broker_func, mock_broker_instance_qworker):
    mock_get_broker_func.return_value = mock_broker_instance_qworker
    mock_broker_instance_qworker.create_consumer_group.side_effect = Exception("BUSYGROUP")

    await queue_worker_module.preload_components() # Should handle BUSYGROUP

    mock_broker_instance_qworker.create_consumer_group.assert_called_once()
    mock_logger.info.assert_any_call("Consumer group already exists or error creating it (BUSYGROUP expected if exists).")


# --- Tests for run_worker_main_loop and main (queue_worker context) ---

@patch('viralStoryGenerator.src.queue_worker.preload_components', new_callable=AsyncMock)
@patch('asyncio.gather') # To control the behavior of concurrent consumers
@patch('viralStoryGenerator.src.queue_worker.run_api_job_consumer', new_callable=AsyncMock) # Mock the actual consumer coroutine
@patch('viralStoryGenerator.src.queue_worker.signal.signal')
@patch('viralStoryGenerator.src.queue_worker._logger')
async def test_qw_run_worker_main_loop_orchestration(
    mock_logger, mock_signal_signal, mock_run_consumer_coro, 
    mock_asyncio_gather, mock_preload, mock_app_config_qworker, monkeypatch
):
    # Configure app_config for number of consumers
    num_consumers = 2
    monkeypatch.setattr(mock_app_config_qworker.redis, 'WORKER_MAX_CONCURRENT', num_consumers)
    
    # asyncio.gather will just be checked for call, not awaited fully in this unit test
    # unless we make its side_effect more complex.
    # For simplicity, let it return normally or mock its return if needed.
    mock_asyncio_gather.return_value = None 

    # Ensure shutdown_event is clear initially
    queue_worker_module.shutdown_event.clear()
    
    # To make the main loop exit for the test, we can set the shutdown_event
    # after a short delay, simulating an external stop or a short run.
    # run_worker_main_loop has `while not shutdown_event.is_set(): await asyncio.sleep(1)`
    # We need to break this loop.
    
    # Let's have asyncio.gather raise KeyboardInterrupt to simulate user stopping,
    # as this is a common way the main loop is expected to terminate in examples.
    # Or, more directly, patch asyncio.sleep within the main loop.
    
    with patch('asyncio.sleep') as mock_main_loop_sleep:
        # Make the main loop's sleep set the shutdown event to exit after one iteration
        async def sleep_then_shutdown(duration):
            queue_worker_module.shutdown_event.set()
            await asyncio.sleep(0) # Yield to allow the event to be processed
            return None
        mock_main_loop_sleep.side_effect = sleep_then_shutdown

        await queue_worker_module.run_worker_main_loop()

    mock_preload.assert_called_once()
    
    # Verify consumer tasks were created and passed to asyncio.gather
    assert mock_run_consumer_coro.call_count == num_consumers
    consumer_ids_generated = set()
    for call_arg in mock_run_consumer_coro.call_args_list:
        args, _ = call_arg
        consumer_id = args[0] # First arg to run_api_job_consumer is consumer_name
        assert f"{mock_app_config_qworker.redis.QUEUE_NAME}_consumer_" in consumer_id
        consumer_ids_generated.add(consumer_id)
    assert len(consumer_ids_generated) == num_consumers # Ensure unique consumer IDs

    mock_asyncio_gather.assert_called_once()
    # Check that the tasks passed to gather are the ones from run_api_job_consumer calls
    # This is a bit complex to assert directly on the coroutine objects.
    # Call count of run_api_job_consumer is a good proxy.
    
    mock_logger.info.assert_any_call(f"Starting {num_consumers} API job consumers...")
    mock_logger.info.assert_any_call("All API job consumers have completed.")
    mock_main_loop_sleep.assert_called_once_with(1) # Main loop sleep


@patch('viralStoryGenerator.src.queue_worker.run_worker_main_loop', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.queue_worker.signal.signal') # To verify signal setup
@patch('viralStoryGenerator.src.queue_worker._logger')
def test_qw_main_function_setup_and_run(
    mock_logger, mock_signal_signal, mock_run_main_loop, mock_app_config_qworker
):
    # Simulate run_worker_main_loop raising KeyboardInterrupt to test graceful shutdown logging
    mock_run_main_loop.side_effect = KeyboardInterrupt("Test interrupt")

    # Call main()
    queue_worker_module.main()

    # Verify signal handlers are set
    mock_signal_signal.assert_any_call(signal.SIGINT, queue_worker_module.shutdown_handler)
    mock_signal_signal.assert_any_call(signal.SIGTERM, queue_worker_module.shutdown_handler)
    
    # Verify run_worker_main_loop was called
    mock_run_main_loop.assert_called_once()
    
    # Verify logging
    mock_logger.info.assert_any_call("ViralStoryGenerator Queue Worker starting...")
    mock_logger.info.assert_any_call("Queue worker shut down by KeyboardInterrupt.")


def test_qw_shutdown_handler(mock_app_config_qworker): # No mocks needed beyond what fixtures provide
    queue_worker_module.shutdown_event.clear()
    assert not queue_worker_module.shutdown_event.is_set()
    
    # Call the handler (as if a signal was received)
    queue_worker_module.shutdown_handler(signal.SIGINT, None) 
    
    assert queue_worker_module.shutdown_event.is_set()
    # Logger call is inside the handler, would need to patch _logger there if checking.
    # For this test, just ensuring the event is set is primary.
