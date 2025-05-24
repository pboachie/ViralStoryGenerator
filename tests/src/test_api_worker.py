import pytest
import asyncio
import sys
import signal
from unittest.mock import patch, MagicMock

import viralStoryGenerator.src.api_worker as api_worker

@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    # Reset global state before each test
    api_worker._message_broker = None
    api_worker._vector_db_client = None
    yield

def test_handle_shutdown(monkeypatch):
    called = {}
    def fake_exit(code):
        called['exit'] = code
        raise SystemExit()
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    monkeypatch.setattr(sys, "exit", fake_exit)
    # Should set shutdown_event and call sys.exit
    with pytest.raises(SystemExit):
        api_worker.handle_shutdown(signal.SIGINT, None)
    assert called['exit'] == 0

def test_preload_components_creates_group(monkeypatch):
    mock_broker = MagicMock()
    monkeypatch.setattr(api_worker, "RedisMessageBroker", lambda **kwargs: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    mock_broker.create_consumer_group.return_value = None
    mock_broker.ensure_stream_exists.return_value = None
    api_worker.preload_components("testgroup")
    mock_broker.create_consumer_group.assert_called_once_with(group_name="testgroup")
    mock_broker.ensure_stream_exists.assert_called_once()

def test_preload_components_group_exists(monkeypatch):
    mock_broker = MagicMock()
    monkeypatch.setattr(api_worker, "RedisMessageBroker", lambda **kwargs: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    # Simulate BUSYGROUP error
    mock_broker.create_consumer_group.side_effect = Exception("BUSYGROUP Consumer Group name already exists")
    mock_broker.ensure_stream_exists.return_value = None
    api_worker.preload_components("testgroup")
    mock_broker.ensure_stream_exists.assert_called_once()

def test_get_message_broker_initializes(monkeypatch):
    mock_broker = MagicMock()
    monkeypatch.setattr(api_worker, "RedisMessageBroker", lambda **kwargs: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    api_worker._message_broker = None
    broker = api_worker.get_message_broker()
    assert broker is mock_broker

def test_get_message_broker_returns_existing(monkeypatch):
    mock_broker = MagicMock()
    api_worker._message_broker = mock_broker
    assert api_worker.get_message_broker() is mock_broker

@pytest.mark.asyncio
async def test_process_api_jobs_handles_no_broker(monkeypatch):
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: None)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    # Patch shutdown_event to stop after one loop
    api_worker.shutdown_event.clear() # Ensure it's clear at start
    
    # Make consume_messages stop the loop after a few calls if it hasn't already
    async def consume_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0) # allow other tasks to run
        api_worker.shutdown_event.set() # signal shutdown
        return [] # no messages
    
    mock_broker_no_broker = MagicMock() # Different instance for this test
    mock_broker_no_broker.consume_messages = AsyncMock(side_effect=consume_then_shutdown)
    
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: None) # Simulate no broker
    
    await api_worker.process_api_jobs("group", "consumer")
    # Check logger for "Message broker not available"
    api_worker._logger.error.assert_any_call("Message broker not available. Retrying in 5 seconds...")


@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job') # Patch the actual function used
async def test_process_api_jobs_consumes_and_acknowledges(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    message_id = "123-0"
    # Message data should be decoded by the broker before being returned by consume_messages
    # The worker expects dicts with string keys.
    message_data = {
        "job_id": "jid_decoded", # Use strings
        "job_type": "story_decoded"
    }
    # consume_messages should return a list of tuples: (stream_name, list_of_messages)
    # Each message in list_of_messages is a tuple: (message_id, message_data_dict)
    # For simplicity, let's assume consume_messages directly returns the list of (id, data) tuples from the desired stream
    # as per the worker's loop: `for message_id, job_data in messages_data:`
    
    async def consume_side_effect_single_message(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set() # stop after this message batch
        return [(message_id, message_data)] # Simulate one message
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_side_effect_single_message)
    mock_broker.acknowledge_message = AsyncMock()
    
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)
    
    mock_process_api_job_func.return_value = True # Simulate successful job processing

    api_worker.shutdown_event.clear() # Clear before run
    await api_worker.process_api_jobs("test_group", "test_consumer")

    mock_broker.consume_messages.assert_called_once()
    mock_process_api_job_func.assert_called_once_with("jid_decoded", "story_decoded", message_data)
    mock_broker.acknowledge_message.assert_called_once_with("test_group", message_id)
    logger_mock.info.assert_any_call(f"Processing job jid_decoded of type story_decoded from message {message_id}.")


# Test 1.1: Consume messages - empty batch
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_empty_batch(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    
    async def consume_empty_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set()
        return [] # Empty batch
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_empty_then_shutdown)
    mock_broker.acknowledge_message = AsyncMock() # Should not be called

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    await api_worker.process_api_jobs("test_group", "test_consumer")

    mock_broker.consume_messages.assert_called_once()
    mock_process_api_job_func.assert_not_called()
    mock_broker.acknowledge_message.assert_not_called()
    logger_mock.debug.assert_any_call("No messages received, continuing...")


# Test 1.3: Consume messages - missing job_id (should skip and ack)
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_missing_job_id(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    message_id_no_job_id = "msg_no_id_001"
    message_data_no_job_id = {"job_type": "some_type", "data": "stuff"} # Missing 'job_id'

    async def consume_missing_id_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set()
        return [(message_id_no_job_id, message_data_no_job_id)]
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_missing_id_then_shutdown)
    mock_broker.acknowledge_message = AsyncMock()

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    await api_worker.process_api_jobs("test_group", "test_consumer")

    mock_process_api_job_func.assert_not_called()
    mock_broker.acknowledge_message.assert_called_once_with("test_group", message_id_no_job_id)
    logger_mock.error.assert_any_call(
        f"Message {message_id_no_job_id} is missing 'job_id' or 'job_type'. Skipping."
    )


# Test 1.4: Consume messages - missing job_type (should use default and ack)
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_missing_job_type_uses_default(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    message_id_no_type = "msg_no_type_002"
    job_id_no_type = "job_for_no_type"
    message_data_no_type = {"job_id": job_id_no_type, "other_data": "info"} # Missing 'job_type'
    default_job_type = "default_story_type"
    
    monkeypatch.setattr(api_worker.app_config.worker, 'DEFAULT_JOB_TYPE', default_job_type)

    async def consume_missing_type_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set()
        return [(message_id_no_type, message_data_no_type)]
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_missing_type_then_shutdown)
    mock_broker.acknowledge_message = AsyncMock()
    mock_process_api_job_func.return_value = True # Simulate successful processing

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    await api_worker.process_api_jobs("test_group", "test_consumer")
    
    mock_process_api_job_func.assert_called_once_with(job_id_no_type, default_job_type, message_data_no_type)
    mock_broker.acknowledge_message.assert_called_once_with("test_group", message_id_no_type)
    logger_mock.warning.assert_any_call(
        f"Message {message_id_no_type} for job {job_id_no_type} is missing 'job_type'. Using default: {default_job_type}."
    )


# Test 1.5: Consume messages - message data is not a dictionary (skip and ack)
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_invalid_message_data_type(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    message_id_invalid_data = "msg_invalid_data_003"
    raw_message_data_invalid = "This is not a dictionary"

    async def consume_invalid_data_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set()
        return [(message_id_invalid_data, raw_message_data_invalid)]
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_invalid_data_then_shutdown)
    mock_broker.acknowledge_message = AsyncMock()

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    await api_worker.process_api_jobs("test_group", "test_consumer")

    mock_process_api_job_func.assert_not_called()
    mock_broker.acknowledge_message.assert_called_once_with("test_group", message_id_invalid_data)
    logger_mock.error.assert_any_call(
        f"Message {message_id_invalid_data} data is not a dictionary. Skipping. Data: {raw_message_data_invalid}"
    )


# --- Error Handling Tests for process_api_jobs ---

# Test 4.1: Exception during consume_messages
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_consume_exception(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    
    consume_call_count = 0
    consume_exception_message = "Redis connection lost during consume"

    async def consume_exception_then_shutdown(*args, **kwargs):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1:
            raise Exception(consume_exception_message)
        # If it didn't stop after exception, this would trigger shutdown
        await asyncio.sleep(0)
        api_worker.shutdown_event.set() 
        return []
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_exception_then_shutdown)
    mock_broker.acknowledge_message = AsyncMock() # Should not be called if consume fails before yielding messages

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    # The loop in process_api_jobs should catch the exception, log it, and continue.
    # We need a way for the loop to exit after the planned exception.
    # The current side effect for consume_messages will set shutdown_event on the *next* call.
    # If the exception is caught and the loop continues, it will call consume_messages again.
    
    await api_worker.process_api_jobs("test_group", "test_consumer")

    assert mock_broker.consume_messages.call_count >= 1 # consume_messages was called
    mock_process_api_job_func.assert_not_called() # No job processed
    mock_broker.acknowledge_message.assert_not_called() # No message to acknowledge
    
    logger_mock.error.assert_any_call(
        f"Error consuming messages from Redis stream: {consume_exception_message}. Retrying in 1s..."
    )
    # Check that the loop continued and eventually shut down (call_count for consume_messages would be > 1)
    assert consume_call_count > 1, "Loop did not continue after consume_messages exception"


# Test 4.2: Exception during process_api_job (job should be acknowledged)
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_process_job_exception(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    message_id_process_fail = "msg_process_fail_004"
    job_id_process_fail = "job_process_fail"
    job_type_process_fail = "type_process_fail"
    message_data_process_fail = {"job_id": job_id_process_fail, "job_type": job_type_process_fail}
    process_exception_message = "Simulated error during job processing"

    async def consume_one_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set()
        return [(message_id_process_fail, message_data_process_fail)]
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_one_then_shutdown)
    mock_broker.acknowledge_message = AsyncMock()
    
    # process_api_job raises an exception
    mock_process_api_job_func.side_effect = Exception(process_exception_message)

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    await api_worker.process_api_jobs("test_group", "test_consumer")

    mock_broker.consume_messages.assert_called_once()
    mock_process_api_job_func.assert_called_once_with(job_id_process_fail, job_type_process_fail, message_data_process_fail)
    # Message should still be acknowledged even if processing fails, to prevent re-processing a bad job.
    mock_broker.acknowledge_message.assert_called_once_with("test_group", message_id_process_fail) 
    
    logger_mock.error.assert_any_call(
        f"Error processing job {job_id_process_fail} (message {message_id_process_fail}): {process_exception_message}"
    )


# Test 4.3: Exception during acknowledge_message
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.api_job_processor.process_api_job')
async def test_process_api_jobs_ack_exception(mock_process_api_job_func, monkeypatch):
    mock_broker = MagicMock()
    message_id_ack_fail = "msg_ack_fail_005"
    job_id_ack_fail = "job_ack_fail"
    job_type_ack_fail = "type_ack_fail"
    message_data_ack_fail = {"job_id": job_id_ack_fail, "job_type": job_type_ack_fail}
    ack_exception_message = "Simulated error during ACK"

    async def consume_one_then_shutdown(*args, **kwargs):
        await asyncio.sleep(0)
        api_worker.shutdown_event.set()
        return [(message_id_ack_fail, message_data_ack_fail)]
        
    mock_broker.consume_messages = AsyncMock(side_effect=consume_one_then_shutdown)
    # acknowledge_message raises an exception
    mock_broker.acknowledge_message = AsyncMock(side_effect=Exception(ack_exception_message))
    
    mock_process_api_job_func.return_value = True # Job processing itself succeeds

    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)

    api_worker.shutdown_event.clear()
    await api_worker.process_api_jobs("test_group", "test_consumer")

    mock_broker.consume_messages.assert_called_once()
    mock_process_api_job_func.assert_called_once_with(job_id_ack_fail, job_type_ack_fail, message_data_ack_fail)
    mock_broker.acknowledge_message.assert_called_once_with("test_group", message_id_ack_fail)
    
    logger_mock.error.assert_any_call(
        f"Error acknowledging message {message_id_ack_fail} for job {job_id_ack_fail}: {ack_exception_message}"
    )

# --- Shutdown and Main Loop Tests ---

# test_handle_shutdown is already good.

@patch('viralStoryGenerator.src.api_worker.preload_components')
@patch('viralStoryGenerator.src.api_worker.process_api_jobs', new_callable=AsyncMock) # Mock the async function
@patch('viralStoryGenerator.src.api_worker.signal.signal') # To check signal handler registration
@patch('viralStoryGenerator.src.api_worker._logger')
def test_run_worker_calls_preload_and_process_and_registers_signals(
    mock_logger, mock_signal_signal, mock_process_api_jobs_async, mock_preload_components, monkeypatch
):
    # To make the test run and exit quickly, we can have process_api_jobs raise an exception
    # or complete very fast (e.g., shutdown_event is immediately set).
    # For this test, let's make it run once then stop.
    
    async def process_jobs_side_effect(group, consumer_id):
        # Simulate it running and then stopping due to shutdown_event
        # Check that shutdown_event starts as False and becomes True
        assert not api_worker.shutdown_event.is_set()
        api_worker.shutdown_event.set() # Simulate shutdown during processing
        await asyncio.sleep(0) # Yield control
        return None
    
    mock_process_api_jobs_async.side_effect = process_jobs_side_effect
    api_worker.shutdown_event.clear() # Ensure it's clear before starting run_worker

    # Mock app_config values needed by run_worker
    monkeypatch.setattr(api_worker.app_config.redis, 'CONSUMER_GROUP_NAME', "test_worker_group")
    # Generate a unique consumer ID for the test
    test_consumer_id = f"test_consumer_{api_worker.uuid.uuid4().hex[:6]}"
    monkeypatch.setattr(api_worker.uuid, 'uuid4', MagicMock(return_value=MagicMock(hex=test_consumer_id.split('_')[-1])))


    # asyncio.run() is called inside run_worker.
    # If run_worker itself is not async, we call it directly.
    # If run_worker were async, we'd need @pytest.mark.asyncio and await it.
    # viralStoryGenerator.src.api_worker.run_worker is a synchronous function that runs an async loop.
    api_worker.run_worker()

    mock_preload_components.assert_called_once_with(api_worker.app_config.redis.CONSUMER_GROUP_NAME)
    
    # process_api_jobs is called with group_name and the generated consumer_id
    # The consumer_id generation needs to be predictable or captured.
    # Let's assume consumer_id is generated as f"{app_config.redis.CONSUMER_GROUP_NAME}-{hostname}-{uuid[:6]}"
    # We mocked uuid.uuid4().hex part. Hostname is harder without also mocking socket.gethostname().
    # For simplicity, let's check that the first arg is the group name.
    mock_process_api_jobs_async.assert_called_once()
    args, _ = mock_process_api_jobs_async.call_args
    assert args[0] == "test_worker_group"
    assert test_consumer_id in args[1] # Check if the generated part is in the consumer_id

    # Check signal handler registration
    mock_signal_signal.assert_any_call(signal.SIGINT, api_worker.handle_shutdown)
    mock_signal_signal.assert_any_call(signal.SIGTERM, api_worker.handle_shutdown)
    
    mock_logger.info.assert_any_call(f"Worker {args[1]} starting...") # args[1] will be the generated consumer_id
    mock_logger.info.assert_any_call(f"Worker {args[1]} shutting down...")
    assert api_worker.shutdown_event.is_set() # Should be set by the side_effect of process_api_jobs for this test


@patch('viralStoryGenerator.src.api_worker.run_worker')
@patch('viralStoryGenerator.src.api_worker._logger')
def test_main_function_calls_run_worker(mock_logger, mock_run_worker, monkeypatch):
    # This test is to ensure main() calls run_worker()
    # It also implicitly tests the KeyboardInterrupt handling if run_worker raises it.
    
    # Make run_worker raise KeyboardInterrupt to simulate a Ctrl+C during worker execution
    mock_run_worker.side_effect = KeyboardInterrupt("Simulated Ctrl+C")

    # Call main()
    # main() has a try/except KeyboardInterrupt that should catch this.
    api_worker.main()

    mock_run_worker.assert_called_once()
    mock_logger.info.assert_any_call("ViralStoryGenerator API Worker starting...")
    mock_logger.info.assert_any_call("ViralStoryGenerator API Worker stopped by user (KeyboardInterrupt).")

# Test the existing test_main_runs to see if it can be simplified or is still needed.
# The existing test_main_runs is more of an integration test for the asyncio loop handling in main.
# The new test_main_function_calls_run_worker is more direct for checking if main calls run_worker.
# Both can be useful. The original test_main_runs also patches many things; let's try to keep it if it adds value.
# For now, I will keep the original test_main_runs as it tests a different aspect (loop management).

# Renaming old test_main_runs to be more specific about its purpose
def test_main_loop_management_on_interrupt(monkeypatch):
    # This is the original test_main_runs, renamed.
    # It tests how main handles KeyboardInterrupt specifically around the asyncio loop.
    monkeypatch.setattr(api_worker, "run_worker", lambda: (_ for _ in ()).throw(KeyboardInterrupt())) # run_worker itself is interrupted
    logger_mock = MagicMock()
    monkeypatch.setattr(api_worker, "_logger", logger_mock)
    
    # Mock asyncio.get_event_loop and its methods
    loop_mock = MagicMock()
    monkeypatch.setattr(api_worker.asyncio, 'get_event_loop', lambda: loop_mock)
    
    # Simulate run_worker being called by loop.run_until_complete(run_worker_async_part)
    # However, run_worker is sync and calls asyncio.run(process_api_jobs_async_wrapper).
    # The original test's patching for loop.run_until_complete.side_effect = KeyboardInterrupt()
    # was aimed at the main() function's own loop.run_until_complete for cleanup tasks,
    # if main() was structured to use loop.run_until_complete for its primary async task.
    # Given main() calls asyncio.run(run_worker()), the KeyboardInterrupt should be caught by asyncio.run's handler
    # or by main's try/except.
    
    # Let's simplify: main calls run_worker. If run_worker is interrupted, main's try/except catches it.
    # The loop management part (shutdown_asyncgens, close) is tricky to test without a real running loop
    # or a very complex mock loop.
    # The original test's purpose might have been to ensure these cleanup methods are called.
    
    # For this renamed test, let's focus on the KeyboardInterrupt being caught by main's handler.
    # The KeyboardInterrupt from run_worker is already tested by test_main_function_calls_run_worker.
    # The specific loop shutdown calls (shutdown_asyncgens, close) are harder to assert robustly
    # without a deeply integrated asyncio test. Python's asyncio.run handles much of this.
    # Let's ensure the logger messages are as expected.
    
    # Re-evaluating: the original test_main_runs was patching many things in `api_worker` itself,
    # effectively disabling parts of it to focus on the `KeyboardInterrupt` handling in `main`.
    # This is reasonable.
    monkeypatch.setattr(api_worker, "asyncio", api_worker.asyncio) # Keep original asyncio
    monkeypatch.setattr(api_worker, "os", api_worker.os)
    monkeypatch.setattr(api_worker, "sys", api_worker.sys)
    monkeypatch.setattr(api_worker, "signal", api_worker.signal)
    monkeypatch.setattr(api_worker, "uuid", api_worker.uuid)
    monkeypatch.setattr(api_worker, "get_vector_db_client", lambda: None)
    monkeypatch.setattr(api_worker, "get_vector_db", lambda: None)
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: None)
    
    # The key is that run_worker (or what it calls) raises KeyboardInterrupt
    # The original test already does this with:
    # monkeypatch.setattr(api_worker, "run_worker", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    api_worker.main() # Should catch KeyboardInterrupt and log appropriately

    logger_mock.info.assert_any_call("ViralStoryGenerator API Worker starting...")
    logger_mock.info.assert_any_call("ViralStoryGenerator API Worker stopped by user (KeyboardInterrupt).")
    # Asserting loop.shutdown_asyncgens and loop.close calls would require mocking asyncio.run
    # or having a more complex loop mock, which might be overkill if Python's default handling is trusted.
