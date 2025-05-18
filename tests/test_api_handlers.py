import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import signal
import os
import logging
import sys # For sys.exit mocking, if used (though current handle_shutdown doesn't)

from viralStoryGenerator.src import api_worker
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For spec
# This import is crucial if appconfig is patched directly in its source module
from viralStoryGenerator.utils import config as utils_config_module


# --- Fixtures ---

@pytest.fixture(autouse=True)
def reset_worker_globals(monkeypatch):
    """Ensures global states in api_worker are reset before each test."""
    if hasattr(api_worker, '_message_broker'):
        monkeypatch.setattr(api_worker, "_message_broker", None, raising=False)
    if hasattr(api_worker, '_vector_db_client'): # If your worker uses this
        monkeypatch.setattr(api_worker, "_vector_db_client", None, raising=False)
    # Reset shutdown_flag if it's a module global in api_worker
    if hasattr(api_worker, 'shutdown_flag'):
        monkeypatch.setattr(api_worker, "shutdown_flag", False, raising=False)
    # If api_worker uses an asyncio.Event for shutdown:
    if hasattr(api_worker, 'shutdown_event') and hasattr(api_worker.shutdown_event, 'clear'):
        api_worker.shutdown_event.clear() # Assuming it's an asyncio.Event

@pytest.fixture
def mock_appconfig_worker(monkeypatch):
    class DummyRedisConfig:
        HOST = "worker-redis-host.test"
        PORT = 16379
        STREAM_NAME_API_JOBS = "test_api_jobs_stream_worker"
        CONSUMER_GROUP_API_JOBS = "test_api_jobs_group_worker"

    class Config:
        redis = DummyRedisConfig()
        # Add other necessary config attributes your api_worker.py might use
        # Example: LOG_LEVEL = "DEBUG"

    config_instance = Config()
    # --- CRITICAL: ADJUST THIS PATCH TARGET BASED ON YOUR api_worker.py ---
    # Option 1: If api_worker.py has: "from viralStoryGenerator.utils.config import config as appconfig"
    # monkeypatch.setattr(api_worker, "appconfig", config_instance)
    # Option 2: If api_worker.py (or functions it calls) has: "from viralStoryGenerator.utils.config import config"
    #           and uses it as "config.redis.HOST", then patch the 'config' object in its source module.
    monkeypatch.setattr(utils_config_module, "config", config_instance)
    # Option 3: If api_worker.py has: "from viralStoryGenerator.utils.config import config as some_other_alias"
    # monkeypatch.setattr(api_worker, "some_other_alias", config_instance)
    return config_instance

@pytest.fixture
def mock_broker_class(monkeypatch):
    """Mocks the RedisMessageBroker class itself."""
    mock_class = MagicMock(spec=RedisMessageBroker, name="RedisMessageBroker_ClassMock")

    # This instance is what RedisMessageBroker(...) will return
    mock_instance = MagicMock(spec=RedisMessageBroker, name="RedisMessageBroker_InstanceMock")
    mock_instance.initialize = AsyncMock(name="initialize_AsyncMock")
    mock_instance.ensure_stream_exists = AsyncMock(name="ensure_stream_exists_AsyncMock")
    mock_instance.create_consumer_group = AsyncMock(name="create_consumer_group_AsyncMock")
    mock_instance.consume_messages = AsyncMock(name="consume_messages_AsyncMock", return_value=[]) # Default: no messages
    mock_instance.acknowledge_message = AsyncMock(name="acknowledge_message_AsyncMock")

    mock_class.return_value = mock_instance
    # Patch RedisMessageBroker where it's imported/defined for api_worker.py
    monkeypatch.setattr(api_worker, "RedisMessageBroker", mock_class)
    return mock_class

@pytest.fixture
def mock_logger_worker(monkeypatch):
    """Mocks the _logger instance in api_worker."""
    logger_mock = MagicMock(spec=logging.Logger, name="ApiWorker_LoggerMock")
    # Explicitly mock methods used for assertions if spec isn't enough
    logger_mock.info = MagicMock(name="logger_info_mock")
    logger_mock.debug = MagicMock(name="logger_debug_mock")
    logger_mock.warning = MagicMock(name="logger_warning_mock")
    logger_mock.error = MagicMock(name="logger_error_mock")
    logger_mock.exception = MagicMock(name="logger_exception_mock")

    monkeypatch.setattr(api_worker, "_logger", logger_mock)
    return logger_mock

# --- Tests ---

def test_handle_shutdown(mock_logger_worker):
    # Assuming SUT's handle_shutdown sets a global flag and logs.
    # The example SUT provided previously does this.
    # Ensure the `reset_worker_globals` fixture has reset `shutdown_flag`.
    # Or set it explicitly here if it's not covered by the global reset.
    if hasattr(api_worker, 'shutdown_flag'):
        api_worker.shutdown_flag = False

    api_worker.handle_shutdown(signal.SIGINT, None)

    if hasattr(api_worker, 'shutdown_flag'):
        assert api_worker.shutdown_flag is True
    mock_logger_worker.info.assert_called_with(f"Shutdown signal {signal.SIGINT} received, initiating graceful shutdown...")

    # Reset flag again if it's a global, for isolation from other tests (though reset_worker_globals should handle this)
    if hasattr(api_worker, 'shutdown_flag'):
        api_worker.shutdown_flag = False


def test_preload_components_creates_group(mock_appconfig_worker, mock_broker_class, mock_logger_worker):
    target_group_name = "preload_test_group_new"
    mock_broker_instance = mock_broker_class.return_value # The instance returned by RedisMessageBroker()

    api_worker.preload_components(consumer_group_name=target_group_name)

    expected_redis_url = f"redis://{mock_appconfig_worker.redis.HOST}:{mock_appconfig_worker.redis.PORT}"
    mock_broker_class.assert_called_once() # Was RedisMessageBroker(...) called?

    instantiation_kwargs = mock_broker_class.call_args[1] # Get kwargs of instantiation
    assert instantiation_kwargs['redis_url'] == expected_redis_url
    assert instantiation_kwargs['stream_name'] == mock_appconfig_worker.redis.STREAM_NAME_API_JOBS
    assert instantiation_kwargs['group_name'] == target_group_name
    assert 'consumer_name' in instantiation_kwargs # Preload uses a dynamic consumer name

    # Assertions on the instance methods
    mock_broker_instance.initialize.assert_awaited_once()
    mock_broker_instance.ensure_stream_exists.assert_awaited_once()
    # The arguments to create_consumer_group might depend on your actual SUT's get_message_broker
    mock_broker_instance.create_consumer_group.assert_awaited_once_with(
        group_name=target_group_name,
        stream_name=mock_appconfig_worker.redis.STREAM_NAME_API_JOBS,
        create_stream_if_not_exists=False # Based on SUT example
    )

    mock_logger_worker.info.assert_any_call("Preloading API worker components...")
    mock_logger_worker.info.assert_any_call("Message broker initialized successfully.") # From SUT's get_message_broker
    mock_logger_worker.info.assert_any_call("Message broker pre-warmed (stream/group checked/created).")
    mock_logger_worker.info.assert_any_call("API worker components preloaded.")


def test_preload_components_group_exists(mock_appconfig_worker, mock_broker_class, mock_logger_worker):
    target_group_name = "preload_existing_group"
    mock_broker_instance = mock_broker_class.return_value
    # Simulate BUSYGROUP error
    mock_broker_instance.create_consumer_group.side_effect = Exception("BUSYGROUP Consumer Group name already exists")

    api_worker.preload_components(consumer_group_name=target_group_name)

    mock_broker_class.assert_called_once() # Still instantiated
    # ... other instantiation checks if necessary ...

    mock_broker_instance.initialize.assert_awaited_once()
    mock_broker_instance.ensure_stream_exists.assert_awaited_once()
    mock_broker_instance.create_consumer_group.assert_awaited_once() # Called, and it raised the side_effect

    # Check for specific logging of BUSYGROUP (SUT example logs this as INFO)
    assert any(
        ("already exists" in call.args[0].lower() or "busygroup" in call.args[0].lower())
        for call in mock_logger_worker.info.call_args_list
    ), "Log message for existing group (BUSYGROUP) not found or not handled as expected."
    mock_logger_worker.info.assert_any_call("Message broker pre-warmed (stream/group checked/created).")


@pytest.mark.anyio # Use anyio for async tests
async def test_get_message_broker_initializes_new_async(mock_appconfig_worker, mock_broker_class, mock_logger_worker):
    # reset_worker_globals fixture ensures _message_broker is None
    assert api_worker._message_broker is None

    mock_broker_instance = mock_broker_class.return_value # Instance from RedisMessageBroker(...)

    redis_url = f"redis://{mock_appconfig_worker.redis.HOST}:{mock_appconfig_worker.redis.PORT}"
    stream_name = mock_appconfig_worker.redis.STREAM_NAME_API_JOBS
    group_name = "test_init_group_async"
    consumer_name = "test_init_consumer_async"

    # Call the SUT's async get_message_broker function
    broker = await api_worker.get_message_broker(
        redis_url=redis_url,
        stream_name=stream_name,
        group_name=group_name,
        consumer_name=consumer_name
    )

    assert broker == mock_broker_instance # Should return the mocked instance
    mock_broker_class.assert_called_once_with( # Check instantiation arguments
        redis_url=redis_url, stream_name=stream_name, group_name=group_name, consumer_name=consumer_name
    )
    mock_broker_instance.initialize.assert_awaited_once()
    mock_broker_instance.ensure_stream_exists.assert_awaited_once()
    mock_broker_instance.create_consumer_group.assert_awaited_once_with(
        group_name=group_name, stream_name=stream_name, create_stream_if_not_exists=False # From SUT example
    )
    assert api_worker._message_broker == mock_broker_instance # Global should be set
    mock_logger_worker.info.assert_any_call("Message broker initialized successfully.")


@pytest.mark.anyio
async def test_get_message_broker_returns_existing_async(monkeypatch, mock_appconfig_worker, mock_logger_worker):
    # Setup: Pre-set the global _message_broker
    existing_mock_broker_instance = MagicMock(spec=RedisMessageBroker, name="ExistingBrokerInstance")
    monkeypatch.setattr(api_worker, "_message_broker", existing_mock_broker_instance)

    # This mock ensures RedisMessageBroker CLASS is NOT called for a new instance
    non_called_broker_class_constructor = MagicMock(name="NonCalledBrokerClassConstructor")
    monkeypatch.setattr(api_worker, "RedisMessageBroker", non_called_broker_class_constructor)

    broker = await api_worker.get_message_broker( # Call SUT's async function
        redis_url="dummy_url_not_used_if_existing", stream_name="dummy_stream_not_used_if_existing",
        group_name="dummy_group_not_used_if_existing", consumer_name="dummy_consumer_not_used_if_existing"
    )

    assert broker == existing_mock_broker_instance # Should return the pre-set one
    non_called_broker_class_constructor.assert_not_called() # Critically, no new instance made
    mock_logger_worker.debug.assert_any_call("Reusing existing message broker.")


@pytest.mark.anyio
async def test_process_api_jobs_handles_broker_init_failure_async(mock_appconfig_worker, mock_logger_worker, monkeypatch):
    # Simulate SUT's get_message_broker failing by returning None
    async def mock_get_broker_returns_none_async(*args, **kwargs): # Must be async if SUT awaits it
        mock_logger_worker.error("Simulated get_message_broker async failure (returning None).")
        return None
    monkeypatch.setattr(api_worker, "get_message_broker", mock_get_broker_returns_none_async)

    # Ensure the loop in process_api_jobs terminates for the test
    original_async_sleep = asyncio.sleep
    async def single_pass_after_fail_async_sleep(duration):
        # This sleep is called if broker is None and loop continues before shutdown_flag check
        # Or if SUT has a sleep inside the main while loop
        if hasattr(api_worker, 'shutdown_flag'):
            monkeypatch.setattr(api_worker, "shutdown_flag", True, raising=False)
        elif hasattr(api_worker, 'shutdown_event'):
             api_worker.shutdown_event.set()
        await original_async_sleep(0.0001)
    monkeypatch.setattr(asyncio, "sleep", single_pass_after_fail_async_sleep)

    await api_worker.process_api_jobs("test_group_fail_async", "test_consumer_fail_async")

    # Check that our mock for get_message_broker was called
    mock_logger_worker.error.assert_any_call("Simulated get_message_broker async failure (returning None).")
    # Check that SUT logs the failure to get a broker
    mock_logger_worker.error.assert_any_call("Failed to initialize message broker. Worker cannot start.")


@pytest.mark.anyio
async def test_process_api_jobs_consumes_and_acknowledges_async(mock_appconfig_worker, mock_broker_class, mock_logger_worker, monkeypatch):
    mock_broker_instance = mock_broker_class.return_value # Instance from RedisMessageBroker(...)

    test_message_id = "msg-consume-ack-0"
    test_job_id = "job-consume-ack-1"
    test_message_data = {"job_id": test_job_id, "type": "test_job"}

    consume_call_count = 0
    async def mock_consume_side_effect_controlled_async(*args, **kwargs):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1: # First call, return one message
            return [(test_message_id, test_message_data)]
        else: # Subsequent calls, return empty and trigger shutdown
            if hasattr(api_worker, 'shutdown_flag'):
                monkeypatch.setattr(api_worker, "shutdown_flag", True, raising=False)
            elif hasattr(api_worker, 'shutdown_event'):
                api_worker.shutdown_event.set()
            return []
    mock_broker_instance.consume_messages = AsyncMock(side_effect=mock_consume_side_effect_controlled_async)

    # Mock the actual job processing function if your SUT calls one
    # For example, if api_worker.py has: from viralStoryGenerator.utils.api_job_processor import process_job_payload
    # mock_actual_job_processor = AsyncMock(name="ActualJobProcessorMock")
    # monkeypatch.setattr("viralStoryGenerator.utils.api_job_processor.process_job_payload", mock_actual_job_processor, raising=False)
    # Or if it's a method within api_worker.py itself:
    # monkeypatch.setattr(api_worker, "some_internal_job_processor_func", mock_actual_job_processor, raising=False)

    await api_worker.process_api_jobs("test_consume_group_ack", "test_consumer_main_ack")

    mock_broker_instance.consume_messages.assert_called() # consume_messages was called

    # If you mocked a specific job processor and expect it to be called:
    # mock_actual_job_processor.assert_awaited_once_with(test_message_data, mock_broker_instance, test_message_id) # Args depend on SUT

    # If your SUT's process_api_jobs includes acknowledgement:
    # mock_broker_instance.acknowledge_message.assert_awaited_once_with(test_message_id) # Or (group_name, message_id)

    mock_logger_worker.info.assert_any_call(f"Worker test_consumer_main_ack starting to process jobs from group test_consume_group_ack...")
    # If your SUT logs message receipt:
    # mock_logger_worker.info.assert_any_call(f"Worker test_consumer_main_ack received job: {test_message_id} - {test_job_id}")
    mock_logger_worker.info.assert_any_call(f"Worker test_consumer_main_ack shutting down.")


@patch("viralStoryGenerator.src.api_worker.preload_components")
@patch("viralStoryGenerator.src.api_worker.process_api_jobs", new_callable=AsyncMock) # process_api_jobs is async
@patch("signal.signal") # Mock signal registration
@patch("os.getpid", return_value=54321) # Mock os.getpid for predictable consumer name in main
def test_main_runs(
    mock_os_getpid, mock_signal_registration,
    mock_process_api_jobs_for_main, mock_preload_components_for_main,
    mock_appconfig_worker, mock_logger_worker # These fixtures are still needed for SUT's main
):
    # Simulate SUT's process_api_jobs running then being interrupted by KeyboardInterrupt
    async def process_side_effect_in_main(*args, **kwargs):
        mock_logger_worker.info("mock_process_api_jobs_for_main called, simulating run then raising KeyboardInterrupt.")
        # SUT's main() has asyncio.run(process_api_jobs(...))
        # This mock will be run by that asyncio.run() call.
        raise KeyboardInterrupt
    mock_process_api_jobs_for_main.side_effect = process_side_effect_in_main

    # Call the SUT's main function.
    # It should catch the KeyboardInterrupt from mock_process_api_jobs_for_main.
    try:
        api_worker.main()
    except KeyboardInterrupt: # Should NOT happen if SUT's main catches it
        pytest.fail("KeyboardInterrupt was not caught by SUT's main function's try/except block.")

    mock_preload_components_for_main.assert_called_once_with(mock_appconfig_worker.redis.CONSUMER_GROUP_API_JOBS)
    mock_process_api_jobs_for_main.assert_awaited_once_with(
        mock_appconfig_worker.redis.CONSUMER_GROUP_API_JOBS,
        f"api-worker-54321" # Uses mocked os.getpid value
    )
    assert mock_signal_registration.call_count == 2 # SIGINT and SIGTERM
    mock_signal_registration.assert_any_call(signal.SIGINT, api_worker.handle_shutdown)
    mock_signal_registration.assert_any_call(signal.SIGTERM, api_worker.handle_shutdown)

    # Check logs from main's try/except/finally blocks
    mock_logger_worker.info.assert_any_call(f"Starting API worker: api-worker-54321 in group: {mock_appconfig_worker.redis.CONSUMER_GROUP_API_JOBS}")
    mock_logger_worker.info.assert_any_call("KeyboardInterrupt caught in main, shutting down.")
    mock_logger_worker.info.assert_any_call("API worker main function finished.")