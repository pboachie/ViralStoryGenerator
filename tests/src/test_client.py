import pytest
import asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming the client module is viralStoryGenerator.src.client
from viralStoryGenerator.src import client as client_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For type hinting and spec for mocks

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_app_config(monkeypatch):
    """Fixture to provide a mock app_config with default test values."""
    monkeypatch.setattr(app_config.redis, 'HOST', "mockhost")
    monkeypatch.setattr(app_config.redis, 'PORT', 1234)
    monkeypatch.setattr(app_config.redis, 'API_JOB_STREAM_NAME', "test_api_stream")
    monkeypatch.setattr(app_config.redis, 'SCRAPER_JOB_STREAM_NAME', "test_scrape_stream")
    # Add other config values as needed by client functions
    monkeypatch.setattr(app_config.client, 'JOB_COMPLETION_TIMEOUT_SECONDS', 0.1) # Short timeout for tests
    monkeypatch.setattr(app_config.client, 'JOB_STATUS_POLL_INTERVAL_SECONDS', 0.01)
    return app_config

@pytest.fixture(autouse=True)
def reset_client_module_globals():
    """Reset global broker instances in client.py before each test."""
    client_module._api_message_broker = None
    client_module._scrape_message_broker = None

# --- Tests for Scenario 1: get_api_message_broker and get_scrape_message_broker ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.RedisMessageBroker', spec=RedisMessageBroker) # Patch where it's imported in client.py
async def test_get_api_message_broker_instantiates_correctly(MockRedisMessageBroker, mock_app_config):
    # Ensure it's reset for this specific test of creation
    client_module._api_message_broker = None 
    
    mock_broker_instance = MagicMock(spec=RedisMessageBroker)
    mock_broker_instance.initialize = AsyncMock() # Mock the async initialize method
    MockRedisMessageBroker.return_value = mock_broker_instance

    broker = await client_module.get_api_message_broker()

    assert broker is mock_broker_instance
    expected_redis_url = f"redis://{mock_app_config.redis.HOST}:{mock_app_config.redis.PORT}"
    MockRedisMessageBroker.assert_called_once_with(
        redis_url=expected_redis_url,
        stream_name=mock_app_config.redis.API_JOB_STREAM_NAME
    )
    broker.initialize.assert_called_once() # Ensure initialize was called

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.RedisMessageBroker', spec=RedisMessageBroker)
async def test_get_api_message_broker_returns_existing_instance(MockRedisMessageBroker, mock_app_config):
    mock_existing_broker = MagicMock(spec=RedisMessageBroker)
    client_module._api_message_broker = mock_existing_broker # Pre-set the global broker

    broker = await client_module.get_api_message_broker()

    assert broker is mock_existing_broker
    MockRedisMessageBroker.assert_not_called() # Should not create a new one

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.RedisMessageBroker', spec=RedisMessageBroker)
async def test_get_scrape_message_broker_instantiates_correctly(MockRedisMessageBroker, mock_app_config):
    client_module._scrape_message_broker = None

    mock_broker_instance = MagicMock(spec=RedisMessageBroker)
    mock_broker_instance.initialize = AsyncMock()
    MockRedisMessageBroker.return_value = mock_broker_instance

    broker = await client_module.get_scrape_message_broker()

    assert broker is mock_broker_instance
    expected_redis_url = f"redis://{mock_app_config.redis.HOST}:{mock_app_config.redis.PORT}"
    MockRedisMessageBroker.assert_called_once_with(
        redis_url=expected_redis_url,
        stream_name=mock_app_config.redis.SCRAPER_JOB_STREAM_NAME
    )
    broker.initialize.assert_called_once()

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.RedisMessageBroker', spec=RedisMessageBroker)
async def test_get_scrape_message_broker_returns_existing_instance(MockRedisMessageBroker, mock_app_config):
    mock_existing_broker = MagicMock(spec=RedisMessageBroker)
    client_module._scrape_message_broker = mock_existing_broker

    broker = await client_module.get_scrape_message_broker()

    assert broker is mock_existing_broker
    MockRedisMessageBroker.assert_not_called()

# --- Tests for Scenario 2: queue_api_request ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_api_message_broker')
@patch('uuid.uuid4')
async def test_queue_api_request_no_wait_successful(mock_uuid4, mock_get_api_broker, mock_app_config):
    job_id = "api_job_no_wait_123"
    message_id = "msg_id_api_no_wait"
    mock_uuid4.return_value = MagicMock(hex=job_id)
    
    mock_broker = AsyncMock(spec=RedisMessageBroker) # Use AsyncMock for async methods
    mock_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_api_broker.return_value = mock_broker

    payload = {"topic": "Test API Request", "param": "value"}
    
    returned_job_id = await client_module.queue_api_request(payload, wait_for_result=False)

    assert returned_job_id == job_id
    mock_get_api_broker.assert_called_once()
    
    mock_broker.publish_message.assert_called_once()
    args, _ = mock_broker.publish_message.call_args
    published_payload = args[0]
    assert published_payload["job_id"] == job_id
    assert published_payload["message_type"] == "api_request"
    assert published_payload["payload"]["topic"] == "Test API Request"
    assert published_payload["payload"]["param"] == "value"


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.wait_for_job_result') # Patch wait_for_job_result
@patch('viralStoryGenerator.src.client.get_api_message_broker')
@patch('uuid.uuid4')
async def test_queue_api_request_with_wait_completes(mock_uuid4, mock_get_api_broker, mock_wait_for_result, mock_app_config):
    job_id = "api_job_wait_complete_456"
    message_id = "msg_id_api_wait_complete"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_api_broker.return_value = mock_broker

    # Simulate wait_for_job_result returning a completed status
    completed_status = {"job_id": job_id, "status": "completed", "result": "some_result"}
    mock_wait_for_result.return_value = completed_status

    payload = {"topic": "Test API Wait Complete"}
    
    result = await client_module.queue_api_request(payload, wait_for_result=True)

    assert result == completed_status # Should return the result from wait_for_job_result
    mock_get_api_broker.assert_called_once()
    mock_broker.publish_message.assert_called_once()
    mock_wait_for_result.assert_called_once_with(
        job_id, 
        mock_broker, # The broker instance should be passed
        timeout_seconds=mock_app_config.client.JOB_COMPLETION_TIMEOUT_SECONDS,
        poll_interval_seconds=mock_app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS
    )


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.wait_for_job_result')
@patch('viralStoryGenerator.src.client.get_api_message_broker')
@patch('uuid.uuid4')
async def test_queue_api_request_with_wait_times_out(mock_uuid4, mock_get_api_broker, mock_wait_for_result, mock_app_config):
    job_id = "api_job_wait_timeout_789"
    message_id = "msg_id_api_wait_timeout"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_api_broker.return_value = mock_broker

    # Simulate wait_for_job_result returning a timeout status (or None, depending on its contract)
    # Let's assume it returns a status dict indicating timeout as per scenario 5 description
    timeout_status = {"job_id": job_id, "status": "timeout", "error_message": "Job timed out"}
    mock_wait_for_result.return_value = timeout_status

    payload = {"topic": "Test API Wait Timeout"}
    
    result = await client_module.queue_api_request(payload, wait_for_result=True)

    assert result == timeout_status # Should return the timeout status
    mock_wait_for_result.assert_called_once()


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_api_message_broker')
@patch('uuid.uuid4')
async def test_queue_api_request_publish_fails(mock_uuid4, mock_get_api_broker, mock_app_config):
    job_id = "api_job_publish_fail_000"
    mock_uuid4.return_value = MagicMock(hex=job_id) # job_id is generated before publish attempt

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    # Simulate publish_message failing by returning None
    mock_broker.publish_message = AsyncMock(return_value=None) 
    mock_get_api_broker.return_value = mock_broker

    payload = {"topic": "Test API Publish Fail"}
    
    result = await client_module.queue_api_request(payload, wait_for_result=False)

    assert result is None # Should return None if publishing fails
    mock_broker.publish_message.assert_called_once()


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_api_message_broker')
@patch('uuid.uuid4')
async def test_queue_api_request_publish_raises_exception(mock_uuid4, mock_get_api_broker, mock_app_config):
    job_id = "api_job_publish_exception_111"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    # Simulate publish_message raising an exception
    mock_broker.publish_message = AsyncMock(side_effect=Exception("Redis dead"))
    mock_get_api_broker.return_value = mock_broker

    payload = {"topic": "Test API Publish Exception"}
    
    result = await client_module.queue_api_request(payload, wait_for_result=False)

    assert result is None # Should return None if publishing raises an exception
    mock_broker.publish_message.assert_called_once()

# --- Tests for Scenario 3: queue_scrape_request ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_no_wait_successful(mock_uuid4, mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_job_no_wait_123"
    message_id = "msg_id_scrape_no_wait"
    mock_uuid4.return_value = MagicMock(hex=job_id)
    
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_scrape_broker.return_value = mock_broker

    urls_to_scrape = ["http://example.com/scrape1", "http://example.com/scrape2"]
    
    returned_job_id = await client_module.queue_scrape_request(urls_to_scrape, wait_for_result=False)

    assert returned_job_id == job_id
    mock_get_scrape_broker.assert_called_once()
    
    mock_broker.publish_message.assert_called_once()
    args, _ = mock_broker.publish_message.call_args
    published_payload = args[0]
    assert published_payload["job_id"] == job_id
    assert published_payload["message_type"] == "scrape_request"
    assert published_payload["payload"]["urls"] == urls_to_scrape


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.wait_for_job_result')
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_with_wait_completes(mock_uuid4, mock_get_scrape_broker, mock_wait_for_result, mock_app_config):
    job_id = "scrape_job_wait_complete_456"
    message_id = "msg_id_scrape_wait_complete"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_scrape_broker.return_value = mock_broker

    completed_status = {"job_id": job_id, "status": "completed", "scraped_content": ["content1"]}
    mock_wait_for_result.return_value = completed_status

    urls_to_scrape = ["http://example.com/scrape_wait"]
    
    result = await client_module.queue_scrape_request(urls_to_scrape, wait_for_result=True)

    assert result == completed_status
    mock_get_scrape_broker.assert_called_once()
    mock_broker.publish_message.assert_called_once()
    mock_wait_for_result.assert_called_once_with(
        job_id, 
        mock_broker,
        timeout_seconds=mock_app_config.client.JOB_COMPLETION_TIMEOUT_SECONDS,
        poll_interval_seconds=mock_app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS
    )


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.wait_for_job_result')
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_with_wait_times_out(mock_uuid4, mock_get_scrape_broker, mock_wait_for_result, mock_app_config):
    job_id = "scrape_job_wait_timeout_789"
    message_id = "msg_id_scrape_wait_timeout"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_scrape_broker.return_value = mock_broker

    timeout_status = {"job_id": job_id, "status": "timeout", "error_message": "Scrape job timed out"}
    mock_wait_for_result.return_value = timeout_status

    urls_to_scrape = ["http://example.com/scrape_timeout"]
    
    result = await client_module.queue_scrape_request(urls_to_scrape, wait_for_result=True)

    assert result == timeout_status
    mock_wait_for_result.assert_called_once()


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_publish_fails(mock_uuid4, mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_job_publish_fail_000"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(return_value=None) 
    mock_get_scrape_broker.return_value = mock_broker

    urls_to_scrape = ["http://example.com/scrape_publish_fail"]
    
    result = await client_module.queue_scrape_request(urls_to_scrape, wait_for_result=False)

    assert result is None

# --- Tests for Scenario 5: wait_for_job_result ---

@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock) # Mock asyncio.sleep
async def test_wait_for_job_result_completes_quickly(mock_asyncio_sleep, mock_app_config):
    job_id = "wait_completes_fast_001"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    # Simulate get_job_status: first 'processing', then 'completed'
    completed_status = {"job_id": job_id, "status": "completed", "result": "done"}
    call_count = 0
    async def get_status_side_effect(jid):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"job_id": jid, "status": "processing"}
        return completed_status
        
    mock_broker.get_job_status = AsyncMock(side_effect=get_status_side_effect)

    result = await client_module.wait_for_job_result(
        job_id, 
        mock_broker, 
        timeout_seconds=0.1, # Short timeout, but should complete before
        poll_interval_seconds=0.01
    )

    assert result == completed_status
    assert mock_broker.get_job_status.call_count == 2 # Processing, then Completed
    # asyncio.sleep should have been called once (after the 'processing' status)
    mock_asyncio_sleep.assert_called_once_with(mock_app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS)


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_wait_for_job_result_times_out(mock_asyncio_sleep, mock_app_config):
    job_id = "wait_times_out_002"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    # Simulate get_job_status always returning 'processing'
    processing_status = {"job_id": job_id, "status": "processing"}
    mock_broker.get_job_status = AsyncMock(return_value=processing_status)

    timeout_seconds = 0.05 # Set a short timeout for the test
    poll_interval = 0.01
    monkeypatch = pytest.MonkeyPatch() # Use monkeypatch from pytest for app_config if needed here
    monkeypatch.setattr(mock_app_config.client, 'JOB_COMPLETION_TIMEOUT_SECONDS', timeout_seconds)
    monkeypatch.setattr(mock_app_config.client, 'JOB_STATUS_POLL_INTERVAL_SECONDS', poll_interval)


    result = await client_module.wait_for_job_result(
        job_id, 
        mock_broker,
        timeout_seconds=timeout_seconds, # Pass explicitly or rely on patched app_config
        poll_interval_seconds=poll_interval
    )

    assert result["job_id"] == job_id
    assert result["status"] == "timeout"
    assert "timed out after" in result["error_message"]
    
    # Check how many times get_job_status and sleep were called
    # Expected calls: timeout / poll_interval = 0.05 / 0.01 = 5 calls
    # The loop runs as long as elapsed_time < timeout_seconds.
    # On the last check before timeout, it might fetch status, then timeout.
    assert mock_broker.get_job_status.call_count >= (timeout_seconds / poll_interval)
    assert mock_asyncio_sleep.call_count >= (timeout_seconds / poll_interval) -1 # Sleep happens before next check


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock) # Mock asyncio.sleep, not strictly needed if fails fast
async def test_wait_for_job_result_fails(mock_asyncio_sleep, mock_app_config):
    job_id = "wait_job_fails_003"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    failed_status = {"job_id": job_id, "status": "failed", "error_message": "Job processing error"}
    mock_broker.get_job_status = AsyncMock(return_value=failed_status)

    result = await client_module.wait_for_job_result(
        job_id, 
        mock_broker,
        timeout_seconds=0.1, 
        poll_interval_seconds=0.01
    )

    assert result == failed_status
    mock_broker.get_job_status.assert_called_once() # Fails on first check
    mock_asyncio_sleep.assert_not_called() # No sleep if it fails immediately


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_wait_for_job_result_no_status_initially_then_completes(mock_asyncio_sleep, mock_app_config):
    job_id = "wait_no_status_then_complete_004"
    mock_broker = AsyncMock(spec=RedisMessageBroker)

    completed_status = {"job_id": job_id, "status": "completed", "result": "eventual success"}
    call_count = 0
    async def get_status_side_effect_none_then_complete(jid):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None # No status found initially
        return completed_status
        
    mock_broker.get_job_status = AsyncMock(side_effect=get_status_side_effect_none_then_complete)

    result = await client_module.wait_for_job_result(
        job_id, mock_broker, timeout_seconds=0.1, poll_interval_seconds=0.01
    )

    assert result == completed_status
    assert mock_broker.get_job_status.call_count == 2
    mock_asyncio_sleep.assert_called_once_with(mock_app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS)

# --- Tests for Scenario 6: close_redis_connections ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client._logger') # Patch the logger in client.py
async def test_close_redis_connections_no_brokers_initialized(mock_client_logger, mock_app_config):
    # Ensure brokers are None (as per autouse fixture reset_client_module_globals)
    assert client_module._api_message_broker is None
    assert client_module._scrape_message_broker is None

    await client_module.close_redis_connections()

    # Logger message for no active connections
    mock_client_logger.debug.assert_any_call("No active Redis connections to close.")
    # If it iterates through a list or attempts to close None, ensure no errors.


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client._logger')
async def test_close_redis_connections_with_active_brokers(mock_client_logger, mock_app_config):
    # Simulate that brokers were initialized
    mock_api_broker_instance = AsyncMock(spec=RedisMessageBroker)
    mock_scrape_broker_instance = AsyncMock(spec=RedisMessageBroker)
    
    client_module._api_message_broker = mock_api_broker_instance
    client_module._scrape_message_broker = mock_scrape_broker_instance

    await client_module.close_redis_connections()

    mock_api_broker_instance.close.assert_called_once()
    mock_scrape_broker_instance.close.assert_called_once()
    
    mock_client_logger.debug.assert_any_call("Closing API message broker connection...")
    mock_client_logger.debug.assert_any_call("API message broker connection closed.")
    mock_client_logger.debug.assert_any_call("Closing Scrape message broker connection...")
    mock_client_logger.debug.assert_any_call("Scrape message broker connection closed.")


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client._logger')
async def test_close_redis_connections_one_broker_active(mock_client_logger, mock_app_config):
    mock_api_broker_instance = AsyncMock(spec=RedisMessageBroker)
    client_module._api_message_broker = mock_api_broker_instance
    client_module._scrape_message_broker = None # Scrape broker not initialized

    await client_module.close_redis_connections()

    mock_api_broker_instance.close.assert_called_once()
    mock_client_logger.debug.assert_any_call("Closing API message broker connection...")
    mock_client_logger.debug.assert_any_call("API message broker connection closed.")
    # Check that logs for scrape broker are not present if it was None
    # This depends on how the logging is structured in close_redis_connections
    # Assuming it checks if broker exists before trying to close and log.
    
    # Assert that "Closing Scrape message broker" was NOT called.
    # This requires checking all calls to logger.debug.
    scrape_close_log_found = False
    for call_args in mock_client_logger.debug.call_args_list:
        if "Closing Scrape message broker connection..." in call_args[0][0]:
            scrape_close_log_found = True
            break
    assert not scrape_close_log_found


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client._logger')
async def test_close_redis_connections_broker_close_raises_exception(mock_client_logger, mock_app_config):
    mock_api_broker_instance = AsyncMock(spec=RedisMessageBroker)
    mock_api_broker_instance.close = AsyncMock(side_effect=Exception("Redis close error"))
    client_module._api_message_broker = mock_api_broker_instance
    client_module._scrape_message_broker = None

    await client_module.close_redis_connections() # Should not raise exception itself

    mock_api_broker_instance.close.assert_called_once()
    mock_client_logger.error.assert_called_once()
    args, _ = mock_client_logger.error.call_args
    assert "Error closing API message broker connection: Redis close error" in args[0]
    mock_broker.publish_message.assert_called_once()

# --- Tests for Scenario 4: get_scrape_result ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
async def test_get_scrape_result_completed(mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_result_completed_111"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    # Simulate get_job_status returning a completed status with results
    scrape_results_data = ["Scraped content 1", "Scraped content 2"]
    completed_status = {
        "job_id": job_id, 
        "status": "completed", 
        "payload": {"scraped_content": scrape_results_data} # Assuming result is in payload.scraped_content
    }
    mock_broker.get_job_status = AsyncMock(return_value=completed_status)
    mock_get_scrape_broker.return_value = mock_broker

    result = await client_module.get_scrape_result(job_id)

    assert result == scrape_results_data
    mock_get_scrape_broker.assert_called_once()
    mock_broker.get_job_status.assert_called_once_with(job_id)


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
async def test_get_scrape_result_processing(mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_result_processing_222"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    processing_status = {"job_id": job_id, "status": "processing"}
    mock_broker.get_job_status = AsyncMock(return_value=processing_status)
    mock_get_scrape_broker.return_value = mock_broker

    result = await client_module.get_scrape_result(job_id)

    assert result is None # Should return None if not completed
    mock_broker.get_job_status.assert_called_once_with(job_id)


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
async def test_get_scrape_result_failed(mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_result_failed_333"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    failed_status = {"job_id": job_id, "status": "failed", "error_message": "Scraping failed"}
    mock_broker.get_job_status = AsyncMock(return_value=failed_status)
    mock_get_scrape_broker.return_value = mock_broker

    result = await client_module.get_scrape_result(job_id)

    assert result is None # Should return None if failed
    mock_broker.get_job_status.assert_called_once_with(job_id)


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
async def test_get_scrape_result_no_status_found(mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_result_no_status_444"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    mock_broker.get_job_status = AsyncMock(return_value=None) # Simulate job status not found
    mock_get_scrape_broker.return_value = mock_broker

    result = await client_module.get_scrape_result(job_id)

    assert result is None
    mock_broker.get_job_status.assert_called_once_with(job_id)


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
async def test_get_scrape_result_completed_no_payload(mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_result_completed_no_payload_555"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    completed_status_no_payload = {"job_id": job_id, "status": "completed"} # Missing 'payload'
    mock_broker.get_job_status = AsyncMock(return_value=completed_status_no_payload)
    mock_get_scrape_broker.return_value = mock_broker

    result = await client_module.get_scrape_result(job_id)
    assert result is None


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
async def test_get_scrape_result_completed_no_scraped_content(mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_result_completed_no_content_666"
    mock_broker = AsyncMock(spec=RedisMessageBroker)
    
    completed_status_no_content = {
        "job_id": job_id, 
        "status": "completed", 
        "payload": {} # Missing 'scraped_content' in payload
    }
    mock_broker.get_job_status = AsyncMock(return_value=completed_status_no_content)
    mock_get_scrape_broker.return_value = mock_broker

    result = await client_module.get_scrape_result(job_id)
    assert result is None


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.client.get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_publish_raises_exception(mock_uuid4, mock_get_scrape_broker, mock_app_config):
    job_id = "scrape_job_publish_exception_111"
    mock_uuid4.return_value = MagicMock(hex=job_id)

    mock_broker = AsyncMock(spec=RedisMessageBroker)
    mock_broker.publish_message = AsyncMock(side_effect=Exception("Redis dead for scrape"))
    mock_get_scrape_broker.return_value = mock_broker

    urls_to_scrape = ["http://example.com/scrape_publish_exception"]
    
    result = await client_module.queue_scrape_request(urls_to_scrape, wait_for_result=False)

    assert result is None
    mock_broker.publish_message.assert_called_once()
