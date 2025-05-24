import pytest
import asyncio
import signal
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming the module is viralStoryGenerator.src.scrape_worker
from viralStoryGenerator.src import scrape_worker as scrape_worker_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For type hinting and spec for mocks
from viralStoryGenerator.utils.models import URLMetadata # For type hinting if needed

# --- Mocks & Fixtures ---

@pytest.fixture
def mock_app_config_scraper(monkeypatch):
    """Fixture to provide a mock app_config for scrape_worker tests."""
    monkeypatch.setattr(app_config.redis, 'HOST', "mock_redis_host_scrape")
    monkeypatch.setattr(app_config.redis, 'PORT', 6379)
    monkeypatch.setattr(app_config.redis, 'SCRAPER_JOB_STREAM_NAME', "test_scrape_jobs")
    monkeypatch.setattr(app_config.redis, 'PROCESSING_JOB_STREAM_NAME', "test_processing_jobs")
    monkeypatch.setattr(app_config.redis, 'CONSUMER_GROUP_NAME', "test_scrape_group") # Assuming a common group name prefix
    monkeypatch.setattr(app_config.redis, 'WORKER_BATCH_SIZE', 1) # Scraper typically processes one job at a time
    monkeypatch.setattr(app_config.redis, 'WORKER_SLEEP_INTERVAL', 0.01)
    
    # Scraper specific configs
    monkeypatch.setattr(app_config.scraper, 'HEADLESS_BROWSER', True)
    monkeypatch.setattr(app_config.scraper, 'USER_AGENT', "TestScraperAgent/1.0")
    monkeypatch.setattr(app_config.scraper, 'BROWSER_TIMEOUT_SECONDS', 15)
    monkeypatch.setattr(app_config.scraper, 'MAX_DEPTH_SCRAPING', 1)
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', True)
    return app_config

@pytest.fixture
def mock_broker_instance_scraper():
    """Provides a fresh MagicMock for RedisMessageBroker for each test."""
    broker = MagicMock(spec=RedisMessageBroker)
    broker.initialize = AsyncMock()
    broker.consume_messages = AsyncMock()
    broker.acknowledge_message = AsyncMock()
    broker.track_job_progress = AsyncMock()
    broker.publish_message = AsyncMock() # For publishing to processing stream
    broker.close = AsyncMock()
    return broker

@pytest.fixture(autouse=True)
def reset_scrape_worker_globals(monkeypatch):
    """Reset global instances in scrape_worker.py before each test."""
    if hasattr(scrape_worker_module, '_message_broker'):
        monkeypatch.setattr(scrape_worker_module, '_message_broker', None)
    if hasattr(scrape_worker_module, '_shared_crawler_instance'):
        monkeypatch.setattr(scrape_worker_module, '_shared_crawler_instance', None)
    if hasattr(scrape_worker_module, 'shutdown_event'): # Assuming scrape_worker has its own shutdown_event
         scrape_worker_module.shutdown_event.clear()


# --- Tests for Scenario 1: get_shared_crawler_instance & close_shared_crawler_instance ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.AsyncWebCrawler') # Patch where AsyncWebCrawler is imported
@patch('viralStoryGenerator.src.scrape_worker.BrowserConfig')   # Patch where BrowserConfig is imported
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_get_shared_crawler_instance_creates_new(
    mock_logger, MockBrowserConfig, MockAsyncWebCrawler, mock_app_config_scraper, monkeypatch
):
    # Ensure CRAWL4AI_AVAILABLE is True for this test to use the real classes (mocked here)
    monkeypatch.setattr(scrape_worker_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(scrape_worker_module, '_shared_crawler_instance', None) # Ensure it's reset

    mock_crawler_instance = AsyncMock() # Mock instance of AsyncWebCrawler
    MockAsyncWebCrawler.return_value = mock_crawler_instance
    
    mock_browser_config_instance = MagicMock()
    MockBrowserConfig.return_value = mock_browser_config_instance

    crawler = await scrape_worker_module.get_shared_crawler_instance()

    assert crawler is mock_crawler_instance
    MockBrowserConfig.assert_called_once_with(
        headless=app_config.scraper.HEADLESS_BROWSER,
        user_agent=app_config.scraper.USER_AGENT,
        browser_timeout=app_config.scraper.BROWSER_TIMEOUT_SECONDS * 1000 # ms
    )
    MockAsyncWebCrawler.assert_called_once_with(config=mock_browser_config_instance)
    mock_logger.info.assert_any_call("Shared AsyncWebCrawler instance created.")


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.AsyncWebCrawler')
@patch('viralStoryGenerator.src.scrape_worker.BrowserConfig')
async def test_get_shared_crawler_instance_returns_existing(
    MockBrowserConfig, MockAsyncWebCrawler, mock_app_config_scraper, monkeypatch
):
    monkeypatch.setattr(scrape_worker_module, 'CRAWL4AI_AVAILABLE', True)
    mock_existing_crawler = AsyncMock()
    monkeypatch.setattr(scrape_worker_module, '_shared_crawler_instance', mock_existing_crawler)

    crawler = await scrape_worker_module.get_shared_crawler_instance()

    assert crawler is mock_existing_crawler
    MockAsyncWebCrawler.assert_not_called()
    MockBrowserConfig.assert_not_called()


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_close_shared_crawler_instance_closes_and_resets(mock_logger, monkeypatch):
    monkeypatch.setattr(scrape_worker_module, 'CRAWL4AI_AVAILABLE', True)
    mock_crawler_to_close = AsyncMock()
    monkeypatch.setattr(scrape_worker_module, '_shared_crawler_instance', mock_crawler_to_close)

    await scrape_worker_module.close_shared_crawler_instance()

    mock_crawler_to_close.close.assert_called_once()
    assert scrape_worker_module._shared_crawler_instance is None
    mock_logger.info.assert_any_call("Shared AsyncWebCrawler instance closed.")


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_close_shared_crawler_instance_no_instance(mock_logger, monkeypatch):
    monkeypatch.setattr(scrape_worker_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(scrape_worker_module, '_shared_crawler_instance', None) # No instance to close

    await scrape_worker_module.close_shared_crawler_instance()

    mock_logger.debug.assert_any_call("No shared AsyncWebCrawler instance to close.")
    # Ensure no error if trying to close a None instance


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_get_shared_crawler_instance_crawl4ai_not_available(mock_logger, monkeypatch):
    monkeypatch.setattr(scrape_worker_module, 'CRAWL4AI_AVAILABLE', False)
    monkeypatch.setattr(scrape_worker_module, '_shared_crawler_instance', None)

    # When CRAWL4AI_AVAILABLE is False, get_shared_crawler_instance should return None
    # and log a warning or info.
    crawler = await scrape_worker_module.get_shared_crawler_instance()
    assert crawler is None
    mock_logger.warning.assert_any_call(
        "Crawl4AI library not available. Shared crawler instance cannot be created. Scraper will use mock data."
    )

# --- Tests for Scenario 2: process_single_scrape_job ---

@pytest.fixture
def mock_scrape_callable(monkeypatch):
    """Mocks the _scrape_callable_to_use in scrape_worker.py."""
    mock_callable = AsyncMock()
    monkeypatch.setattr(scrape_worker_module, '_scrape_callable_to_use', mock_callable)
    return mock_callable

# Scenario 2.1: Successful scrape and publish
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_process_single_scrape_job_successful(
    mock_logger, mock_broker_instance_scraper, mock_scrape_callable, mock_app_config_scraper
):
    message_id = "msg_scrape_success_001"
    job_id = "job_scrape_success_001"
    urls_to_scrape = ["http://example.com/page1"]
    job_data_raw = { # Raw data from Redis (bytes keys/values)
        b"job_id": job_id.encode('utf-8'),
        b"payload": json.dumps({"urls": urls_to_scrape}).encode('utf-8') # Payload is JSON string
    }
    # Decoded job data (what process_single_scrape_job works with after initial parsing)
    decoded_job_payload = {"urls": urls_to_scrape}

    # Mock scraper result
    scraped_data_mock = [
        URLMetadata(url=urls_to_scrape[0], title="Title 1", markdown_content="Content 1", error_message=None)
    ]
    mock_scrape_callable.return_value = scraped_data_mock
    
    # Mock broker's publish_message for the processing stream
    processing_message_id = "proc_msg_id_001"
    mock_broker_instance_scraper.publish_message.return_value = processing_message_id

    await scrape_worker_module.process_single_scrape_job(
        mock_broker_instance_scraper, message_id, job_data_raw, "test_scrape_group", "test_scraper_consumer"
    )

    # Verify _scrape_callable_to_use call
    mock_scrape_callable.assert_called_once_with(
        urls=urls_to_scrape,
        job_id=job_id,
        crawler=None, # Default when USE_SHARED_CRAWLER_INSTANCE is False or crawler not passed
        max_depth=app_config.scraper.MAX_DEPTH_SCRAPING,
        include_metadata=True # Default for the scraper utility
    )
    
    # Verify track_job_progress calls
    mock_broker_instance_scraper.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Scraping URLs"}, publish_to_live=True
    )
    final_status_call = mock_broker_instance_scraper.track_job_progress.call_args_list[-1]
    assert final_status_call[0][0] == job_id
    assert final_status_call[0][1]["status"] == "completed"
    assert len(final_status_call[0][1]["results"]) == 1
    assert final_status_call[0][1]["results"][0]["title"] == "Title 1"
    assert final_status_call[0][2] is True # publish_to_live

    # Verify publish_message to processing stream
    mock_broker_instance_scraper.publish_message.assert_called_once()
    args_publish, _ = mock_broker_instance_scraper.publish_message.call_args
    published_to_processing_payload = args_publish[0] # First positional arg is payload
    target_stream_for_publish = args_publish[1] # Second positional arg is routing_key (stream name)

    assert target_stream_for_publish == app_config.redis.PROCESSING_JOB_STREAM_NAME
    assert published_to_processing_payload["job_id"] == job_id
    assert published_to_processing_payload["message_type"] == "scraping_completed"
    assert published_to_processing_payload["original_request_payload"] == decoded_job_payload # Original payload part
    assert len(published_to_processing_payload["scraped_data"]) == 1
    assert published_to_processing_payload["scraped_data"][0]["title"] == "Title 1"
    
    mock_logger.info.assert_any_call(
        f"Scraping job {job_id} completed. Results published to processing stream with message ID: {processing_message_id}"
    )


# Scenario 2.2: scrape_urls_efficiently returns empty list or partial errors
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_process_single_scrape_job_partial_error_or_empty(
    mock_logger, mock_broker_instance_scraper, mock_scrape_callable, mock_app_config_scraper
):
    message_id = "msg_scrape_partial_002"
    job_id = "job_scrape_partial_002"
    urls_to_scrape = ["http://example.com/page1", "http://example.com/badpage"]
    job_data_raw = {b"job_id": job_id.encode('utf-8'), b"payload": json.dumps({"urls": urls_to_scrape}).encode('utf-8')}

    # Simulate one success, one error
    scraped_data_mock = [
        URLMetadata(url=urls_to_scrape[0], title="Title OK", markdown_content="Content OK", error_message=None),
        URLMetadata(url=urls_to_scrape[1], title=None, markdown_content=None, error_message="Failed to fetch badpage")
    ]
    mock_scrape_callable.return_value = scraped_data_mock
    mock_broker_instance_scraper.publish_message.return_value = "proc_msg_id_002"


    await scrape_worker_module.process_single_scrape_job(
        mock_broker_instance_scraper, message_id, job_data_raw, "group", "consumer"
    )

    mock_scrape_callable.assert_called_once()
    
    # Verify track_job_progress status is "completed_with_errors"
    final_status_call = mock_broker_instance_scraper.track_job_progress.call_args_list[-1]
    assert final_status_call[0][0] == job_id
    assert final_status_call[0][1]["status"] == "completed_with_errors"
    assert len(final_status_call[0][1]["results"]) == 2
    assert final_status_call[0][1]["results"][1]["error_message"] == "Failed to fetch badpage"
    
    # Verify publish_message payload reflects partial success
    mock_broker_instance_scraper.publish_message.assert_called_once()
    published_payload = mock_broker_instance_scraper.publish_message.call_args[0][0]
    assert len(published_payload["scraped_data"]) == 2
    assert published_payload["scraped_data"][1]["error_message"] == "Failed to fetch badpage"
    
    mock_logger.warning.assert_any_call(
        f"Scraping job {job_id} completed with some errors. Check results for details."
    )


# Scenario 2.3: scrape_urls_efficiently raises an exception
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_process_single_scrape_job_scraper_exception(
    mock_logger, mock_broker_instance_scraper, mock_scrape_callable, mock_app_config_scraper
):
    message_id = "msg_scrape_exc_003"
    job_id = "job_scrape_exc_003"
    urls_to_scrape = ["http://example.com/page_exc"]
    job_data_raw = {b"job_id": job_id.encode('utf-8'), b"payload": json.dumps({"urls": urls_to_scrape}).encode('utf-8')}
    
    scraper_exception_message = "Max retries exceeded with page_exc"
    mock_scrape_callable.side_effect = Exception(scraper_exception_message)
    mock_broker_instance_scraper.publish_message.return_value = "proc_msg_id_003" # For the error message

    await scrape_worker_module.process_single_scrape_job(
        mock_broker_instance_scraper, message_id, job_data_raw, "group", "consumer"
    )

    mock_scrape_callable.assert_called_once()
    
    # Verify track_job_progress status is "failed"
    mock_broker_instance_scraper.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Scraping URLs"}, publish_to_live=True # Initial processing update
    )
    final_status_call = mock_broker_instance_scraper.track_job_progress.call_args_list[-1]
    assert final_status_call[0][0] == job_id
    assert final_status_call[0][1]["status"] == "failed"
    assert scraper_exception_message in final_status_call[0][1]["error_message"]
    
    # Verify publish_message to processing stream with error info
    mock_broker_instance_scraper.publish_message.assert_called_once()
    published_payload = mock_broker_instance_scraper.publish_message.call_args[0][0]
    assert published_payload["job_id"] == job_id
    assert published_payload["message_type"] == "scraping_failed" # Specific error type
    assert scraper_exception_message in published_payload["error_message"]
    assert "scraped_data" not in published_payload # Or empty list
    
    mock_logger.error.assert_any_call(
        f"Exception during scraping for job {job_id}: {scraper_exception_message}"
    )


# Scenario 2.4: Invalid job data (missing urls or job_id)
@pytest.mark.asyncio
@pytest.mark.parametrize("bad_raw_data, expected_log_fragment", [
    ({b"payload": json.dumps({"urls": ["url1"]}).encode('utf-8')}, "Missing 'job_id'"), # Missing job_id
    ({b"job_id": b"job_no_payload"}, "Missing 'payload'"), # Missing payload
    ({b"job_id": b"job_no_urls", b"payload": json.dumps({}).encode('utf-8')}, "Missing 'urls' in payload"), # Missing urls in payload
    ({b"job_id": b"job_urls_not_list", b"payload": json.dumps({"urls": "not_a_list"}).encode('utf-8')}, "'urls' is not a list"), # urls not a list
    (b"not_a_dict_at_all", "Message data is not a dictionary"), # Raw data not a dict
])
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_process_single_scrape_job_invalid_data(
    mock_logger, mock_broker_instance_scraper, mock_scrape_callable, mock_app_config_scraper,
    bad_raw_data, expected_log_fragment
):
    message_id = "msg_scrape_invalid_004"
    # job_id might be missing from bad_raw_data, so use a default for logging checks if needed
    job_id_for_logging = bad_raw_data.get(b'job_id', b'unknown_job_id').decode('utf-8', 'replace')


    result = await scrape_worker_module.process_single_scrape_job(
        mock_broker_instance_scraper, message_id, bad_raw_data, "group", "consumer"
    )

    assert result is False # Should indicate failure to process
    mock_scrape_callable.assert_not_called() # Scraper should not be called
    
    # track_job_progress should be called with "failed" status IF job_id was parseable
    if b"job_id" in bad_raw_data and isinstance(bad_raw_data, dict):
        final_status_call = mock_broker_instance_scraper.track_job_progress.call_args_list[-1]
        assert final_status_call[0][0] == job_id_for_logging
        assert final_status_call[0][1]["status"] == "failed"
        assert expected_log_fragment in final_status_call[0][1]["error_message"]
    else: # If job_id itself cannot be parsed, track_job_progress might not be called with job_id
        # It might be called with a placeholder or not at all for this specific error.
        # Based on current code, it tries to get job_id first for logging.
        # If job_id is missing, it logs "Missing 'job_id'", then acks. No tracking.
        mock_broker_instance_scraper.track_job_progress.assert_not_called()


    # publish_message to processing stream should NOT be called for these validation errors
    mock_broker_instance_scraper.publish_message.assert_not_called()
    
    # Check for specific error log
    log_found = False
    for call in mock_logger.error.call_args_list:
        if expected_log_fragment in call[0][0]:
            log_found = True
            break
    assert log_found, f"Expected log fragment '{expected_log_fragment}' not found."
    
    # Message should still be acknowledged
    mock_broker_instance_scraper.acknowledge_message.assert_called_once_with("group", message_id)

# --- Tests for Scenario 3: consume_scrape_jobs ---

# Scenario 3.1: Consumes and processes a valid scrape job
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.get_message_broker')
@patch('viralStoryGenerator.src.scrape_worker.process_single_scrape_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock) # To control loop delays
async def test_consume_scrape_jobs_consumes_and_processes(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_scraper, mock_app_config_scraper # Use fixtures
):
    mock_get_broker.return_value = mock_broker_instance_scraper
    
    message_id = "msg_consume_scrape_001"
    job_data_raw = {b"job_id": b"job_consume_scrape_001"} # Minimal raw data
    
    consume_call_count = 0
    async def consume_side_effect_scrape(batch_size, block_ms=None):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1:
            # The consume_messages in RedisMessageBroker returns:
            # stream_name, [(message_id, message_data_dict_bytes_keys_values)]
            # But the loop in consume_scrape_jobs expects:
            # for message_id, job_data_raw in messages_data:
            # So, the mock should return a list of (message_id, job_data_raw)
            return [(message_id, job_data_raw)]
        scrape_worker_module.shutdown_event.set() # After first batch, signal shutdown
        return []
    mock_broker_instance_scraper.consume_messages.side_effect = consume_side_effect_scrape
    
    mock_process_single_job.return_value = True # Simulate successful processing

    scrape_worker_module.shutdown_event.clear()
    consumer_id = "test_scraper_consumer_01"
    group_name = mock_app_config_scraper.redis.CONSUMER_GROUP_NAME # Group name used by consumer

    await scrape_worker_module.consume_scrape_jobs(group_name, consumer_id)

    mock_broker_instance_scraper.initialize.assert_called_once()
    assert mock_broker_instance_scraper.consume_messages.call_count >= 1
    
    mock_process_single_job.assert_called_once_with(
        mock_broker_instance_scraper,
        message_id,
        job_data_raw,
        group_name,
        consumer_id
    )
    # Based on loop structure, if a message is processed, sleep is not called immediately after.
    # It's called when `if not messages:` is true.
    # Here, first call gets messages, second call (after processing) gets empty and sets shutdown.
    # So, sleep should be called once after the empty batch.
    if consume_call_count > 1: # if it looped after processing the first message
         mock_asyncio_sleep.assert_called_once_with(mock_app_config_scraper.redis.WORKER_SLEEP_INTERVAL)
    
    mock_broker_instance_scraper.close.assert_called_once()


# Scenario 3.2: Handles empty message batch
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.get_message_broker')
@patch('viralStoryGenerator.src.scrape_worker.process_single_scrape_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_consume_scrape_jobs_empty_batch(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_scraper, mock_app_config_scraper
):
    mock_get_broker.return_value = mock_broker_instance_scraper
    
    consume_call_count = 0
    async def consume_empty_side_effect_scrape(batch_size, block_ms=None):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count >= 2: # Return empty a few times then shutdown
            scrape_worker_module.shutdown_event.set()
        return [] # Always empty
    mock_broker_instance_scraper.consume_messages.side_effect = consume_empty_side_effect_scrape
    
    scrape_worker_module.shutdown_event.clear()
    consumer_id = "test_scraper_consumer_empty"
    group_name = mock_app_config_scraper.redis.CONSUMER_GROUP_NAME

    await scrape_worker_module.consume_scrape_jobs(group_name, consumer_id)

    assert mock_broker_instance_scraper.consume_messages.call_count >= 1
    mock_process_single_job.assert_not_called()
    mock_asyncio_sleep.assert_any_call(mock_app_config_scraper.redis.WORKER_SLEEP_INTERVAL)
    mock_logger.debug.assert_any_call(
        f"Consumer {consumer_id}: No scrape jobs received, sleeping for {mock_app_config_scraper.redis.WORKER_SLEEP_INTERVAL}s..."
    )
    mock_broker_instance_scraper.close.assert_called_once()


# Scenario 3.3: Graceful shutdown (already implicitly tested by use of shutdown_event)

# Scenario 3.4: Error handling for invalid message structure from Redis
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.get_message_broker')
@patch('viralStoryGenerator.src.scrape_worker.process_single_scrape_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_consume_scrape_jobs_invalid_message_structure(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_scraper, mock_app_config_scraper
):
    mock_get_broker.return_value = mock_broker_instance_scraper
    
    message_id_invalid = "msg_invalid_struct_scrape_001"
    # Example of invalid structure: consume_messages returning something other than list of tuples
    invalid_messages_data = "This is not a list of messages" 
    
    consume_call_count = 0
    async def consume_invalid_struct_side_effect(batch_size, block_ms=None):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1:
            return invalid_messages_data # Return invalid structure
        scrape_worker_module.shutdown_event.set()
        return [] # Then empty to allow shutdown
    mock_broker_instance_scraper.consume_messages.side_effect = consume_invalid_struct_side_effect
    
    scrape_worker_module.shutdown_event.clear()
    consumer_id = "test_scraper_consumer_invalid_struct"
    group_name = mock_app_config_scraper.redis.CONSUMER_GROUP_NAME

    await scrape_worker_module.consume_scrape_jobs(group_name, consumer_id)

    mock_process_single_job.assert_not_called() # Should not be called if message list itself is bad
    # Error should be logged by consume_scrape_jobs itself
    log_found = False
    for call in mock_logger.error.call_args_list:
        if f"Consumer {consumer_id}: Error processing messages from Redis stream" in call[0][0] and \
           isinstance(call[0][1], TypeError): # Expecting a TypeError due to bad iteration
            log_found = True
            break
    assert log_found, "Expected error log for invalid message structure not found or incorrect."
    
    # Sleep should be called after the error, before shutdown
    mock_asyncio_sleep.assert_called_once_with(mock_app_config_scraper.redis.WORKER_SLEEP_INTERVAL)
    mock_broker_instance_scraper.close.assert_called_once()

# --- Tests for Scenario 4: main_async and main (overall orchestration) ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.preload_components', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker.consume_scrape_jobs', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker.close_shared_crawler_instance', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker.get_message_broker') # To mock the broker for close check
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_main_async_happy_path(
    mock_logger, mock_get_broker, mock_close_crawler, 
    mock_consume_jobs, mock_preload, mock_broker_instance_scraper, # broker fixture
    mock_app_config_scraper # app_config fixture
):
    # Simulate successful execution
    mock_get_broker.return_value = mock_broker_instance_scraper # get_message_broker returns our mock broker
    mock_consume_jobs.return_value = None # consume_scrape_jobs runs and completes

    await scrape_worker_module.main_async()

    mock_preload.assert_called_once()
    
    # consume_scrape_jobs is called with group_name and a generated consumer_id
    mock_consume_jobs.assert_called_once()
    args, _ = mock_consume_jobs.call_args
    assert args[0] == mock_app_config_scraper.redis.CONSUMER_GROUP_NAME # Group name
    assert isinstance(args[1], str) # Consumer ID (string)
    assert f"{mock_app_config_scraper.redis.SCRAPER_JOB_STREAM_NAME}_consumer_" in args[1] # Default naming convention

    # Verify cleanup calls
    mock_close_crawler.assert_called_once()
    mock_broker_instance_scraper.close.assert_called_once() # Ensure broker from get_message_broker is closed
    mock_logger.info.assert_any_call("Scrape worker main task completed.")


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.preload_components', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker.consume_scrape_jobs', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker.close_shared_crawler_instance', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker.get_message_broker')
@patch('viralStoryGenerator.src.scrape_worker._logger')
async def test_main_async_consume_jobs_raises_exception(
    mock_logger, mock_get_broker, mock_close_crawler, 
    mock_consume_jobs, mock_preload, mock_broker_instance_scraper,
    mock_app_config_scraper
):
    mock_get_broker.return_value = mock_broker_instance_scraper
    consume_exception = Exception("Error during job consumption")
    mock_consume_jobs.side_effect = consume_exception

    # main_async should catch the exception and still perform cleanup
    await scrape_worker_module.main_async()

    mock_preload.assert_called_once()
    mock_consume_jobs.assert_called_once()
    
    mock_logger.error.assert_called_once_with(f"Scrape worker main task encountered an error: {consume_exception}", exc_info=True)
    
    # Cleanup should still be called
    mock_close_crawler.assert_called_once()
    mock_broker_instance_scraper.close.assert_called_once()


@patch('viralStoryGenerator.src.scrape_worker.asyncio.run') # Patch asyncio.run used by main()
@patch('viralStoryGenerator.src.scrape_worker.main_async', new_callable=AsyncMock) # Mock the async main part
@patch('viralStoryGenerator.src.scrape_worker.signal.signal')
@patch('viralStoryGenerator.src.scrape_worker._logger')
def test_main_function_sets_up_signals_and_runs_main_async(
    mock_logger, mock_signal_signal, mock_main_async_func, mock_asyncio_run,
    mock_app_config_scraper # For app_config access if needed by main directly
):
    # Simulate main_async raising KeyboardInterrupt to test that path in main()
    mock_main_async_func.side_effect = KeyboardInterrupt("Simulated Ctrl+C in main_async")

    # Call the synchronous main() function
    scrape_worker_module.main()

    # Verify signal handlers
    mock_signal_signal.assert_any_call(signal.SIGINT, scrape_worker_module.shutdown_handler)
    mock_signal_signal.assert_any_call(signal.SIGTERM, scrape_worker_module.shutdown_handler)
    
    # Verify asyncio.run was called with main_async
    mock_asyncio_run.assert_called_once_with(mock_main_async_func())
    
    # Verify logging from main()
    mock_logger.info.assert_any_call("ViralStoryGenerator Scraper Worker starting...")
    mock_logger.info.assert_any_call("Scraper worker shut down by KeyboardInterrupt.")


def test_scrape_worker_shutdown_handler(mock_app_config_scraper, monkeypatch):
    # Ensure shutdown_event is accessible and can be cleared/set
    # If scrape_worker.shutdown_event is not defined, this test needs adjustment
    # For now, assume it's defined similarly to queue_worker
    event_mock = MagicMock(spec=asyncio.Event)
    event_mock.is_set.return_value = False
    monkeypatch.setattr(scrape_worker_module, 'shutdown_event', event_mock)
    
    assert not scrape_worker_module.shutdown_event.is_set()
    
    with patch.object(scrape_worker_module._logger, 'info') as mock_log_info:
        scrape_worker_module.shutdown_handler(signal.SIGINT, None) 
        assert scrape_worker_module.shutdown_event.set.called_once()
        mock_log_info.assert_any_call("Shutdown signal received. Initiating graceful shutdown...")

    # Test again to ensure it's idempotent or handles being called multiple times
    scrape_worker_module.shutdown_event.set.reset_mock() # Reset for second call check
    scrape_worker_module.shutdown_handler(signal.SIGINT, None)
    # Depending on implementation, set might be called again or skipped if already set.
    # If it's just `shutdown_event.set()`, it's fine to be called multiple times.
    assert scrape_worker_module.shutdown_event.set.called # At least once from this second call too


# Scenario 3.5: Error handling for Redis connection issues during consumption
@pytest.mark.asyncio
@patch('viralStoryGenerator.src.scrape_worker.get_message_broker')
@patch('viralStoryGenerator.src.scrape_worker.process_single_scrape_job', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.scrape_worker._logger')
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_consume_scrape_jobs_consume_exception(
    mock_asyncio_sleep, mock_logger, mock_process_single_job, mock_get_broker,
    mock_broker_instance_scraper, mock_app_config_scraper
):
    mock_get_broker.return_value = mock_broker_instance_scraper
    
    consume_exception_message = "Simulated Redis connection error during consume"
    consume_call_count = 0
    async def consume_exception_side_effect(batch_size, block_ms=None):
        nonlocal consume_call_count
        consume_call_count += 1
        if consume_call_count == 1:
            raise Exception(consume_exception_message)
        scrape_worker_module.shutdown_event.set() # Allow shutdown on next iteration
        return []
    mock_broker_instance_scraper.consume_messages.side_effect = consume_exception_side_effect
    
    scrape_worker_module.shutdown_event.clear()
    consumer_id = "test_scraper_consumer_conn_err"
    group_name = mock_app_config_scraper.redis.CONSUMER_GROUP_NAME

    await scrape_worker_module.consume_scrape_jobs(group_name, consumer_id)

    assert mock_broker_instance_scraper.consume_messages.call_count >= 1
    mock_process_single_job.assert_not_called()
    
    log_found = False
    for call in mock_logger.error.call_args_list:
        if f"Consumer {consumer_id}: Error consuming messages from Redis stream" in call[0][0] and \
           consume_exception_message in str(call[0][1]):
            log_found = True
            break
    assert log_found, "Expected error log for consume_messages exception not found or incorrect."
    
    mock_asyncio_sleep.assert_called_once_with(mock_app_config_scraper.redis.WORKER_SLEEP_INTERVAL)
    mock_broker_instance_scraper.close.assert_called_once()
