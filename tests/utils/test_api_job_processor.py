import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, call

# Assuming the module is viralStoryGenerator.utils.api_job_processor
from viralStoryGenerator.utils import api_job_processor as ajp_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For type hinting and spec for mocks
from viralStoryGenerator.utils.models import StoryboardData, StoryboardScene, URLMetadata # For type hints/obj creation

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_api_processor_defaults(monkeypatch):
    """Set default app_config values for api_job_processor tests."""
    # RAG settings
    monkeypatch.setattr(app_config.rag, 'ENABLED', True)
    monkeypatch.setattr(app_config.rag, 'MAX_CHARS_BEFORE_RAG', 1000) # Example value
    monkeypatch.setattr(app_config.rag, 'CHUNK_SIZE', 500)
    monkeypatch.setattr(app_config.rag, 'CHUNK_OVERLAP', 50)
    monkeypatch.setattr(app_config.rag, 'NUM_RELEVANT_CHUNKS', 3)
    monkeypatch.setattr(app_config.rag, 'RAG_RELEVANCE_THRESHOLD', 0.7)
    
    # LLM settings (used by process_with_llm directly)
    # These are usually set in llm module tests, but also relevant here if process_with_llm is called
    monkeypatch.setattr(app_config.llm, 'MODEL', "mock_llm_processor_model")
    monkeypatch.setattr(app_config.llm, 'STORY_SYSTEM_PROMPT', "System prompt for story: {context}")
    monkeypatch.setattr(app_config.llm, 'STORY_USER_PROMPT_TEMPLATE', "User prompt for story: {topic} with urls: {urls}")
    monkeypatch.setattr(app_config.llm, 'STORY_USER_PROMPT_TEMPLATE_NO_URLS', "User prompt for story (no urls): {topic}")


    # Storyboard settings
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', True)
    
    # Storage settings
    monkeypatch.setattr(app_config.storage, 'S3_PUBLIC_URL_PREFIX', "http://s3.mock.com")

    # Default job type if not specified in message
    monkeypatch.setattr(app_config.worker, 'DEFAULT_JOB_TYPE', "generate_story_and_video")
    return app_config

@pytest.fixture
def mock_broker_passed_to_processor():
    """Provides a MagicMock for the RedisMessageBroker instance passed to process_api_job."""
    broker = MagicMock(spec=RedisMessageBroker)
    broker.track_job_progress = AsyncMock()
    return broker

@pytest.fixture
def mock_ajp_logger():
    """Fixture to mock the _logger in api_job_processor.py."""
    with patch('viralStoryGenerator.utils.api_job_processor._logger') as mock_logger:
        yield mock_logger

# --- Dependency Mocks for process_api_job ---
# These will be patched directly in each test where process_api_job is called

@pytest.fixture
def mock_scraper_client(monkeypatch):
    mock_qsr = AsyncMock()
    mock_gsr = AsyncMock()
    monkeypatch.setattr(ajp_module.crawl4ai_scraper, 'queue_scrape_request', mock_qsr)
    monkeypatch.setattr(ajp_module.crawl4ai_scraper, 'get_scrape_result', mock_gsr)
    return {'queue': mock_qsr, 'get': mock_gsr}

@pytest.fixture
def mock_llm_utils(monkeypatch):
    mock_pwm = AsyncMock() # process_with_llm
    mock_cmwl = MagicMock() # clean_markdown_with_llm (sync)
    monkeypatch.setattr(ajp_module.llm, 'process_with_llm', mock_pwm)
    monkeypatch.setattr(ajp_module.llm, 'clean_markdown_with_llm', mock_cmwl)
    return {'process': mock_pwm, 'clean': mock_cmwl}

@pytest.fixture
def mock_storyboard_utils(monkeypatch):
    mock_gs = AsyncMock() # generate_storyboard
    monkeypatch.setattr(ajp_module.storyboard, 'generate_storyboard', mock_gs)
    return mock_gs

@pytest.fixture
def mock_storage_utils(monkeypatch):
    mock_sf = AsyncMock() # store_file
    monkeypatch.setattr(ajp_module.storage_manager, 'store_file', mock_sf)
    return mock_sf

@pytest.fixture
def mock_vector_db_utils(monkeypatch):
    mock_add = AsyncMock()
    mock_query = AsyncMock()
    monkeypatch.setattr(ajp_module.vector_db_manager, 'add_chunks_to_collection', mock_add)
    monkeypatch.setattr(ajp_module.vector_db_manager, 'query_collection', mock_query)
    return {'add': mock_add, 'query': mock_query}

@pytest.fixture
def mock_text_processing_utils(monkeypatch):
    mock_split = MagicMock() # split_text_into_chunks (sync)
    monkeypatch.setattr(ajp_module.text_processing, 'split_text_into_chunks', mock_split)
    return mock_split


# --- Test Scenarios for process_api_job ---

# Scenario 1: Successful end-to-end story generation (RAG enabled, storyboard enabled)
@pytest.mark.asyncio
async def test_process_api_job_success_rag_storyboard_enabled(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, # fixtures
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils, # dep mocks
    mock_vector_db_utils, mock_text_processing_utils # dep mocks
):
    job_id = "job_success_e2e_001"
    job_type = "generate_story_and_video"
    consumer_id = "consumer_e2e_001"
    job_data = {
        "job_id": job_id,
        "job_type": job_type,
        "topic": "A grand adventure",
        "urls": ["http://example.com/source1", "http://example.com/source2"]
    }

    # --- Configure Mock Behaviors ---
    # Scraper
    scrape_job_id = f"scrape_{job_id}"
    mock_scraper_client['queue'].return_value = scrape_job_id
    scraped_content_list = [
        URLMetadata(url="http://example.com/source1", title="Source 1", markdown_content="Markdown content from source 1, long enough for RAG.", error_message=None),
        URLMetadata(url="http://example.com/source2", title="Source 2", markdown_content="More markdown from source 2, also long.", error_message=None)
    ]
    mock_scraper_client['get'].return_value = scraped_content_list
    
    # Content Cleaning
    cleaned_content_str = "Cleaned combined markdown content. " * 20 # Ensure it's long
    mock_llm_utils['clean'].return_value = cleaned_content_str
    
    # RAG - Text Processing
    text_chunks = ["chunk1_of_cleaned", "chunk2_of_cleaned", "chunk3_of_cleaned"]
    mock_text_processing_utils.return_value = text_chunks
    # RAG - Vector DB
    mock_vector_db_utils['add'].return_value = True
    relevant_chunks = ["chunk1_of_cleaned", "chunk3_of_cleaned"] # Simulate query result
    mock_vector_db_utils['query'].return_value = relevant_chunks
    
    # LLM Story Generation
    generated_story_script = "### Story Script:\nThis is the final generated story script.\n### Video Description:\nDetailed video description."
    llm_thought_process = "LLM thought process for story..."
    mock_llm_utils['process'].return_value = (generated_story_script, llm_thought_process)
    
    # Storage (Story Script)
    story_script_storage_url = f"http://s3.mock.com/{job_id}/story_script.txt"
    mock_storage_utils.side_effect = [ # Order of store_file calls matters
        story_script_storage_url, # For story script
        # storyboard assets next, then final metadata
    ]
    
    # Storyboard Generation
    storyboard_data_mock = StoryboardData(
        job_id=job_id,
        story_script=generated_story_script, # or just the story part
        scenes=[StoryboardScene(scene_number=1, image_prompt="img pr", narration="narr", image_url="s3_img_url", audio_url="s3_audio_url", duration_seconds=5.0)]
    )
    mock_storyboard_utils.return_value = storyboard_data_mock
    
    # Storage (Metadata) - this will be the last store_file call
    final_metadata_storage_url = f"http://s3.mock.com/{job_id}/metadata.json"
    
    # Reconfigure side_effect for store_file to handle multiple calls correctly
    # Store script, then storyboard assets (image, audio per scene), then final metadata
    # For this test, assume 1 scene in storyboard, so 1 image, 1 audio.
    expected_store_calls = [
        story_script_storage_url, # story_script.txt
        "s3_img_url", # scene_1_image.png (from storyboard_data_mock)
        "s3_audio_url", # scene_1_audio.mp3 (from storyboard_data_mock)
        f"http://s3.mock.com/{job_id}/storyboard.json", # storyboard.json
        final_metadata_storage_url  # metadata.json
    ]
    mock_storage_utils.side_effect = expected_store_calls


    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is True

    # Verify Scraper calls
    mock_scraper_client['queue'].assert_called_once_with(job_data["urls"], job_id_prefix="scrape_")
    mock_scraper_client['get'].assert_called_once_with(scrape_job_id)

    # Verify Content Cleaning call
    combined_raw_markdown = "\n\n".join([sm.markdown_content for sm in scraped_content_list if sm.markdown_content])
    mock_llm_utils['clean'].assert_called_once_with(combined_raw_markdown)

    # Verify RAG calls
    mock_text_processing_utils.assert_called_once_with(
        cleaned_content_str, 
        chunk_size=app_config.rag.CHUNK_SIZE, 
        chunk_overlap=app_config.rag.CHUNK_OVERLAP
    )
    mock_vector_db_utils['add'].assert_called_once_with(job_id, text_chunks, scraped_content_list)
    mock_vector_db_utils['query'].assert_called_once_with(
        job_id, 
        job_data["topic"], 
        num_results=app_config.rag.NUM_RELEVANT_CHUNKS,
        relevance_threshold=app_config.rag.RAG_RELEVANCE_THRESHOLD
    )

    # Verify LLM Story Generation call
    mock_llm_utils['process'].assert_called_once()
    llm_args, _ = mock_llm_utils['process'].call_args
    assert llm_args[0] == job_data["topic"] # topic
    assert job_data["urls"][0] in llm_args[2] # user_prompt_template (urls included)
    assert "\n---\n".join(relevant_chunks) in llm_args[3] # context_text

    # Verify Storyboard Generation call
    mock_storyboard_utils.assert_called_once_with(job_id, generated_story_script)

    # Verify store_file calls
    assert mock_storage_utils.call_count == len(expected_store_calls)
    mock_storage_utils.assert_any_call(generated_story_script, job_id, "story_script.txt", is_json=False)
    # The storyboard assets are stored by generate_storyboard itself, store_file is mocked for it there.
    # Here we check the final metadata store.
    # The last call to store_file should be for metadata.json
    final_store_call_args = mock_storage_utils.call_args_list[-1]
    assert final_store_call_args[0][1] == job_id # job_id
    assert final_store_call_args[0][2] == "metadata.json" # filename_key
    assert final_store_call_args[1]['is_json'] is True
    final_metadata_payload = final_store_call_args[0][0]
    assert final_metadata_payload['job_id'] == job_id
    assert final_metadata_payload['status'] == "completed"
    assert final_metadata_payload['story_script_url'] == story_script_storage_url
    assert final_metadata_payload['storyboard_data'] == storyboard_data_mock.model_dump()


    # Verify track_job_progress calls
    mock_broker_passed_to_processor.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Scraping URLs..."}, publish_to_live=True
    )
    mock_broker_passed_to_processor.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Generating story script..."}, publish_to_live=True
    )
    mock_broker_passed_to_processor.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Generating storyboard..."}, publish_to_live=True
    )
    # Final "completed" status check
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "completed"
    assert last_track_call_args[0][1]["metadata_url"] == final_metadata_storage_url


# Scenario 2: Successful generation (RAG disabled, storyboard disabled)
@pytest.mark.asyncio
async def test_process_api_job_success_rag_sb_disabled(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils
):
    job_id = "job_success_disabled_002"
    job_type = "generate_story_and_video" # Type implies video, but storyboard flag is off
    consumer_id = "consumer_disabled_002"
    job_data = {
        "job_id": job_id, "job_type": job_type, "topic": "Simple Story, No RAG, No SB",
        # No URLs, so scraper won't be called, RAG won't have external content
    }

    # --- Configure App Config ---
    monkeypatch.setattr(app_config.rag, 'ENABLED', False)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', False)

    # --- Configure Mock Behaviors ---
    # Scraper (won't be called if no URLs)
    mock_scraper_client['queue'].return_value = f"scrape_{job_id}"
    mock_scraper_client['get'].return_value = [] # No scraped content

    # Content Cleaning (will be called on empty string if no URLs/scraped content)
    mock_llm_utils['clean'].return_value = "" # Cleaned empty content

    # LLM Story Generation
    generated_story_script_no_rag = "### Story Script:\nJust a simple story, no RAG.\n### Video Description:\nSimple video desc."
    llm_thought_no_rag = "LLM thought for no RAG..."
    mock_llm_utils['process'].return_value = (generated_story_script_no_rag, llm_thought_no_rag)
    
    # Storage (Story Script)
    story_script_storage_url_no_rag = f"http://s3.mock.com/{job_id}/story_script.txt"
    # Storage (Metadata)
    final_metadata_storage_url_no_rag = f"http://s3.mock.com/{job_id}/metadata.json"
    
    # store_file will be called for script and metadata
    mock_storage_utils.side_effect = [
        story_script_storage_url_no_rag,
        final_metadata_storage_url_no_rag
    ]

    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is True

    # Verify Scraper calls (not called if no URLs)
    mock_scraper_client['queue'].assert_not_called()
    mock_scraper_client['get'].assert_not_called()

    # Verify Content Cleaning call (called with empty string if no content)
    mock_llm_utils['clean'].assert_called_once_with("")

    # Verify RAG functions NOT called
    mock_text_processing_utils.assert_not_called()
    mock_vector_db_utils['add'].assert_not_called()
    mock_vector_db_utils['query'].assert_not_called()

    # Verify LLM Story Generation call
    mock_llm_utils['process'].assert_called_once()
    llm_args, _ = mock_llm_utils['process'].call_args
    assert llm_args[0] == job_data["topic"]
    assert llm_args[3] is None # context_text should be None as RAG is disabled/no content

    # Verify Storyboard Generation NOT called
    mock_storyboard_utils.assert_not_called()

    # Verify store_file calls (script and metadata)
    assert mock_storage_utils.call_count == 2
    mock_storage_utils.assert_any_call(generated_story_script_no_rag, job_id, "story_script.txt", is_json=False)
    
    final_store_call_args = mock_storage_utils.call_args_list[-1]
    assert final_store_call_args[0][1] == job_id
    assert final_store_call_args[0][2] == "metadata.json"
    final_metadata_payload = final_store_call_args[0][0]
    assert final_metadata_payload['status'] == "completed"
    assert final_metadata_payload['story_script_url'] == story_script_storage_url_no_rag
    assert "storyboard_data" not in final_metadata_payload # Storyboard was disabled
    assert "storyboard_url" not in final_metadata_payload

    # Verify track_job_progress calls
    mock_broker_passed_to_processor.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Generating story script..."}, publish_to_live=True
    )
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "completed"
    assert last_track_call_args[0][1]["metadata_url"] == final_metadata_storage_url_no_rag

# Scenario 3: Scraping fails or returns no content

@pytest.mark.asyncio
@pytest.mark.parametrize("scrape_result_scenario", ["returns_none", "returns_empty_list"])
async def test_process_api_job_scrape_fails_or_empty(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils,
    scrape_result_scenario
):
    job_id = f"job_scrape_fail_{scrape_result_scenario}_003"
    job_type = "generate_story_and_video"
    consumer_id = f"consumer_scrape_fail_{scrape_result_scenario}_003"
    job_data = {
        "job_id": job_id, "job_type": job_type, "topic": "Topic Scrape Fail",
        "urls": ["http://example.com/source_to_fail"]
    }

    # --- Configure App Config (RAG and Storyboard can be enabled, but won't be reached) ---
    monkeypatch.setattr(app_config.rag, 'ENABLED', True)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', True)

    # --- Configure Mock Behaviors ---
    scrape_job_id = f"scrape_{job_id}"
    mock_scraper_client['queue'].return_value = scrape_job_id
    
    if scrape_result_scenario == "returns_none":
        mock_scraper_client['get'].return_value = None # Simulate get_scrape_result returning None
        expected_error_fragment = f"Failed to retrieve scrape results for scrape job ID {scrape_job_id}"
    else: # returns_empty_list
        mock_scraper_client['get'].return_value = [] # Simulate get_scrape_result returning empty list
        expected_error_fragment = "No content successfully scraped from provided URLs"
        
    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is False

    # Verify Scraper calls
    mock_scraper_client['queue'].assert_called_once_with(job_data["urls"], job_id_prefix="scrape_")
    mock_scraper_client['get'].assert_called_once_with(scrape_job_id)

    # Verify subsequent functions are NOT called
    mock_llm_utils['clean'].assert_not_called()
    mock_vector_db_utils['add'].assert_not_called()
    mock_vector_db_utils['query'].assert_not_called()
    mock_llm_utils['process'].assert_not_called()
    mock_storyboard_utils.assert_not_called()
    mock_storage_utils.assert_not_called() # No script or metadata should be stored

    # Verify track_job_progress calls
    # Initial "processing" call
    mock_broker_passed_to_processor.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Scraping URLs..."}, publish_to_live=True
    )
    # Final "failed" call
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "failed"
    assert expected_error_fragment in last_track_call_args[0][1]["error_message"]
    
    # Verify logger call for the specific failure
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list:
        if expected_error_fragment in call_arg[0][0]: # Check first arg of the call (the message string)
            error_log_found = True
            break
    assert error_log_found, f"Expected error log containing '{expected_error_fragment}' not found."

# Scenario 4: Content cleaning fails
@pytest.mark.asyncio
@pytest.mark.parametrize("cleaning_failure_mode", ["returns_none", "raises_exception"])
async def test_process_api_job_content_cleaning_fails(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils,
    cleaning_failure_mode
):
    job_id = f"job_clean_fail_{cleaning_failure_mode}_004"
    job_type = "generate_story_and_video"
    consumer_id = f"consumer_clean_fail_{cleaning_failure_mode}_004"
    job_data = {
        "job_id": job_id, "job_type": job_type, "topic": "Topic Clean Fail",
        "urls": ["http://example.com/source_for_cleaning_fail"]
    }

    # --- Configure Mock Behaviors ---
    # Scraper succeeds
    scrape_job_id = f"scrape_{job_id}"
    mock_scraper_client['queue'].return_value = scrape_job_id
    scraped_content_list = [
        URLMetadata(url=job_data["urls"][0], title="Source", markdown_content="Raw markdown to be cleaned.", error_message=None)
    ]
    mock_scraper_client['get'].return_value = scraped_content_list
    
    # Content Cleaning fails
    cleaning_exception_message = "LLM unavailable for cleaning"
    if cleaning_failure_mode == "returns_none":
        mock_llm_utils['clean'].return_value = None
    else: # raises_exception
        mock_llm_utils['clean'].side_effect = Exception(cleaning_exception_message)
        
    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is False

    # Verify Scraper calls
    mock_scraper_client['queue'].assert_called_once()
    mock_scraper_client['get'].assert_called_once()

    # Verify Content Cleaning call
    combined_raw_markdown = "\n\n".join([sm.markdown_content for sm in scraped_content_list if sm.markdown_content])
    mock_llm_utils['clean'].assert_called_once_with(combined_raw_markdown)

    # Verify subsequent functions are NOT called
    mock_text_processing_utils.assert_not_called()
    mock_vector_db_utils['add'].assert_not_called()
    mock_vector_db_utils['query'].assert_not_called()
    mock_llm_utils['process'].assert_not_called() # LLM for story generation
    mock_storyboard_utils.assert_not_called()
    mock_storage_utils.assert_not_called()

    # Verify track_job_progress calls
    mock_broker_passed_to_processor.track_job_progress.assert_any_call(
        job_id, {"status": "processing", "message": "Cleaning scraped content..."}, publish_to_live=True
    )
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "failed"
    expected_error_in_status = "Failed to clean scraped content."
    if cleaning_failure_mode == "raises_exception":
        # Note: The actual error message in status might be generic, check logs for specifics
        assert "Exception during content cleaning" in last_track_call_args[0][1]["error_message"]
    else:
        assert expected_error_in_status in last_track_call_args[0][1]["error_message"]
    
    # Verify logger call
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list:
        if cleaning_failure_mode == "raises_exception":
            if f"Exception during content cleaning for job {job_id}" in call_arg[0][0] and \
               cleaning_exception_message in str(call_arg[0][1]): # Check for original exception in log
                error_log_found = True
                break
        else: # returns_none
            if f"Content cleaning returned None for job {job_id}" in call_arg[0][0]:
                error_log_found = True
                break
    assert error_log_found, "Expected error log for content cleaning failure not found or incorrect."

# Scenario 5: LLM script generation fails
@pytest.mark.asyncio
@pytest.mark.parametrize("llm_failure_mode", ["returns_none", "raises_exception"])
async def test_process_api_job_llm_script_generation_fails(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils,
    llm_failure_mode
):
    job_id = f"job_llm_fail_{llm_failure_mode}_005"
    job_type = "generate_story_and_video"
    consumer_id = f"consumer_llm_fail_{llm_failure_mode}_005"
    job_data = {
        "job_id": job_id, "job_type": job_type, "topic": "Topic LLM Fail",
        "urls": ["http://example.com/source_for_llm_fail"]
    }

    # --- Configure Mock Behaviors ---
    # Scraping and Cleaning succeed
    scrape_job_id = f"scrape_{job_id}"
    mock_scraper_client['queue'].return_value = scrape_job_id
    scraped_content_list = [URLMetadata(url=job_data["urls"][0], title="S", markdown_content="MD", error_message=None)]
    mock_scraper_client['get'].return_value = scraped_content_list
    mock_llm_utils['clean'].return_value = "Cleaned MD content"
    
    # RAG path (assuming enabled, can be simple for this test)
    monkeypatch.setattr(app_config.rag, 'ENABLED', False) # Disable RAG to simplify, focus on LLM script gen

    # LLM Story Generation fails
    llm_exception_message = "LLM model is offline"
    if llm_failure_mode == "returns_none":
        mock_llm_utils['process'].return_value = (None, None) # No script, no thought
    else: # raises_exception
        mock_llm_utils['process'].side_effect = Exception(llm_exception_message)
        
    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is False

    mock_llm_utils['process'].assert_called_once() # LLM for story generation was attempted

    # Verify subsequent functions are NOT called
    mock_storyboard_utils.assert_not_called()
    # store_file might be called for raw_scraped_content.json if that's logged before this failure point
    # For this test, let's assume it's not, or check call_args_list for specific filenames.
    # Based on current flow, story_script.txt is the first main content store after LLM.
    # If store_file for raw content is added, this needs adjustment.
    # For now, assume no store_file calls if LLM fails.
    mock_storage_utils.assert_not_called()


    # Verify track_job_progress calls
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "failed"
    expected_error_in_status = "LLM failed to generate story script."
    if llm_failure_mode == "raises_exception":
        assert "Exception during LLM story generation" in last_track_call_args[0][1]["error_message"]
    else:
        assert expected_error_in_status in last_track_call_args[0][1]["error_message"]
    
    # Verify logger call
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list:
        log_msg = call_arg[0][0]
        if llm_failure_mode == "raises_exception":
            if f"Exception during LLM story generation for job {job_id}" in log_msg and \
               llm_exception_message in str(call_arg[0][1]):
                error_log_found = True
                break
        else: # returns_none
            if f"LLM process returned None for script for job {job_id}" in log_msg:
                error_log_found = True
                break
    assert error_log_found, "Expected error log for LLM script generation failure not found or incorrect."

# Scenario 6: Storing story script fails
@pytest.mark.asyncio
@pytest.mark.parametrize("store_script_failure_mode", ["returns_none", "raises_exception"])
async def test_process_api_job_store_script_fails(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils,
    store_script_failure_mode
):
    job_id = f"job_store_script_fail_{store_script_failure_mode}_006"
    job_type = "generate_story_and_video"
    consumer_id = f"consumer_store_script_fail_{store_script_failure_mode}_006"
    job_data = {"job_id": job_id, "job_type": job_type, "topic": "Topic Store Script Fail"} # No URLs for simplicity

    # --- Configure Mock Behaviors ---
    # Scraper and Cleaning not critical if no URLs, RAG disabled for simplicity
    monkeypatch.setattr(app_config.rag, 'ENABLED', False)
    mock_llm_utils['clean'].return_value = "" # Assume no scraped content to clean

    # LLM Story Generation succeeds
    generated_story_script = "### Story Script:\nStory to fail storage.\n### Video Description:\nDesc."
    mock_llm_utils['process'].return_value = (generated_story_script, "LLM thought")
    
    # Storage (Story Script) fails
    store_script_exception_message = "S3 is down, cannot store script"
    if store_script_failure_mode == "returns_none":
        # store_file for script returns None
        mock_storage_utils.side_effect = [None] 
    else: # raises_exception
        mock_storage_utils.side_effect = [Exception(store_script_exception_message)]
        
    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is False

    mock_llm_utils['process'].assert_called_once() # LLM was called
    mock_storage_utils.assert_called_once_with(generated_story_script, job_id, "story_script.txt", is_json=False)

    # Verify subsequent functions are NOT called
    mock_storyboard_utils.assert_not_called()
    # If metadata storage was attempted after this failure, it would be a different test.
    # Based on current logic, it fails before trying to store metadata if script store fails.

    # Verify track_job_progress calls
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "failed"
    expected_error_in_status = f"Failed to store story script for job {job_id}"
    if store_script_failure_mode == "raises_exception":
        assert expected_error_in_status in last_track_call_args[0][1]["error_message"]
        assert store_script_exception_message in last_track_call_args[0][1]["error_message"]
    else:
        assert expected_error_in_status == last_track_call_args[0][1]["error_message"] # Exact match if no exception detail
    
    # Verify logger call
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list:
        log_msg = call_arg[0][0]
        if store_script_failure_mode == "raises_exception":
            if f"Failed to store story script for job {job_id}" in log_msg and \
               store_script_exception_message in str(call_arg[0][1]):
                error_log_found = True
                break
        else: # returns_none
            if f"Failed to store story script for job {job_id}: store_file returned None" in log_msg:
                error_log_found = True
                break
    assert error_log_found, "Expected error log for store script failure not found or incorrect."

# Scenario 7: Storyboard generation fails (if enabled)
@pytest.mark.asyncio
@pytest.mark.parametrize("storyboard_failure_mode", ["returns_none", "raises_exception"])
async def test_process_api_job_storyboard_generation_fails(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils,
    storyboard_failure_mode
):
    job_id = f"job_sb_fail_{storyboard_failure_mode}_007"
    job_type = "generate_story_and_video"
    consumer_id = f"consumer_sb_fail_{storyboard_failure_mode}_007"
    job_data = {"job_id": job_id, "job_type": job_type, "topic": "Topic SB Fail"}

    # --- Configure App Config ---
    monkeypatch.setattr(app_config.rag, 'ENABLED', False) # Simplify
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', True) # Storyboard is ON

    # --- Configure Mock Behaviors ---
    mock_llm_utils['clean'].return_value = ""
    generated_story_script = "### Story Script:\nStory for SB fail.\n### Video Description:\nDesc."
    mock_llm_utils['process'].return_value = (generated_story_script, "LLM thought")
    
    story_script_storage_url = f"http://s3.mock.com/{job_id}/story_script.txt"
    final_metadata_storage_url = f"http://s3.mock.com/{job_id}/metadata.json"
    
    # storage_manager.store_file: first for script (succeeds), then for metadata (succeeds)
    mock_storage_utils.side_effect = [
        story_script_storage_url, 
        final_metadata_storage_url # This assumes metadata is stored even if SB fails
    ]
    
    # Storyboard Generation fails
    sb_exception_message = "DALL-E API key invalid for storyboard"
    if storyboard_failure_mode == "returns_none":
        mock_storyboard_utils.return_value = None
    else: # raises_exception
        mock_storyboard_utils.side_effect = Exception(sb_exception_message)
        
    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    # Job is considered completed even if storyboard fails, but with errors.
    assert result is True 

    mock_storyboard_utils.assert_called_once_with(job_id, generated_story_script)

    # Verify track_job_progress calls
    # Check the final "completed" status update
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    final_status_data = last_track_call_args[0][1]
    assert final_status_data["status"] == "completed" # Still completed overall
    assert final_status_data["story_script_url"] == story_script_storage_url
    assert "storyboard_url" not in final_status_data or final_status_data["storyboard_url"] is None
    assert "storyboard_data" not in final_status_data or final_status_data["storyboard_data"] is None
    assert "error_message" in final_status_data # Should include a note about storyboard failure
    expected_error_in_status = f"Storyboard generation failed for job {job_id}"
    if storyboard_failure_mode == "raises_exception":
        assert expected_error_in_status in final_status_data["error_message"]
        assert sb_exception_message in final_status_data["error_message"]
    else:
        assert expected_error_in_status == final_status_data["error_message"]
    
    # Verify logger call for storyboard failure
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list: # Check error logs for storyboard specific error
        log_msg = call_arg[0][0]
        if storyboard_failure_mode == "raises_exception":
            if f"Error during storyboard generation for job {job_id}" in log_msg and \
               sb_exception_message in str(call_arg[0][1]):
                error_log_found = True
                break
        else: # returns_none
            if f"Storyboard generation returned None for job {job_id}" in log_msg:
                error_log_found = True
                break
    assert error_log_found, "Expected error log for storyboard failure not found or incorrect."

    # Verify metadata was still stored
    # The number of store_file calls: script + metadata (no storyboard assets)
    assert mock_storage_utils.call_count == 2 
    final_metadata_store_call = mock_storage_utils.call_args_list[-1]
    assert final_metadata_store_call[0][2] == "metadata.json" # filename_key
    stored_metadata_payload = final_metadata_store_call[0][0]
    assert stored_metadata_payload.get("storyboard_url") is None
    assert "Storyboard generation failed" in stored_metadata_payload.get("processing_errors", [])

# Scenario 8: Storing metadata fails
@pytest.mark.asyncio
@pytest.mark.parametrize("store_metadata_failure_mode", ["returns_none", "raises_exception"])
async def test_process_api_job_store_metadata_fails(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils,
    store_metadata_failure_mode
):
    job_id = f"job_store_meta_fail_{store_metadata_failure_mode}_008"
    job_type = "generate_story_and_video"
    consumer_id = f"consumer_store_meta_fail_{store_metadata_failure_mode}_008"
    job_data = {"job_id": job_id, "job_type": job_type, "topic": "Topic Store Meta Fail"}

    # --- Configure App Config & Mocks for successful run up to metadata storage ---
    monkeypatch.setattr(app_config.rag, 'ENABLED', False)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', True) # Storyboard ON

    mock_llm_utils['clean'].return_value = ""
    generated_story_script = "### Story Script:\nStory for meta fail.\n### Video Description:\nDesc."
    mock_llm_utils['process'].return_value = (generated_story_script, "LLM thought")
    
    story_script_storage_url = f"http://s3.mock.com/{job_id}/story_script.txt"
    storyboard_json_storage_url = f"http://s3.mock.com/{job_id}/storyboard.json" # Assuming storyboard.json is also stored
    
    # Storyboard Generation succeeds
    storyboard_data_mock = StoryboardData(
        job_id=job_id, story_script=generated_story_script,
        scenes=[StoryboardScene(scene_number=1, image_prompt="P", narration="N", image_url="s3_img", audio_url="s3_audio", duration_seconds=5)]
    )
    mock_storyboard_utils.return_value = storyboard_data_mock

    # store_file calls:
    # 1. Story script (succeeds)
    # 2. Storyboard JSON (succeeds - generate_storyboard calls store_file for its own output)
    # 3. Metadata JSON (this one will fail)
    store_metadata_exception_message = "S3 is full, cannot store metadata"
    
    def store_file_side_effect_for_metadata_failure(data_or_filepath, job_id_arg, filename_key, **kwargs):
        if filename_key == "story_script.txt":
            return story_script_storage_url
        elif filename_key == "storyboard.json": # generate_storyboard stores this
            return storyboard_json_storage_url
        elif filename_key == "metadata.json": # This is the one that fails
            if store_metadata_failure_mode == "returns_none":
                return None
            else:
                raise Exception(store_metadata_exception_message)
        # For storyboard assets (images/audio) if any, assume they are handled by mock_storyboard_utils
        # or are not relevant if storyboard_data_mock above has pre-filled URLs.
        # For this test, let's assume storyboard_data_mock has URLs, so no asset storage calls here.
        return f"http://s3.mock.com/{job_id_arg}/{filename_key}" # Default for other calls if any

    mock_storage_utils.side_effect = store_file_side_effect_for_metadata_failure
        
    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is False # Overall job processing fails if metadata cannot be stored

    mock_storyboard_utils.assert_called_once() # Storyboard generation was attempted

    # Verify store_file calls - script, storyboard.json (by generate_storyboard), then metadata
    # store_file for storyboard assets (image/audio) are part of generate_storyboard's internal calls
    # and are mocked/verified there. generate_storyboard itself returns StoryboardData with URLs.
    # The store_file for "storyboard.json" is from generate_storyboard.
    # The store_file for "metadata.json" is from process_api_job.
    
    # Expected calls:
    # 1. story_script.txt
    # 2. storyboard.json (from generate_storyboard's successful run)
    # 3. metadata.json (this one fails)
    
    # Check that metadata storage was attempted
    metadata_store_attempted = False
    for call_arg in mock_storage_utils.call_args_list:
        if call_arg[0][1] == job_id and call_arg[0][2] == "metadata.json":
            metadata_store_attempted = True
            break
    assert metadata_store_attempted, "Attempt to store metadata.json was not made."
    
    # Verify track_job_progress calls
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "failed"
    expected_error_in_status = f"Failed to store final metadata for job {job_id}"
    if store_metadata_failure_mode == "raises_exception":
        assert expected_error_in_status in last_track_call_args[0][1]["error_message"]
        assert store_metadata_exception_message in last_track_call_args[0][1]["error_message"]
    else:
        assert expected_error_in_status == last_track_call_args[0][1]["error_message"]
    
    # Verify logger call
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list:
        log_msg = call_arg[0][0]
        if store_metadata_failure_mode == "raises_exception":
            if f"Failed to store metadata for job {job_id}" in log_msg and \
               store_metadata_exception_message in str(call_arg[0][1]):
                error_log_found = True
                break
        else: # returns_none
            if f"Failed to store metadata for job {job_id}: store_file returned None" in log_msg:
                error_log_found = True
                break
    assert error_log_found, "Expected error log for store metadata failure not found or incorrect."

# Scenario 9: Invalid job data

@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_job_data_case, missing_field_or_value, expected_error_msg_fragment", [
    ({"job_id": "job_no_urls_009", "job_type": "generate_story_and_video", "topic": "Topic"}, "urls (when RAG enabled)", "URLs are required for RAG-enabled story generation"),
    ({"job_id": "job_unknown_type_009", "job_type": "unknown_job_type", "topic": "Topic"}, "job_type", "Unknown job_type: unknown_job_type"),
    ({"job_id": "job_no_topic_009", "job_type": "generate_story_and_video"}, "topic", "Missing 'topic' in job data"),
])
async def test_process_api_job_invalid_job_data(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, # Other mocks might not be called
    invalid_job_data_case, missing_field_or_value, expected_error_msg_fragment
):
    job_id = invalid_job_data_case["job_id"]
    job_type = invalid_job_data_case["job_type"]
    consumer_id = f"consumer_invalid_data_{missing_field_or_value}_009"

    # Ensure RAG is enabled for the "urls missing" case to trigger that specific validation
    if missing_field_or_value == "urls (when RAG enabled)":
        monkeypatch.setattr(app_config.rag, 'ENABLED', True)
    else: # For other cases, RAG status doesn't matter as much as it won't be reached
        monkeypatch.setattr(app_config.rag, 'ENABLED', False)


    result = await ajp_module.process_api_job(
        job_id, job_type, invalid_job_data_case, mock_broker_passed_to_processor, consumer_id
    )

    assert result is False

    # Verify track_job_progress for "failed" status
    # The number of track_job_progress calls can vary depending on how early the validation fails.
    # We are interested in the *final* status update.
    assert mock_broker_passed_to_processor.track_job_progress.called # Should be called at least once for failure
    
    # Check the arguments of the last call to track_job_progress
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][0] == job_id
    assert last_track_call_args[0][1]["status"] == "failed"
    assert expected_error_msg_fragment in last_track_call_args[0][1]["error_message"]
    
    # Verify logger call
    error_log_found = False
    for call_arg in mock_ajp_logger.error.call_args_list:
        if expected_error_msg_fragment in call_arg[0][0]:
            error_log_found = True
            break
    assert error_log_found, f"Expected error log containing '{expected_error_msg_fragment}' not found."

    # Ensure core processing functions were not called if validation failed early
    if missing_field_or_value == "urls (when RAG enabled)":
        mock_scraper_client['queue'].assert_not_called() # Fails before scraping
    elif missing_field_or_value == "job_type": # Fails very early
        mock_scraper_client['queue'].assert_not_called()
        mock_llm_utils['process'].assert_not_called()


# Scenario 10: RAG path - Content too short, RAG skipped
@pytest.mark.asyncio
async def test_process_api_job_rag_content_too_short(
    mock_broker_passed_to_processor, mock_ajp_logger, mock_appconfig_api_processor_defaults, monkeypatch,
    mock_scraper_client, mock_llm_utils, mock_storyboard_utils, mock_storage_utils,
    mock_vector_db_utils, mock_text_processing_utils
):
    job_id = "job_rag_short_content_010"
    job_type = "generate_story_and_video"
    consumer_id = "consumer_rag_short_010"
    job_data = {
        "job_id": job_id, "job_type": job_type, "topic": "Topic RAG Short",
        "urls": ["http://example.com/short_content_source"]
    }

    # --- Configure App Config ---
    monkeypatch.setattr(app_config.rag, 'ENABLED', True)
    monkeypatch.setattr(app_config.rag, 'MAX_CHARS_BEFORE_RAG', 5000) # High threshold
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', False) # Simplify by disabling SB

    # --- Configure Mock Behaviors ---
    # Scraper returns short content
    scrape_job_id = f"scrape_{job_id}"
    mock_scraper_client['queue'].return_value = scrape_job_id
    short_markdown_content = "This is short content." # Length < MAX_CHARS_BEFORE_RAG
    scraped_content_list = [URLMetadata(url=job_data["urls"][0], title="Short Source", markdown_content=short_markdown_content, error_message=None)]
    mock_scraper_client['get'].return_value = scraped_content_list
    
    # Content Cleaning
    mock_llm_utils['clean'].return_value = short_markdown_content # Assume cleaning doesn't change it much

    # LLM Story Generation (will use short_markdown_content directly, or None if it was empty)
    generated_story_script_short_rag = "### Story Script:\nStory from short content.\n### Video Description:\nDesc."
    llm_thought_short_rag = "LLM thought for short RAG..."
    mock_llm_utils['process'].return_value = (generated_story_script_short_rag, llm_thought_short_rag)
    
    # Storage
    story_script_storage_url = f"http://s3.mock.com/{job_id}/story_script.txt"
    final_metadata_storage_url = f"http://s3.mock.com/{job_id}/metadata.json"
    mock_storage_utils.side_effect = [story_script_storage_url, final_metadata_storage_url]

    # --- Call the function ---
    result = await ajp_module.process_api_job(
        job_id, job_type, job_data, mock_broker_passed_to_processor, consumer_id
    )

    # --- Assertions ---
    assert result is True

    # Verify RAG functions NOT called
    mock_text_processing_utils.assert_not_called()
    mock_vector_db_utils['add'].assert_not_called()
    mock_vector_db_utils['query'].assert_not_called()
    
    mock_ajp_logger.info.assert_any_call(
        f"Combined scraped content for job {job_id} is too short ({len(short_markdown_content)} chars). Skipping RAG, using content directly."
    )

    # Verify LLM Story Generation call - context_text should be the short_markdown_content itself
    mock_llm_utils['process'].assert_called_once()
    llm_args, _ = mock_llm_utils['process'].call_args
    assert llm_args[3] == short_markdown_content # context_text is the short content

    # Verify final status
    last_track_call_args = mock_broker_passed_to_processor.track_job_progress.call_args_list[-1]
    assert last_track_call_args[0][1]["status"] == "completed"
