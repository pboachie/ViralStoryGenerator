import pytest
import requests
import os
import time
import json
import base64
import uuid
import tempfile
from unittest.mock import patch, MagicMock, mock_open

# Assuming the module is viralStoryGenerator.src.elevenlabs_tts
from viralStoryGenerator.src import elevenlabs_tts as tts_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_defaults(monkeypatch):
    """Set default app_config values for elevenlabs_tts tests."""
    monkeypatch.setattr(app_config.elevenlabs, 'API_KEY', "test_api_key")
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_VOICE_ID', "default_voice_test")
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_MODEL_ID', "eleven_multilingual_v2")
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_STABILITY', 0.5)
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_SIMILARITY_BOOST', 0.7)
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_STYLE', 0.0) # Assuming 0.0 is a valid default
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_USE_SPEAKER_BOOST', True)
    monkeypatch.setattr(app_config.elevenlabs, 'MAX_RETRIES', 2)
    monkeypatch.setattr(app_config.elevenlabs, 'RETRY_DELAY_SECONDS', 0.01) # Short delay for tests
    monkeypatch.setattr(app_config, 'APP_TITLE', "TestApp")
    monkeypatch.setattr(app_config, 'VERSION', "0.1-test")
    # Ensure output directory exists for some tests or is mockable
    monkeypatch.setattr(app_config, 'AUDIO_OUTPUT_DIR', "/tmp/test_audio_output")


# --- Tests for generate_elevenlabs_audio ---

# Scenario 1: Successful audio generation (no timestamps)
@patch('requests.post')
@patch('builtins.open', new_callable=mock_open)
@patch('os.path.isdir')
@patch('os.makedirs')
@patch('os.access')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_success_no_timestamps(
    mock_logger, mock_os_access, mock_os_makedirs, mock_os_isdir, 
    mock_file_open, mock_requests_post, mock_appconfig_defaults # mock_appconfig_defaults is via fixture
):
    mock_os_isdir.return_value = True # Assume output directory exists
    mock_os_access.return_value = True # Assume output directory is writable

    mock_audio_content = b"dummy_audio_stream_content"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [mock_audio_content] # Simulate streaming
    mock_requests_post.return_value = mock_response

    text_to_generate = "Hello world, this is a test."
    output_mp3_path = "/tmp/test_audio_output/test_output.mp3"
    voice_id = "voice_123"

    result = tts_module.generate_elevenlabs_audio(text_to_generate, output_mp3_path, voice_id, include_timestamps=False)

    assert result is True
    
    expected_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    expected_headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "test_api_key",
        "User-Agent": f"{app_config.APP_TITLE}/{app_config.VERSION}"
    }
    expected_payload = {
        "text": text_to_generate,
        "model_id": app_config.elevenlabs.DEFAULT_MODEL_ID,
        "voice_settings": {
            "stability": app_config.elevenlabs.DEFAULT_STABILITY,
            "similarity_boost": app_config.elevenlabs.DEFAULT_SIMILARITY_BOOST,
            "style": app_config.elevenlabs.DEFAULT_STYLE,
            "use_speaker_boost": app_config.elevenlabs.DEFAULT_USE_SPEAKER_BOOST,
        }
    }
    mock_requests_post.assert_called_once_with(expected_url, json=expected_payload, headers=expected_headers, stream=True)
    
    mock_file_open.assert_called_once_with(output_mp3_path, 'wb')
    mock_file_open().write.assert_called_once_with(mock_audio_content)
    mock_logger.info.assert_any_call(f"Successfully generated audio and saved to {output_mp3_path}")


# Scenario 2: Successful audio generation (with timestamps)
@patch('requests.post')
@patch('builtins.open', new_callable=mock_open)
@patch('os.path.isdir')
@patch('os.makedirs')
@patch('os.access')
@patch('base64.b64decode')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_success_with_timestamps(
    mock_logger, mock_b64decode, mock_os_access, mock_os_makedirs, mock_os_isdir, 
    mock_file_open, mock_requests_post, mock_appconfig_defaults
):
    mock_os_isdir.return_value = True
    mock_os_access.return_value = True

    mock_audio_content_raw = b"dummy_audio_raw_for_base64"
    mock_audio_content_b64 = base64.b64encode(mock_audio_content_raw).decode('utf-8')
    mock_timestamps_data = {"timestamps": [[0, 100, "word1"], [101, 200, "word2"]]}
    
    mock_response_json = {
        "audio": mock_audio_content_b64,
        "alignment": mock_timestamps_data # Assuming 'alignment' key holds timestamps
    }
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_json # For timestamp endpoint
    mock_requests_post.return_value = mock_response
    
    mock_b64decode.return_value = mock_audio_content_raw

    text_to_generate = "Hello with timestamps."
    output_mp3_path = "/tmp/test_audio_output/test_timestamps.mp3"
    voice_id = "voice_ts_456"

    result = tts_module.generate_elevenlabs_audio(text_to_generate, output_mp3_path, voice_id, include_timestamps=True)

    assert isinstance(result, dict) # Should return the timestamps dict
    assert result == mock_timestamps_data
    
    expected_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}" # No /stream for timestamps
    expected_headers = {
        "Accept": "application/json", # Expect JSON for timestamps
        "Content-Type": "application/json",
        "xi-api-key": "test_api_key",
        "User-Agent": f"{app_config.APP_TITLE}/{app_config.VERSION}"
    }
    expected_payload = {
        "text": text_to_generate,
        "model_id": app_config.elevenlabs.DEFAULT_MODEL_ID,
        "voice_settings": {
            "stability": app_config.elevenlabs.DEFAULT_STABILITY,
            "similarity_boost": app_config.elevenlabs.DEFAULT_SIMILARITY_BOOST,
            "style": app_config.elevenlabs.DEFAULT_STYLE,
            "use_speaker_boost": app_config.elevenlabs.DEFAULT_USE_SPEAKER_BOOST,
        },
        "output_format": "mp3_44100_128_timestamps" # Specific output format for timestamps
    }
    mock_requests_post.assert_called_once_with(expected_url, json=expected_payload, headers=expected_headers)
    
    mock_b64decode.assert_called_once_with(mock_audio_content_b64)
    mock_file_open.assert_called_once_with(output_mp3_path, 'wb')
    mock_file_open().write.assert_called_once_with(mock_audio_content_raw)
    mock_logger.info.assert_any_call(f"Successfully generated audio with timestamps and saved to {output_mp3_path}")


# Scenario 3: API Error Handling
@pytest.mark.parametrize("error_code, error_reason, response_text", [
    (401, "Unauthorized", "Invalid API key"),
    (422, "Unprocessable Entity", '{"detail": [{"loc": ["body", "text"], "msg": "value is too long"}]}'),
    (500, "Internal Server Error", "Server error during generation")
])
@patch('requests.post')
@patch('os.path.isdir', return_value=True) # Assume dir checks pass
@patch('os.access', return_value=True)   # Assume dir checks pass
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_api_error(
    mock_logger, mock_os_access, mock_os_isdir, mock_requests_post, 
    error_code, error_reason, response_text, mock_appconfig_defaults
):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = error_code
    mock_response.reason = error_reason
    mock_response.text = response_text
    # For 422, .json() might be called by the error handler in the function
    if error_code == 422:
        try:
            mock_response.json.return_value = json.loads(response_text)
        except json.JSONDecodeError:
            mock_response.json.side_effect = json.JSONDecodeError("mock error", "doc", 0)
    else:
        # Ensure .json() raises an error if called for non-JSON responses
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("mock error", "doc", 0)


    mock_requests_post.return_value = mock_response

    text_to_generate = "Text for API error test."
    output_mp3_path = "/tmp/test_audio_output/api_error.mp3"
    voice_id = "voice_api_err"

    result = tts_module.generate_elevenlabs_audio(text_to_generate, output_mp3_path, voice_id)

    assert result is False
    mock_requests_post.assert_called_once() # Should be called once, no retries for these client/server errors by default
    
    # Verify logging
    mock_logger.error.assert_called_once()
    log_message = mock_logger.error.call_args[0][0]
    assert f"ElevenLabs API Error ({error_code} {error_reason})" in log_message
    # Check if response text is included in the log message for more details
    assert response_text in log_message


# Scenario 4: Request Exception Handling (Timeout, ConnectionError with Retries)
@pytest.mark.parametrize("exception_type", [requests.exceptions.Timeout, requests.exceptions.ConnectionError])
@patch('requests.post')
@patch('time.sleep', return_value=None) # Mock time.sleep to avoid actual delay
@patch('os.path.isdir', return_value=True)
@patch('os.access', return_value=True)
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_request_exception_with_retries(
    mock_logger, mock_os_access, mock_os_isdir, mock_time_sleep, 
    mock_requests_post, exception_type, mock_appconfig_defaults # Ensure fixture is used
):
    max_retries = app_config.elevenlabs.MAX_RETRIES # Should be 2 from fixture
    
    # Configure requests.post to raise the specified exception
    mock_requests_post.side_effect = exception_type(f"Simulated {exception_type.__name__}")

    text_to_generate = "Text for request exception test."
    output_mp3_path = "/tmp/test_audio_output/request_exception.mp3"
    voice_id = "voice_req_err"

    result = tts_module.generate_elevenlabs_audio(text_to_generate, output_mp3_path, voice_id)

    assert result is False
    # requests.post should be called 1 (initial) + max_retries times
    assert mock_requests_post.call_count == 1 + max_retries
    # time.sleep should be called max_retries times
    assert mock_time_sleep.call_count == max_retries
    
    # Verify logging for retries and final failure
    initial_error_log_found = False
    retry_log_found = False
    final_error_log_found = False

    for call_args in mock_logger.error.call_args_list:
        log_msg = call_args[0][0]
        if f"Simulated {exception_type.__name__}" in log_msg and "Retrying in" in log_msg:
            initial_error_log_found = True # This also covers retry logging somewhat
        if "Attempt" in log_msg and f"Simulated {exception_type.__name__}" in log_msg : # More specific retry log check
            retry_log_found = True
        if f"Failed to generate audio after {max_retries + 1} attempts" in log_msg:
            final_error_log_found = True
            
    assert initial_error_log_found or retry_log_found # At least one error log indicating a retry attempt
    assert final_error_log_found

    # More specific check on retry logs if needed:
    # e.g., mock_logger.error.assert_any_call(
    #     f"Error during ElevenLabs API request (Attempt 1/{max_retries + 1}): Simulated {exception_type.__name__}. "
    #     f"Retrying in {app_config.elevenlabs.RETRY_DELAY_SECONDS} seconds..."
    # )

# Scenario 5: Input Validation
@patch('requests.post') # Should not be called for these validation errors
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_empty_text(mock_logger, mock_requests_post, mock_appconfig_defaults):
    result = tts_module.generate_elevenlabs_audio("", "/tmp/empty_text.mp3", "voice_empty_text")
    assert result is False
    mock_logger.error.assert_called_once_with("Input text cannot be empty.")
    mock_requests_post.assert_not_called()

@patch('requests.post')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_no_api_key(mock_logger, mock_requests_post, monkeypatch):
    monkeypatch.setattr(app_config.elevenlabs, 'API_KEY', None) # No API Key
    
    result = tts_module.generate_elevenlabs_audio("Text with no API key.", "/tmp/no_api_key.mp3", "voice_no_key")
    
    assert result is False
    mock_logger.error.assert_called_once_with("ElevenLabs API Key not configured.")
    mock_requests_post.assert_not_called()
    # Restore API key for other tests if monkeypatch isn't fully isolating it for app_config module itself
    monkeypatch.setattr(app_config.elevenlabs, 'API_KEY', "test_api_key")


@patch('requests.post')
@patch('os.path.isdir', return_value=False) # Dir does not exist initially
@patch('os.makedirs')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_cannot_create_output_dir(
    mock_logger, mock_os_makedirs, mock_os_isdir, mock_requests_post, mock_appconfig_defaults
):
    mock_os_makedirs.side_effect = OSError("Permission denied to create directory")
    
    output_mp3_path = "/uncreatable_dir/test.mp3"
    result = tts_module.generate_elevenlabs_audio("Text for uncreatable dir.", output_mp3_path, "voice_uncreatable")

    assert result is False
    mock_os_makedirs.assert_called_once_with(os.path.dirname(output_mp3_path), exist_ok=True)
    mock_logger.error.assert_called_once_with(
        f"Output directory {os.path.dirname(output_mp3_path)} does not exist and could not be created. Error: Permission denied to create directory"
    )
    mock_requests_post.assert_not_called()


@patch('requests.post')
@patch('os.path.isdir', return_value=True) # Dir exists
@patch('os.access')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_output_dir_not_writable(
    mock_logger, mock_os_access, mock_os_isdir, mock_requests_post, mock_appconfig_defaults
):
    mock_os_access.return_value = False # Dir not writable
    
    output_mp3_path = "/non_writable_dir/test.mp3"
    result = tts_module.generate_elevenlabs_audio("Text for non-writable dir.", output_mp3_path, "voice_non_writable")

    assert result is False
    # os.access is called with the directory path and os.W_OK
    mock_os_access.assert_called_once_with(os.path.dirname(output_mp3_path), os.W_OK)
    mock_logger.error.assert_called_once_with(
        f"Output directory {os.path.dirname(output_mp3_path)} is not writable."
    )
    mock_requests_post.assert_not_called()

# Scenario 6: Error during file writing
@patch('requests.post')
@patch('builtins.open', new_callable=mock_open)
@patch('os.path.isdir', return_value=True)
@patch('os.access', return_value=True)
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_elevenlabs_audio_file_write_error(
    mock_logger, mock_os_access, mock_os_isdir, mock_file_open_mock, mock_requests_post, 
    mock_appconfig_defaults
):
    mock_audio_content = b"dummy_audio_for_write_error"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [mock_audio_content]
    mock_requests_post.return_value = mock_response

    # Configure the mock_open's file object to raise IOError on write
    mock_file_obj = MagicMock()
    mock_file_obj.write.side_effect = IOError("Disk full error")
    mock_file_open_mock.return_value.__enter__.return_value = mock_file_obj # When 'with open(...) as f:' is used

    text_to_generate = "Text for file write error test."
    output_mp3_path = "/tmp/test_audio_output/file_write_error.mp3"
    voice_id = "voice_file_err"

    result = tts_module.generate_elevenlabs_audio(text_to_generate, output_mp3_path, voice_id)

    assert result is False
    mock_requests_post.assert_called_once() # API call was made
    mock_file_open_mock.assert_called_once_with(output_mp3_path, 'wb')
    mock_file_obj.write.assert_called_once_with(mock_audio_content) # Attempted to write
    
    mock_logger.error.assert_called_once()
    log_message = mock_logger.error.call_args[0][0]
    assert f"Failed to write audio stream to {output_mp3_path}" in log_message
    assert "Disk full error" in log_message # Check if original error is in log


# --- Tests for generate_audio (minimal) ---

@patch('viralStoryGenerator.src.elevenlabs_tts.generate_elevenlabs_audio')
@patch('uuid.uuid4')
@patch('tempfile.gettempdir')
@patch('os.path.join')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_audio_calls_generate_elevenlabs_audio_success(
    mock_logger, mock_os_path_join, mock_tempfile_gettempdir, mock_uuid4, 
    mock_generate_elevenlabs_audio_func, mock_appconfig_defaults
):
    # Setup mocks
    mock_temp_dir = "/fake/temp"
    mock_tempfile_gettempdir.return_value = mock_temp_dir
    
    mock_job_id = "testjob123"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id)
    
    expected_mp3_filename = f"{app_config.APP_TITLE}_TTS_{mock_job_id}.mp3"
    expected_output_path = os.path.join(mock_temp_dir, expected_mp3_filename)
    mock_os_path_join.return_value = expected_output_path
    
    mock_generate_elevenlabs_audio_func.return_value = True # Simulate success from underlying function

    text_input = "This is the story to generate."
    voice_id_input = "voice_for_generate_audio"

    result = tts_module.generate_audio(text_input, voice_id_input)

    assert isinstance(result, dict)
    assert result["mp3_file_path"] == expected_output_path
    assert result["job_id"] == mock_job_id
    assert result["timestamps"] is None # Default, as include_timestamps=False

    mock_tempfile_gettempdir.assert_called_once()
    mock_uuid4.assert_called_once()
    mock_os_path_join.assert_called_once_with(mock_temp_dir, expected_mp3_filename)
    mock_generate_elevenlabs_audio_func.assert_called_once_with(
        text=text_input,
        output_mp3_path=expected_output_path,
        voice_id=voice_id_input,
        include_timestamps=False # Default for generate_audio
    )


@patch('viralStoryGenerator.src.elevenlabs_tts.generate_elevenlabs_audio')
@patch('uuid.uuid4') # Still need to mock these as they are called before the failure
@patch('tempfile.gettempdir')
@patch('os.path.join')
@patch('viralStoryGenerator.src.elevenlabs_tts._logger')
def test_generate_audio_calls_generate_elevenlabs_audio_failure(
    mock_logger, mock_os_path_join, mock_tempfile_gettempdir, mock_uuid4, 
    mock_generate_elevenlabs_audio_func, mock_appconfig_defaults
):
    mock_temp_dir = "/fake/temp_fail"
    mock_tempfile_gettempdir.return_value = mock_temp_dir
    mock_job_id = "testjob_fail_456"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id)
    expected_mp3_filename = f"{app_config.APP_TITLE}_TTS_{mock_job_id}.mp3"
    expected_output_path = os.path.join(mock_temp_dir, expected_mp3_filename)
    mock_os_path_join.return_value = expected_output_path

    mock_generate_elevenlabs_audio_func.return_value = False # Simulate failure

    text_input = "This will fail."
    voice_id_input = "voice_for_failure"

    result = tts_module.generate_audio(text_input, voice_id_input)

    assert result is None
    mock_generate_elevenlabs_audio_func.assert_called_once_with(
        text=text_input,
        output_mp3_path=expected_output_path,
        voice_id=voice_id_input,
        include_timestamps=False
    )
    # Logger might be called by generate_elevenlabs_audio, not directly by generate_audio on failure.
    # If generate_audio itself logs, that can be asserted here.
    # Based on typical structure, generate_elevenlabs_audio would do its own error logging.
