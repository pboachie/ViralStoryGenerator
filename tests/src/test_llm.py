import pytest
import requests
import time
import json
from unittest.mock import patch, MagicMock

# Assuming the module is viralStoryGenerator.src.llm
from viralStoryGenerator.src import llm as llm_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_defaults(monkeypatch):
    """Set default app_config values for llm tests."""
    monkeypatch.setattr(app_config.llm, 'ENDPOINT', "http://mock-llm-api.com/v1/chat/completions")
    monkeypatch.setattr(app_config.llm, 'MODEL', "mock-model-001")
    monkeypatch.setattr(app_config.llm, 'MAX_TOKENS', 1000)
    monkeypatch.setattr(app_config.llm, 'DEFAULT_TEMPERATURE', 0.7)
    monkeypatch.setattr(app_config.llm, 'DEFAULT_TOP_P', 1.0)
    monkeypatch.setattr(app_config.llm, 'PRESENCE_PENALTY', 0.0)
    monkeypatch.setattr(app_config.llm, 'FREQUENCY_PENALTY', 0.0)
    # For _make_llm_request retry logic
    monkeypatch.setattr(app_config.httpOptions, 'TIMEOUT', 10) # Default timeout for requests
    monkeypatch.setattr(app_config.llm, 'RETRY_ATTEMPTS', 3) # tenacity default is 3
    monkeypatch.setattr(app_config.llm, 'RETRY_MIN_WAIT_SECONDS', 0.1) # For tenacity: wait_random_exponential
    monkeypatch.setattr(app_config.llm, 'RETRY_MAX_WAIT_SECONDS', 0.5) # For tenacity
    # For clean_markdown_with_llm
    monkeypatch.setattr(app_config.llm, 'MIN_MARKDOWN_LENGTH_FOR_CLEANING', 50)


@pytest.fixture
def mock_requests_post():
    """Fixture to mock requests.post."""
    with patch('requests.post') as mock_post:
        yield mock_post

@pytest.fixture
def mock_time_sleep():
    """Fixture to mock time.sleep."""
    with patch('time.sleep', return_value=None) as mock_sleep: # return_value=None so it doesn't try to sleep
        yield mock_sleep

@pytest.fixture
def mock_llm_logger():
    """Fixture to mock the _logger in llm.py."""
    with patch('viralStoryGenerator.src.llm._logger') as mock_logger:
        yield mock_logger

# --- Tests for _make_llm_request ---

# Scenario 1.1: Successful request
def test_make_llm_request_successful(mock_requests_post, mock_llm_logger, mock_appconfig_defaults):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    expected_response_data = {"choices": [{"message": {"content": "LLM response text"}}]}
    mock_response.json.return_value = expected_response_data
    mock_requests_post.return_value = mock_response

    messages = [{"role": "user", "content": "Hello LLM"}]
    temperature = 0.8
    max_tokens = 150
    
    response_data = llm_module._make_llm_request(messages, temperature=temperature, max_tokens=max_tokens, model_name="test-model")

    assert response_data == expected_response_data
    mock_requests_post.assert_called_once()
    args, kwargs = mock_requests_post.call_args
    assert args[0] == app_config.llm.ENDPOINT
    
    payload = kwargs['json']
    assert payload['model'] == "test-model" # Model name override
    assert payload['messages'] == messages
    assert payload['temperature'] == temperature
    assert payload['max_tokens'] == max_tokens
    assert payload['top_p'] == app_config.llm.DEFAULT_TOP_P # Default
    assert payload['presence_penalty'] == app_config.llm.PRESENCE_PENALTY
    assert payload['frequency_penalty'] == app_config.llm.FREQUENCY_PENALTY
    
    assert kwargs['timeout'] == app_config.httpOptions.TIMEOUT
    mock_llm_logger.debug.assert_any_call(f"LLM API Request to {app_config.llm.ENDPOINT} with model test-model. Payload: {payload}")
    mock_llm_logger.debug.assert_any_call(f"LLM API Response (200): {expected_response_data}")


# Scenario 1.2: Retry on requests.exceptions.Timeout, then success
def test_make_llm_request_retry_timeout_then_success(
    mock_requests_post, mock_time_sleep, mock_llm_logger, mock_appconfig_defaults
):
    mock_successful_response = MagicMock(spec=requests.Response)
    mock_successful_response.status_code = 200
    expected_response_data = {"choices": [{"message": {"content": "Success after timeout"}}]}
    mock_successful_response.json.return_value = expected_response_data

    # Simulate Timeout on first call, then success
    mock_requests_post.side_effect = [
        requests.exceptions.Timeout("Simulated timeout"),
        mock_successful_response
    ]

    messages = [{"role": "user", "content": "Test timeout retry"}]
    response_data = llm_module._make_llm_request(messages)

    assert response_data == expected_response_data
    assert mock_requests_post.call_count == 2 # Initial call + 1 retry
    mock_time_sleep.assert_called_once() # tenacity calls sleep between retries
    
    mock_llm_logger.warning.assert_any_call(
        "LLM API request error: Simulated timeout. Attempt 1 of 3. Retrying in ..." # Message from tenacity
    )
    mock_llm_logger.info.assert_any_call(
        "LLM API request successful after retry. Attempts: 2"
    )

# Scenario 1.3: Retry on requests.exceptions.ConnectionError, then success
def test_make_llm_request_retry_connection_error_then_success(
    mock_requests_post, mock_time_sleep, mock_llm_logger, mock_appconfig_defaults
):
    mock_successful_response = MagicMock(spec=requests.Response)
    mock_successful_response.status_code = 200
    expected_response_data = {"choices": [{"message": {"content": "Success after ConnectionError"}}]}
    mock_successful_response.json.return_value = expected_response_data

    mock_requests_post.side_effect = [
        requests.exceptions.ConnectionError("Simulated connection error"),
        mock_successful_response
    ]

    messages = [{"role": "user", "content": "Test ConnectionError retry"}]
    response_data = llm_module._make_llm_request(messages)

    assert response_data == expected_response_data
    assert mock_requests_post.call_count == 2
    mock_time_sleep.assert_called_once()
    mock_llm_logger.warning.assert_any_call(
        "LLM API request error: Simulated connection error. Attempt 1 of 3. Retrying in ..."
    )
    mock_llm_logger.info.assert_any_call(
        "LLM API request successful after retry. Attempts: 2"
    )

# Scenario 1.4: Retry on requests.exceptions.HTTPError (5xx), then success
def test_make_llm_request_retry_http_5xx_error_then_success(
    mock_requests_post, mock_time_sleep, mock_llm_logger, mock_appconfig_defaults
):
    mock_http_error_response = MagicMock(spec=requests.Response)
    mock_http_error_response.status_code = 503 # Service Unavailable
    mock_http_error_response.reason = "Service Unavailable"
    mock_http_error_response.text = "The server is temporarily busy"
    # Mock raise_for_status to raise HTTPError for this response
    mock_http_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "503 Server Error", response=mock_http_error_response
    )
    
    mock_successful_response = MagicMock(spec=requests.Response)
    mock_successful_response.status_code = 200
    expected_response_data = {"choices": [{"message": {"content": "Success after 503 HTTPError"}}]}
    mock_successful_response.json.return_value = expected_response_data
    # Ensure raise_for_status does nothing for the successful response
    mock_successful_response.raise_for_status = MagicMock()


    mock_requests_post.side_effect = [
        mock_http_error_response,
        mock_successful_response
    ]

    messages = [{"role": "user", "content": "Test HTTPError 5xx retry"}]
    response_data = llm_module._make_llm_request(messages)

    assert response_data == expected_response_data
    assert mock_requests_post.call_count == 2
    mock_http_error_response.raise_for_status.assert_called_once() # Ensure it was called for the error response
    mock_successful_response.raise_for_status.assert_called_once() # And for the success one
    mock_time_sleep.assert_called_once()
    mock_llm_logger.warning.assert_any_call(
        "LLM API request error: 503 Server Error. Attempt 1 of 3. Retrying in ..."
    )
    mock_llm_logger.info.assert_any_call(
        "LLM API request successful after retry. Attempts: 2"
    )

# Scenario 1.5: Exhaust retries and still fail (e.g., persistent Timeout)
def test_make_llm_request_exhaust_retries_timeout(
    mock_requests_post, mock_time_sleep, mock_llm_logger, mock_appconfig_defaults
):
    # All attempts raise Timeout
    mock_requests_post.side_effect = requests.exceptions.Timeout("Persistent timeout")
    
    messages = [{"role": "user", "content": "Test exhaust retries"}]
    
    with pytest.raises(requests.exceptions.Timeout, match="Persistent timeout"):
        llm_module._make_llm_request(messages)

    max_attempts = app_config.llm.RETRY_ATTEMPTS # Tenacity's default is 3 attempts total
    assert mock_requests_post.call_count == max_attempts 
    assert mock_time_sleep.call_count == max_attempts - 1 # Sleeps between retries
    
    # Check that multiple retry warnings were logged
    assert mock_llm_logger.warning.call_count == max_attempts -1 # Warns before each retry
    mock_llm_logger.warning.assert_any_call(
        "LLM API request error: Persistent timeout. Attempt 1 of 3. Retrying in ..."
    )
    mock_llm_logger.warning.assert_any_call(
        "LLM API request error: Persistent timeout. Attempt 2 of 3. Retrying in ..."
    )
    # The final error log comes from tenacity's @retry(stop=...) handler or if the exception is re-raised
    # Here, the exception is re-raised, so no specific "final failure" log from our code.
    # Tenacity itself might log. For our logger, we check the warnings.


# Scenario 1.6: Handling of model name cleaning (e.g., removing #suffix)
def test_make_llm_request_model_name_cleaning(mock_requests_post, mock_llm_logger, mock_appconfig_defaults):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    expected_response_data = {"choices": [{"message": {"content": "Model name test"}}]}
    mock_response.json.return_value = expected_response_data
    mock_requests_post.return_value = mock_response

    messages = [{"role": "user", "content": "Test model name cleaning"}]
    
    llm_module._make_llm_request(messages, model_name="test-model#with-suffix")

    mock_requests_post.assert_called_once()
    args, kwargs = mock_requests_post.call_args
    payload = kwargs['json']
    assert payload['model'] == "test-model" # Suffix should be cleaned by _get_model_name
    mock_llm_logger.debug.assert_any_call(
        f"LLM API Request to {app_config.llm.ENDPOINT} with model test-model. Payload: {payload}"
    )

# --- Tests for _pre_process_markdown ---

@pytest.mark.parametrize("input_text, expected_output", [
    ("", ""), # Empty string
    ("   ", ""), # String with only clutter (spaces)
    ("  \n\n \t  ", ""), # String with only clutter (mixed whitespace)
    ("<!-- HTML comment -->Hello<!-- another -->", "Hello"), # HTML comments
    ("Text with <script>alert('XSS')</script> evil script", "Text with evil script"), # Script tags (content kept by default)
    ("Text with <style>body { color: red; }</style> style block", "Text with style block"), # Style tags (content kept)
    ("Text with <img src='image.png' alt='Test Image'> image tag", "Text with image tag"), # Image tags (content kept)
    ("Line 1\n\n\nLine 2", "Line 1\nLine 2"), # Multiple newlines to single
    ("  Leading and trailing spaces  ", "Leading and trailing spaces"), # Kept by default, only internal normalized
    ("Text with \t tabs and  multiple   spaces", "Text with tabs and multiple spaces"), # Tabs and multiple spaces
    ("## Unwanted Header\nActual content", "Actual content"), # Specific unwanted headers
    ("### Another Unwanted Header\nMore content", "More content"),
    ("Some text\n\n[toc]\n\nMore text", "Some text\nMore text"), # Table of contents
    ("Content with ![Alt text](image.png) markdown image", "Content with markdown image"), # Markdown image syntax (alt text kept)
    ("Line with &nbsp; non-breaking space", "Line with non-breaking space"), # HTML entities like &nbsp; are not directly handled by this pre-processor
    ("Line with &#123; numeric entity", "Line with &#123; numeric entity"), # Nor are numeric entities
    # Test case for the specific regex `r"^#{1,6}\s+.*?\n"` to remove all headers
    ("# Header 1\nContent under H1", "Content under H1"),
    ("## Header 2\nContent under H2", "Content under H2"),
    ("### Header 3\nContent under H3", "Content under H3"),
    ("#### Header 4\nContent under H4", "Content under H4"),
    ("##### Header 5\nContent under H5", "Content under H5"),
    ("###### Header 6\nContent under H6", "Content under H6"),
    ("Not a header\nStill content", "Not a header\nStill content"),
    ("Content before\n# Header Mid\nContent after", "Content before\nContent after"), # Header in middle
    ("Content\n# \nEmpty Header\nMore Content", "Content\nEmpty Header\nMore Content"), # Header with no text, then content
])
def test_pre_process_markdown_various_inputs(input_text, expected_output, mock_appconfig_defaults):
    # _pre_process_markdown is not a public member, access via module
    processed_text = llm_module._pre_process_markdown(input_text)
    assert processed_text == expected_output

def test_pre_process_markdown_unwanted_headers_custom(mock_appconfig_defaults, monkeypatch):
    # Test with custom UNWANTED_HEADERS_PATTERNS
    custom_patterns = [r"^!{3}\s+Custom Unwanted Header.*?\n"]
    monkeypatch.setattr(llm_module, 'UNWANTED_HEADERS_PATTERNS', custom_patterns)
    
    input_text = "!!! Custom Unwanted Header\nThis should be removed.\nActual content here."
    expected_text = "Actual content here." # Assuming the pattern removes the header line itself.
                                         # If it only removes the header text, expected would be different.
                                         # The current patterns in code are r"^#{1,6}\s+.*?\n", which removes the line.
    
    processed_text = llm_module._pre_process_markdown(input_text)
    assert processed_text == expected_text

# --- Tests for _post_process_llm_output ---

@pytest.mark.parametrize("input_text, expected_output", [
    ("Here is the cleaned markdown:\n```markdown\nActual content here.\n```\nI hope this helps!", "Actual content here."),
    ("```markdown\nContent within backticks.\n```", "Content within backticks."),
    ("```\nContent within simple backticks.\n```", "Content within simple backticks."),
    ("Content without any preambles or backticks.", "Content without any preambles or backticks."),
    ("Here is the story:\nOnce upon a time...", "Once upon a time..."),
    ("The video script is as follows:\nACTION: A bird flies.", "ACTION: A bird flies."),
    ("  \n  ```markdown\n  Indented content  \n  ```  \n  ", "Indented content"), # Check trimming of content inside backticks
    ("No preamble, but ```markdown\ncontent\n```", "content"),
    ("Preamble\nNo backticks but text", "No backticks but text"), # Preamble removed, text kept
    ("```json\n{\"key\": \"value\"}\n```", "{\"key\": \"value\"}"), # Other languages in backticks
    ("Text with multiple\n```markdown\nsection1\n```\nand\n```markdown\nsection2\n```\nblocks.", "section1\nand\nsection2"), # Multiple blocks, text between is joined
    ("", ""), # Empty string
    ("   ", ""), # Blank string
    ("Only preamble: Here is the content:", "Only preamble: Here is the content:"), # Preamble not removed if no clear content follows in a standard way
    ("```markdown\n```", ""), # Empty content within markdown block
    ("Preamble\n```\n```", ""), # Preamble with empty code block
    ("Response:\nHere is the story you requested:\n\n# Title\n\nOnce upon a time...", "# Title\n\nOnce upon a time..."), # Multiple preambles
    ("The result is:\n\n[SCENE START]\nACTION: Test\n[SCENE END]", "[SCENE START]\nACTION: Test\n[SCENE END]"),
])
def test_post_process_llm_output_various_inputs(input_text, expected_output, mock_appconfig_defaults):
    processed_text = llm_module._post_process_llm_output(input_text)
    assert processed_text == expected_output

def test_post_process_llm_output_custom_preamble(mock_appconfig_defaults, monkeypatch):
    # Test with custom PREAMBLE_PATTERNS
    custom_patterns = [r"^My custom preamble:?\s*\n?"]
    monkeypatch.setattr(llm_module, 'PREAMBLE_PATTERNS', custom_patterns)
    
    input_text = "My custom preamble:\nThis is the actual content."
    expected_text = "This is the actual content."
    
    processed_text = llm_module._post_process_llm_output(input_text)
    assert processed_text == expected_text

# --- Tests for clean_markdown_with_llm ---

@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._pre_process_markdown')
@patch('viralStoryGenerator.src.llm._post_process_llm_output')
def test_clean_markdown_with_llm_successful_cleaning(
    mock_post_process, mock_pre_process, mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    input_markdown = "This is some #markdown content that is long enough to be processed."
    pre_processed_markdown = "This is pre-processed markdown."
    llm_response_content = "This is the LLM's cleaned version."
    final_cleaned_content = "This is the final cleaned version after post-processing."

    mock_pre_process.return_value = pre_processed_markdown
    mock_make_request.return_value = {"choices": [{"message": {"content": llm_response_content}}]}
    mock_post_process.return_value = final_cleaned_content

    result = llm_module.clean_markdown_with_llm(input_markdown)

    assert result == final_cleaned_content
    mock_pre_process.assert_called_once_with(input_markdown)
    mock_make_request.assert_called_once() # Args checked below
    # Check messages passed to _make_llm_request
    args, _ = mock_make_request.call_args
    messages = args[0]
    assert len(messages) == 2 # System and User prompts
    assert messages[0]["role"] == "system"
    assert "You are an expert in cleaning and reformatting ugly markdown text." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert pre_processed_markdown in messages[1]["content"]
    
    mock_post_process.assert_called_once_with(llm_response_content)
    mock_llm_logger.info.assert_any_call(f"Cleaned markdown with LLM. Input length: {len(input_markdown)}, Output length: {len(final_cleaned_content)}")


def test_clean_markdown_with_llm_input_too_short(mock_llm_logger, mock_appconfig_defaults, monkeypatch):
    # Markdown shorter than MIN_MARKDOWN_LENGTH_FOR_CLEANING should skip LLM
    monkeypatch.setattr(app_config.llm, 'MIN_MARKDOWN_LENGTH_FOR_CLEANING', 50)
    input_markdown = "Short: ## text" # Length is < 50
    
    # Ensure LLM related functions are not called
    with patch('viralStoryGenerator.src.llm._make_llm_request') as mock_make_req, \
         patch('viralStoryGenerator.src.llm._pre_process_markdown') as mock_pre_proc, \
         patch('viralStoryGenerator.src.llm._post_process_llm_output') as mock_post_proc:

        result = llm_module.clean_markdown_with_llm(input_markdown)

        assert result == input_markdown # Should return original if too short
        mock_make_req.assert_not_called()
        mock_pre_proc.assert_not_called() # Pre-processing might still be called, or not.
                                         # Based on current code, it's called *before* length check.
                                         # Let's assume it's NOT called if too short for LLM.
                                         # If it is, this test needs adjustment.
                                         # Re-checking code: _pre_process_markdown is called, then length check.
                                         # So _pre_process_markdown *will* be called.
                                         # This means the test setup for _pre_process_markdown needs to be active.
                                         # For simplicity here, if it just returns the input, that's fine.
                                         # Let's refine this specific test.
        # Re-doing this test with proper pre_process patching
        mock_pre_proc.reset_mock() # Reset from any previous auto-patch if any
        mock_pre_proc.return_value = input_markdown # Assume pre_process doesn't change it much for this test

    # New attempt for the too_short test
    with patch('viralStoryGenerator.src.llm._make_llm_request') as mock_make_req_2, \
         patch('viralStoryGenerator.src.llm._pre_process_markdown', return_value=input_markdown) as mock_pre_proc_2, \
         patch('viralStoryGenerator.src.llm._post_process_llm_output') as mock_post_proc_2:
        
        result = llm_module.clean_markdown_with_llm(input_markdown)
        assert result == input_markdown
        mock_pre_proc_2.assert_called_once_with(input_markdown) # Called before length check
        mock_make_req_2.assert_not_called() # LLM call skipped
        mock_post_proc_2.assert_not_called() # Post-processing skipped
        mock_llm_logger.info.assert_any_call(
            f"Markdown length ({len(input_markdown)}) is less than min_length_for_cleaning ({app_config.llm.MIN_MARKDOWN_LENGTH_FOR_CLEANING}). Skipping LLM cleaning."
        )


def test_clean_markdown_with_llm_empty_input(mock_llm_logger, mock_appconfig_defaults):
    input_markdown = ""
    # Patching as in the too_short test
    with patch('viralStoryGenerator.src.llm._make_llm_request') as mock_make_req, \
         patch('viralStoryGenerator.src.llm._pre_process_markdown', return_value="") as mock_pre_proc, \
         patch('viralStoryGenerator.src.llm._post_process_llm_output') as mock_post_proc:

        result = llm_module.clean_markdown_with_llm(input_markdown)
        
        assert result == ""
        mock_pre_proc.assert_called_once_with("")
        mock_make_req.assert_not_called()
        mock_post_proc.assert_not_called()
        mock_llm_logger.info.assert_any_call("Input markdown is empty. Skipping LLM cleaning.")


@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._pre_process_markdown', side_effect=lambda x: x) # Pass through
@patch('viralStoryGenerator.src.llm._post_process_llm_output', side_effect=lambda x: x) # Pass through
def test_clean_markdown_with_llm_make_request_fails(
    mock_post_process, mock_pre_process, mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    input_markdown = "This is valid markdown content long enough for processing."
    mock_make_request.return_value = None # Simulate _make_llm_request failing

    result = llm_module.clean_markdown_with_llm(input_markdown)

    assert result == input_markdown # Should return original markdown if LLM fails
    mock_pre_process.assert_called_once_with(input_markdown)
    mock_make_request.assert_called_once()
    mock_post_process.assert_not_called() # Post-processing skipped
    mock_llm_logger.error.assert_any_call(f"Failed to get LLM response for cleaning markdown. Returning original. Input: {input_markdown[:100]}...")


@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._pre_process_markdown', side_effect=lambda x: x)
@patch('viralStoryGenerator.src.llm._post_process_llm_output', side_effect=lambda x: x) # Important for this test
def test_clean_markdown_with_llm_empty_llm_content(
    mock_post_process, mock_pre_process, mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    input_markdown = "Markdown that will result in empty LLM content."
    # LLM returns empty string in content
    mock_make_request.return_value = {"choices": [{"message": {"content": ""}}]} 

    result = llm_module.clean_markdown_with_llm(input_markdown)

    # If LLM returns empty, and _post_process_llm_output also returns empty from that,
    # the function should return the original markdown.
    assert result == input_markdown 
    mock_pre_process.assert_called_once_with(input_markdown)
    mock_make_request.assert_called_once()
    mock_post_process.assert_called_once_with("") # Called with empty content
    mock_llm_logger.warning.assert_any_call(
        f"LLM returned empty or invalid content for cleaning. Returning original markdown. Input: {input_markdown[:100]}..."
    )


@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._pre_process_markdown', side_effect=lambda x: x)
@patch('viralStoryGenerator.src.llm._post_process_llm_output', side_effect=lambda x: x)
def test_clean_markdown_with_llm_malformed_llm_response(
    mock_post_process, mock_pre_process, mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    input_markdown = "Markdown for malformed LLM response test."
    # LLM returns malformed JSON (missing 'message' or 'content')
    mock_make_request.return_value = {"choices": [{"not_message": "where is content?"}]}

    result = llm_module.clean_markdown_with_llm(input_markdown)

    assert result == input_markdown # Return original
    mock_pre_process.assert_called_once_with(input_markdown)
    mock_make_request.assert_called_once()
    mock_post_process.assert_not_called() # Not called as content extraction fails
    mock_llm_logger.error.assert_any_call(
        f"LLM response for cleaning markdown was malformed or missing content. Returning original. Response: {{'choices': [{{'not_message': 'where is content?'}}]}}, Input: {input_markdown[:100]}..."
    )

# --- Tests for _reformat_text ---

@patch('viralStoryGenerator.src.llm._make_llm_request')
def test_reformat_text_successful(mock_make_request, mock_llm_logger, mock_appconfig_defaults):
    original_text = "This text needs reformatting."
    reformatted_text_from_llm = "### Story Script:\nThis is reformatted.\n### Video Description:\nAlso reformatted."
    
    mock_make_request.return_value = {"choices": [{"message": {"content": reformatted_text_from_llm}}]}

    result = llm_module._reformat_text(original_text)

    assert result == reformatted_text_from_llm
    mock_make_request.assert_called_once()
    args, _ = mock_make_request.call_args
    messages = args[0]
    assert messages[0]["role"] == "system" # Check system prompt for reformatting
    assert "You are an expert in reformatting text into a specific two-part structure" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert original_text in messages[1]["content"]
    mock_llm_logger.info.assert_any_call(f"Reformatted text with LLM. Input length: {len(original_text)}, Output length: {len(reformatted_text_from_llm)}")


@patch('viralStoryGenerator.src.llm._make_llm_request')
def test_reformat_text_llm_returns_no_content(mock_make_request, mock_llm_logger, mock_appconfig_defaults):
    original_text = "Reformat this, but LLM returns nothing."
    mock_make_request.return_value = {"choices": [{"message": {"content": ""}}]} # Empty content

    result = llm_module._reformat_text(original_text)

    assert result is None # Should return None if LLM gives empty content
    mock_make_request.assert_called_once()
    mock_llm_logger.warning.assert_any_call(
        f"LLM returned empty or invalid content during reformatting attempt for input: {original_text[:100]}..."
    )


@patch('viralStoryGenerator.src.llm._make_llm_request')
def test_reformat_text_llm_returns_malformed(mock_make_request, mock_llm_logger, mock_appconfig_defaults):
    original_text = "Reformat this, but LLM returns malformed."
    mock_make_request.return_value = {"choices": [{"no_message_here": "..."}]} # Malformed

    result = llm_module._reformat_text(original_text)

    assert result is None # Should return None
    mock_make_request.assert_called_once()
    mock_llm_logger.error.assert_any_call(
        f"LLM response for reformatting was malformed or missing content. Response: {{'choices': [{{'no_message_here': '...'}}]}}, Input: {original_text[:100]}..."
    )


@patch('viralStoryGenerator.src.llm._make_llm_request')
def test_reformat_text_make_request_fails(mock_make_request, mock_llm_logger, mock_appconfig_defaults):
    original_text = "Reformat this, but _make_llm_request fails."
    mock_make_request.return_value = None # Simulate underlying request failure

    result = llm_module._reformat_text(original_text)

    assert result is None
    mock_make_request.assert_called_once()
    mock_llm_logger.error.assert_any_call(
        f"Failed to get LLM response for reformatting text. Input: {original_text[:100]}..."
    )

# --- Tests for _check_format ---

@pytest.mark.parametrize("input_text, expected_result", [
    ("### Story Script:\nSome story.\n### Video Description:\nSome description.", True),
    ("### Story Script:\nSome story.\n### Video Description:", False), # Empty description
    ("### Story Script:\n\n### Video Description:\nSome description.", False), # Empty story
    ("### Story Script:\n\n### Video Description:\n", False), # Both empty
    ("Story only, no markers.", False), # Missing markers
    ("### Story Script:\nSome story.", False), # Missing description marker
    ("### Video Description:\nSome description.", False), # Missing story marker
    # Fallback markers
    ("[STORY SCRIPT]\nSome story.\n[VIDEO DESCRIPTION]\nSome description.", True),
    ("### Story Script:\nSome story.\n[DESCRIPTION]\nSome description.", True), # Mixed valid and fallback
    ("[Script]\nSome story.\n[Video Details]\nSome description.", True),
    ("Text without any standard separators.", False), # No clear separation
    ("Story content only, no second part marker.", False),
    # Edge cases
    ("", False), # Empty string
    ("### Story Script:\n### Video Description:\n", False), # Markers present but no content after them
    ("### Story Script:\nContent\n### Video Description:\nContent", True),
    ("### Story Script:\nContent\n\n\n### Video Description:\nContent", True), # Extra newlines between sections
    ("   ### Story Script:\nContent\n### Video Description:\nContent   ", True), # Leading/trailing spaces around text
    # Test case where description is very short but present
    ("### Story Script:\nSome story.\n### Video Description:\nA", True), 
])
def test_check_format_various_inputs(input_text, expected_result, mock_appconfig_defaults):
    # _check_format is not a public member, access via module
    is_correct_format = llm_module._check_format(input_text)
    assert is_correct_format == expected_result

# --- Tests for _extract_chain_of_thought ---

@pytest.mark.parametrize("input_text, expected_thought, expected_remaining", [
    ("<think>This is my thought process.</think>\nActual content follows.", 
     "This is my thought process.", 
     "Actual content follows."),
    ("No think block here. Just content.", 
     None, 
     "No think block here. Just content."),
    ("<think>Thought 1.</think><think>Thought 2, but only first is taken.</think>Content.",
     "Thought 1.",
     "<think>Thought 2, but only first is taken.</think>Content."), # Only first think block extracted
    ("  <think>\n  Indented thought with newlines.  \n  </think>  \n  Cleaned content.  ",
     "Indented thought with newlines.", # Thought content is stripped
     "Cleaned content."), # Remaining content is stripped
    ("<think></think>Empty thought block.",
     "", # Empty thought
     "Empty thought block."),
    ("No closing tag <think>Malformed thought.",
     None, # No valid block found
     "No closing tag <think>Malformed thought."),
    ("Text before <think>Thought</think> and text after.",
     "Thought",
     "Text before and text after."), # Text before and after is preserved and concatenated
    ("Multiple lines\n<think>\nMy thought\non multiple lines\n</think>\nRemaining text.",
     "My thought\non multiple lines",
     "Multiple lines\nRemaining text."),
    ("", None, ""), # Empty string
    ("   ", None, "   "), # Whitespace only
    ("<think>  </think>", "", ""), # Whitespace in think block, empty remaining
])
def test_extract_chain_of_thought_various_inputs(input_text, expected_thought, expected_remaining, mock_appconfig_defaults):
    thought, remaining_text = llm_module._extract_chain_of_thought(input_text)
    assert thought == expected_thought
    assert remaining_text == expected_remaining

# --- Tests for process_with_llm ---

@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._extract_chain_of_thought')
@patch('viralStoryGenerator.src.llm._check_format')
@patch('viralStoryGenerator.src.llm._reformat_text') # Not called in this specific test path
@patch('viralStoryGenerator.src.llm._post_process_llm_output')
def test_process_with_llm_success_first_try(
    mock_post_process, mock_reformat_text, mock_check_format, mock_extract_cot, 
    mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    topic = "Test Topic"
    system_prompt = "System prompt for test"
    user_prompt_template = "User prompt for {topic}"
    
    raw_llm_response_content = "<think>Thinking...</think>### Story Script:\nStory here.\n### Video Description:\nDesc here."
    extracted_thought = "Thinking..."
    text_after_cot = "### Story Script:\nStory here.\n### Video Description:\nDesc here."
    post_processed_text = "Story here.\n### Video Description:\nDesc here." # Assume post_process also cleans up a bit

    mock_make_request.return_value = {"choices": [{"message": {"content": raw_llm_response_content}}]}
    mock_extract_cot.return_value = (extracted_thought, text_after_cot)
    mock_check_format.return_value = True # Format is correct on first try
    mock_post_process.return_value = post_processed_text # Simulate post-processing

    result_text, result_thought = llm_module.process_with_llm(topic, system_prompt, user_prompt_template)

    assert result_text == post_processed_text
    assert result_thought == extracted_thought
    
    mock_make_request.assert_called_once() # Called once
    args, _ = mock_make_request.call_args
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt
    assert messages[1]["role"] == "user"
    assert topic in messages[1]["content"] # Check if topic was formatted into user_prompt
    
    mock_extract_cot.assert_called_once_with(raw_llm_response_content)
    mock_check_format.assert_called_once_with(text_after_cot)
    mock_reformat_text.assert_not_called() # Not called because format was correct
    mock_post_process.assert_called_once_with(text_after_cot)
    mock_llm_logger.info.assert_any_call(f"LLM processing for topic '{topic}' successful on first try.")


@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._extract_chain_of_thought')
@patch('viralStoryGenerator.src.llm._check_format')
@patch('viralStoryGenerator.src.llm._reformat_text')
@patch('viralStoryGenerator.src.llm._post_process_llm_output')
def test_process_with_llm_success_after_reformatting(
    mock_post_process, mock_reformat_text, mock_check_format, mock_extract_cot,
    mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    topic = "Test Topic Reformat"
    raw_llm_response_content_initial = "<think>Thinking initial.</think>This needs reformatting."
    extracted_thought_initial = "Thinking initial."
    text_after_cot_initial = "This needs reformatting." # Incorrect format
    
    reformatted_text_from_llm = "<think>Thinking reformat.</think>### Story Script:\nReformatted story.\n### Video Description:\nReformatted desc."
    extracted_thought_reformat = "Thinking reformat."
    text_after_cot_reformat = "### Story Script:\nReformatted story.\n### Video Description:\nReformatted desc." # Correct format
    
    post_processed_text_final = "Reformatted story.\n### Video Description:\nReformatted desc."

    # Initial LLM call
    mock_make_request.return_value = {"choices": [{"message": {"content": raw_llm_response_content_initial}}]}
    mock_extract_cot.side_effect = [
        (extracted_thought_initial, text_after_cot_initial), # For initial call
        (extracted_thought_reformat, text_after_cot_reformat)  # For reformat call (if _reformat_text also uses it)
    ]
    mock_check_format.side_effect = [False, True] # Fails first, succeeds after reformat
    mock_reformat_text.return_value = reformatted_text_from_llm # _reformat_text returns the raw reformatted string
    mock_post_process.return_value = post_processed_text_final

    result_text, result_thought = llm_module.process_with_llm(topic, "sys_prompt", "user_prompt_{topic}")

    assert result_text == post_processed_text_final
    assert result_thought == extracted_thought_initial # Should be the thought from the *initial* response
    
    assert mock_make_request.call_count == 1 # Only initial _make_llm_request is from process_with_llm directly
                                             # _reformat_text has its own _make_llm_request call, which is part of its mock here.
    mock_extract_cot.assert_any_call(raw_llm_response_content_initial)
    mock_extract_cot.assert_any_call(reformatted_text_from_llm) # Called again by process_with_llm on the output of _reformat_text
    
    assert mock_check_format.call_count == 2
    mock_check_format.assert_any_call(text_after_cot_initial)
    mock_check_format.assert_any_call(text_after_cot_reformat)
    
    mock_reformat_text.assert_called_once_with(text_after_cot_initial)
    mock_post_process.assert_called_once_with(text_after_cot_reformat)
    mock_llm_logger.info.assert_any_call(f"LLM output for topic '{topic}' required reformatting. Successful after reformat.")


@patch('viralStoryGenerator.src.llm._make_llm_request', return_value=None) # Initial LLM call fails
@patch('viralStoryGenerator.src.llm._extract_chain_of_thought') # Should not be called
def test_process_with_llm_initial_make_request_fails(
    mock_extract_cot, mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    topic = "Test Topic LLM Fail"
    result_text, result_thought = llm_module.process_with_llm(topic, "sys", "user_{topic}")

    assert result_text is None
    assert result_thought is None
    mock_make_request.assert_called_once()
    mock_extract_cot.assert_not_called()
    mock_llm_logger.error.assert_any_call(f"LLM processing failed for topic '{topic}' after initial generation and potential reformat.")


@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._extract_chain_of_thought')
@patch('viralStoryGenerator.src.llm._check_format', return_value=False) # Always needs reformatting
@patch('viralStoryGenerator.src.llm._reformat_text', return_value=None) # Reformatting itself fails
def test_process_with_llm_reformat_text_fails(
    mock_reformat_text, mock_check_format, mock_extract_cot, 
    mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    topic = "Test Topic Reformat Fail"
    raw_llm_response_content = "<think>Thought.</think>Needs reformat."
    mock_make_request.return_value = {"choices": [{"message": {"content": raw_llm_response_content}}]}
    mock_extract_cot.return_value = ("Thought.", "Needs reformat.")

    result_text, result_thought = llm_module.process_with_llm(topic, "sys", "user_{topic}")
    
    assert result_text is None # Overall process fails
    assert result_thought == "Thought." # Thought from initial attempt is kept
    mock_make_request.assert_called_once()
    mock_extract_cot.assert_called_once_with(raw_llm_response_content)
    mock_check_format.assert_called_once_with("Needs reformat.")
    mock_reformat_text.assert_called_once_with("Needs reformat.")
    mock_llm_logger.warning.assert_any_call(f"Reformatting failed for topic '{topic}'.")
    mock_llm_logger.error.assert_any_call(f"LLM processing failed for topic '{topic}' after initial generation and potential reformat.")


@patch('viralStoryGenerator.src.llm._make_llm_request')
@patch('viralStoryGenerator.src.llm._extract_chain_of_thought')
@patch('viralStoryGenerator.src.llm._check_format', return_value=False)
@patch('viralStoryGenerator.src.llm._reformat_text') # Returns empty after reformat
@patch('viralStoryGenerator.src.llm._post_process_llm_output') # To see if it's called
def test_process_with_llm_empty_content_after_reformat(
    mock_post_process, mock_reformat_text, mock_check_format, mock_extract_cot,
    mock_make_request, mock_llm_logger, mock_appconfig_defaults
):
    topic = "Test Topic Empty Reformat"
    # Initial LLM response
    mock_make_request.return_value = {"choices": [{"message": {"content": "<think>T</think>Bad format"}}]}
    # First _extract_chain_of_thought call
    mock_extract_cot.side_effect = [
        ("T", "Bad format"), # From initial response
        (None, "")           # From reformatted response (which is just empty)
    ]
    # _reformat_text returns an empty string
    mock_reformat_text.return_value = "" 
    # _check_format on "" from reformat will be False
    # mock_check_format is already return_value=False, will be called twice.

    result_text, result_thought = llm_module.process_with_llm(topic, "sys", "user_{topic}")

    assert result_text is None
    assert result_thought == "T" # Thought from initial attempt
    
    mock_reformat_text.assert_called_once_with("Bad format")
    # _extract_chain_of_thought called for initial response AND for the (empty) reformatted response
    assert mock_extract_cot.call_count == 2
    mock_extract_cot.assert_any_call("<think>T</think>Bad format")
    mock_extract_cot.assert_any_call("")
    
    # _check_format called for "Bad format" and for ""
    assert mock_check_format.call_count == 2
    mock_check_format.assert_any_call("Bad format")
    mock_check_format.assert_any_call("") 
    
    mock_post_process.assert_not_called() # Not called if content becomes empty after reformat and CoT extraction
    mock_llm_logger.warning.assert_any_call(f"LLM content for topic '{topic}' is empty after reformatting and CoT extraction.")
    mock_llm_logger.error.assert_any_call(f"LLM processing failed for topic '{topic}' after initial generation and potential reformat.")


# Input validation tests for process_with_llm
@pytest.mark.parametrize("test_topic, test_sys_prompt, test_user_template, expected_log_msg", [
    ("", "sys", "user_{topic}", "Input topic cannot be empty."),
    (" ", "sys", "user_{topic}", "Input topic cannot be empty."),
    ("Valid Topic", "", "user_{topic}", "System prompt cannot be empty."),
    ("Valid Topic", "   ", "user_{topic}", "System prompt cannot be empty."),
    ("Valid Topic", "sys", "", "User prompt template cannot be empty."),
    ("Valid Topic", "sys", "  ", "User prompt template cannot be empty."),
])
@patch('viralStoryGenerator.src.llm._make_llm_request') # Should not be called
def test_process_with_llm_input_validation(
    mock_make_request, test_topic, test_sys_prompt, test_user_template, expected_log_msg, 
    mock_llm_logger, mock_appconfig_defaults
):
    result_text, result_thought = llm_module.process_with_llm(test_topic, test_sys_prompt, test_user_template)
    assert result_text is None
    assert result_thought is None
    mock_make_request.assert_not_called()
    mock_llm_logger.error.assert_called_once_with(expected_log_msg)
