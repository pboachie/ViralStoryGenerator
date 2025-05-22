import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from viralStoryGenerator.src.api import app, app_config
from viralStoryGenerator.src.schemas import StoryResponse

client = TestClient(app)

@patch('viralStoryGenerator.utils.health_check.get_service_status')
def test_health_check(mock_get_service_status):
    mock_get_service_status.return_value = {"status": "ok"}
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch('prometheus_client.generate_latest')
def test_metrics(mock_generate_latest):
    mock_generate_latest.return_value = b"some metrics"
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.content == b"some metrics"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


# Tests for POST /api/stories
@patch('viralStoryGenerator.src.api_handlers.create_story_task')
@patch('viralStoryGenerator.utils.security.is_valid_voice_id')
@patch('viralStoryGenerator.utils.security.validate_path_component')
@patch('viralStoryGenerator.utils.security.is_file_in_directory')
@patch('os.path.exists')
@patch('os.path.isdir')
def test_create_story_successful_all_params(
    mock_isdir,
    mock_exists,
    mock_is_file_in_directory,
    mock_validate_path_component,
    mock_is_valid_voice_id,
    mock_create_story_task
):
    # Mock behavior
    mock_create_story_task.return_value = StoryResponse(task_id="test_task_id", topic="Test Topic", status="pending")
    mock_is_valid_voice_id.return_value = True
    mock_validate_path_component.return_value = "valid_folder" # Assume it returns the validated component
    mock_is_file_in_directory.return_value = True
    mock_exists.return_value = True
    mock_isdir.return_value = True

    payload = {
        "topic": "Test Topic",
        "sources_folder": "valid_folder",
        "voice_id": "valid_voice"
    }
    response = client.post("/api/stories", json=payload)

    assert response.status_code == 202
    response_data = response.json()
    assert response_data["task_id"] == "test_task_id"
    assert response_data["topic"] == "Test Topic"
    assert response_data["status"] == "pending"

    mock_create_story_task.assert_called_once()
    mock_is_valid_voice_id.assert_called_once_with("valid_voice")
    mock_validate_path_component.assert_called_once_with("valid_folder")
    # Base path for sources is app_config.storage.SOURCES_DIR
    # We need to ensure os.path.join is called correctly by the endpoint
    expected_sources_path = os.path.join(app_config.storage.SOURCES_DIR, "valid_folder")
    mock_exists.assert_called_once_with(expected_sources_path)
    mock_isdir.assert_called_once_with(expected_sources_path)
    mock_is_file_in_directory.assert_called_once_with(app_config.storage.SOURCES_DIR, expected_sources_path)


@patch('viralStoryGenerator.src.api_handlers.create_story_task')
def test_create_story_successful_only_topic(mock_create_story_task):
    mock_create_story_task.return_value = StoryResponse(task_id="test_task_id_2", topic="Another Topic", status="queued")

    payload = {"topic": "Another Topic"}
    response = client.post("/api/stories", json=payload)

    assert response.status_code == 202
    response_data = response.json()
    assert response_data["task_id"] == "test_task_id_2"
    assert response_data["topic"] == "Another Topic"
    assert response_data["status"] == "queued"
    mock_create_story_task.assert_called_once()


# Invalid topic tests
def test_create_story_invalid_topic_empty():
    # This test assumes that the Pydantic model StoryRequest has topic: constr(min_length=1)
    # or similar validation. If topic: str, FastAPI/Pydantic won't raise 422 for an empty string.
    # The subtask requires asserting 400 or 422. 422 is typical for Pydantic validation.
    payload = {"topic": "", "sources_folder": None, "voice_id": None}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 422 # Expecting Pydantic validation error for empty topic

# Test for topic that becomes empty after sanitization (conceptual)
# This test is tricky because sanitize_input is not directly part of the endpoint's
# explicit logic before Pydantic validation. If sanitize_input is called
# *within* the create_story_task handler, we can't test it here with a mock.
# However, if the requirement is that the API *should* reject such topics,
# it implies a pre-validation step or a more robust Pydantic model.
# For now, we'll rely on the Pydantic model for "empty topic" and skip direct
# testing of sanitize_input leading to empty unless it's part of endpoint logic.

# Invalid sources_folder tests
@patch('viralStoryGenerator.src.api.validate_path_component')
def test_create_story_invalid_sources_folder_chars(mock_validate_path_component):
    mock_validate_path_component.side_effect = ValueError("Invalid characters in folder name test")
    payload = {"topic": "Valid Topic", "sources_folder": "../invalid_folder"}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 400
    assert "Invalid characters in folder name test" in response.json()["detail"]
    mock_validate_path_component.assert_called_once_with("../invalid_folder")

@patch('os.path.exists')
@patch('viralStoryGenerator.src.api.validate_path_component')
@patch('viralStoryGenerator.src.api.is_file_in_directory')
def test_create_story_invalid_sources_folder_not_exists(
    mock_is_file_in_directory, mock_validate_path_component, mock_exists
):
    mock_validate_path_component.return_value = "some_folder"
    # Assume the path up to app_config.storage.SOURCES_DIR is valid and secure
    # and "some_folder" itself is also fine character-wise.
    # is_file_in_directory checks if the *resolved* path is within SOURCES_DIR.
    expected_path = os.path.join(app_config.storage.SOURCES_DIR, "some_folder")
    mock_is_file_in_directory.return_value = True # Path is constructed correctly and within base.
    mock_exists.return_value = False # Folder does not exist at the full path.

    payload = {"topic": "Valid Topic", "sources_folder": "some_folder"}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 404
    assert "Sources folder not found" in response.json()["detail"]
    mock_validate_path_component.assert_called_once_with("some_folder")
    mock_is_file_in_directory.assert_called_once_with(app_config.storage.SOURCES_DIR, expected_path)
    mock_exists.assert_called_once_with(expected_path)


@patch('os.path.isdir')
@patch('os.path.exists')
@patch('viralStoryGenerator.src.api.validate_path_component')
@patch('viralStoryGenerator.src.api.is_file_in_directory')
def test_create_story_invalid_sources_folder_not_dir(
    mock_is_file_in_directory, mock_validate_path_component, mock_exists, mock_isdir
):
    mock_validate_path_component.return_value = "file_not_folder"
    expected_path = os.path.join(app_config.storage.SOURCES_DIR, "file_not_folder")
    mock_is_file_in_directory.return_value = True # Path is constructed correctly and within base.
    mock_exists.return_value = True # Path exists
    mock_isdir.return_value = False # Path is not a directory

    payload = {"topic": "Valid Topic", "sources_folder": "file_not_folder"}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 400 # As per current endpoint logic
    assert "Sources folder is not a directory" in response.json()["detail"]
    mock_validate_path_component.assert_called_once_with("file_not_folder")
    mock_is_file_in_directory.assert_called_once_with(app_config.storage.SOURCES_DIR, expected_path)
    mock_exists.assert_called_once_with(expected_path)
    mock_isdir.assert_called_once_with(expected_path)


@patch('viralStoryGenerator.src.api.is_file_in_directory')
@patch('viralStoryGenerator.src.api.validate_path_component')
def test_create_story_invalid_sources_folder_traversal(
    mock_validate_path_component, mock_is_file_in_directory
):
    # Scenario: validate_path_component allows the component name, but the resulting path
    # after joining with SOURCES_DIR is outside SOURCES_DIR.
    user_input_folder = "outside_folder"
    mock_validate_path_component.return_value = user_input_folder # Assume "outside_folder" is fine char-wise
    
    # Simulate that the resolved path is outside the allowed directory
    # This means is_file_in_directory will be called with the base path and the fully resolved path.
    # e.g. SOURCES_DIR = /app/data/sources, user_input_folder = ../../etc
    # resolved_path = /app/data/sources/../../etc = /etc
    # is_file_in_directory("/app/data/sources", "/etc") -> False
    resolved_path_attempt = os.path.join(app_config.storage.SOURCES_DIR, user_input_folder)
    mock_is_file_in_directory.return_value = False

    payload = {"topic": "Valid Topic", "sources_folder": user_input_folder}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 403
    assert "Path traversal attempt detected" in response.json()["detail"]
    mock_validate_path_component.assert_called_once_with(user_input_folder)
    mock_is_file_in_directory.assert_called_once_with(app_config.storage.SOURCES_DIR, resolved_path_attempt)


# Invalid voice_id tests
@patch('viralStoryGenerator.src.api.is_valid_voice_id')
def test_create_story_invalid_voice_id(mock_is_valid_voice_id):
    mock_is_valid_voice_id.return_value = False
    payload = {"topic": "Valid Topic", "voice_id": "invalid_voice_123"}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 400
    assert "Invalid voice ID" in response.json()["detail"]
    mock_is_valid_voice_id.assert_called_once_with("invalid_voice_123")

# Authentication Tests (API Key)
# Need to patch app_config for these tests
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey")
def test_create_story_auth_no_api_key():
    payload = {"topic": "Test Topic"}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]

@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey")
def test_create_story_auth_invalid_api_key():
    payload = {"topic": "Test Topic"}
    headers = {"X-API-Key": "wrongapikey"}
    response = client.post("/api/stories", json=payload, headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]

@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey")
@patch('viralStoryGenerator.src.api_handlers.create_story_task') # Mock handler as auth should pass
def test_create_story_auth_valid_api_key(mock_create_story_task):
    mock_create_story_task.return_value = StoryResponse(task_id="auth_task", topic="Auth Topic", status="pending")
    payload = {"topic": "Auth Topic"}
    headers = {"X-API-Key": "testapikey"}
    response = client.post("/api/stories", json=payload, headers=headers)
    assert response.status_code == 202
    assert response.json()["topic"] == "Auth Topic"
    mock_create_story_task.assert_called_once()

@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # API key auth disabled
@patch('viralStoryGenerator.src.api_handlers.create_story_task') # Mock handler
def test_create_story_auth_disabled_no_key(mock_create_story_task):
    mock_create_story_task.return_value = StoryResponse(task_id="no_auth_task", topic="No Auth Topic", status="pending")
    payload = {"topic": "No Auth Topic"}
    response = client.post("/api/stories", json=payload)
    assert response.status_code == 202
    assert response.json()["topic"] == "No Auth Topic"
    mock_create_story_task.assert_called_once()


# Tests for GET /api/stories/{task_id}
@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_successful(mock_is_valid_uuid, mock_get_task_status):
    mock_is_valid_uuid.return_value = True
    task_id = "valid-uuid-123"
    expected_status = {
        "task_id": task_id,
        "status": "completed",
        "story_script": "A great story.",
        "video_url": "http://example.com/video.mp4",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:05:00Z"
    }
    mock_get_task_status.return_value = expected_status

    response = client.get(f"/api/stories/{task_id}")

    assert response.status_code == 200
    assert response.json() == expected_status
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_get_task_status.assert_called_once_with(task_id)


# Tests for GET /api/files/{task_id}/{file_type_key}

# Expected filenames and content types
EXPECTED_FILENAMES = {
    "audio": "story_audio.mp3",
    "story": "story_script.txt",
    "storyboard": "story_board.json",
    "metadata": "metadata.json",
}

EXPECTED_CONTENT_TYPES = {
    "audio": "audio/mpeg",
    "story": "text/plain",
    "storyboard": "application/json",
    "metadata": "application/json",
}

MOCK_FILE_CONTENT = {
    "audio": b"dummy audio content" * 10, # ~190 bytes for range requests
    "story": "This is a dummy story script.",
    "storyboard": {"scene": 1, "description": "Dummy storyboard"},
    "metadata": {"title": "Dummy Story", "author": "Test"},
}

# Helper to mock aiofiles.open
async def mock_aiofiles_open_read(mock_content):
    amock = MagicMock()
    amock.read.return_value = mock_content
    # For async context manager
    aenter_mock = MagicMock()
    aenter_mock.__aenter__.return_value = amock
    aenter_mock.__aexit__.return_value = MagicMock()
    return aenter_mock

@pytest.mark.parametrize("file_type_key", ["audio", "story", "storyboard", "metadata"])
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename')
@patch('viralStoryGenerator.utils.security.is_file_in_directory')
@patch('os.path.exists')
@patch('os.path.getsize')
@patch('aiofiles.open', new_callable=MagicMock) # Mock the module itself
def test_get_file_successful_local(
    mock_aio_open,
    mock_getsize,
    mock_exists,
    mock_is_file_in_directory,
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file,
    file_type_key
):
    task_id = "valid-uuid-for-file"
    expected_filename = EXPECTED_FILENAMES[file_type_key]
    mock_local_path = f"/tmp/mock_storage/{task_id}/{expected_filename}"
    expected_content_type = EXPECTED_CONTENT_TYPES[file_type_key]
    mock_content = MOCK_FILE_CONTENT[file_type_key]
    if isinstance(mock_content, dict) or isinstance(mock_content, list):
        mock_content_bytes = os.fsencode(str(mock_content))
    elif isinstance(mock_content, str):
        mock_content_bytes = mock_content.encode('utf-8')
    else:
        mock_content_bytes = mock_content


    mock_is_valid_uuid.return_value = True
    mock_is_safe_filename.return_value = True
    # StorageManager.serve_file is a method, so it needs to be part of a class or instance mock
    # Assuming it's called on an instance `storage_manager` imported in api.py
    # If StorageManager is instantiated in api.py, we might need to patch the class's method
    # For now, assuming `viralStoryGenerator.utils.storage_manager.StorageManager.serve_file` is correct
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}
    mock_exists.return_value = True
    mock_getsize.return_value = len(mock_content_bytes)
    mock_is_file_in_directory.return_value = True

    # Configure aiofiles.open mock
    # mock_aio_open.return_value is what `with aiofiles.open(...) as f:` gives to `f`.
    # This needs to be an async context manager.
    async def async_magic_mock_open(*args, **kwargs):
        amock = MagicMock()
        amock.read.return_value = mock_content_bytes # Read returns bytes
        
        async_context_mock = MagicMock()
        async_context_mock.__aenter__.return_value = amock
        async_context_mock.__aexit__ = MagicMock(return_value=False) # Ensure it's awaitable
        return async_context_mock

    mock_aio_open.side_effect = async_magic_mock_open


    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(expected_content_type)
    
    # For non-audio, check content. For audio, content check is harder due to streaming/chunking.
    if file_type_key != "audio":
         if isinstance(MOCK_FILE_CONTENT[file_type_key], (dict, list)):
             assert response.json() == MOCK_FILE_CONTENT[file_type_key]
         else:
             assert response.text == MOCK_FILE_CONTENT[file_type_key]
    else:
        # For audio, just check that the content length matches if not a range request
        assert int(response.headers["content-length"]) == len(mock_content_bytes)
        assert response.content == mock_content_bytes


    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)
    # The filename passed to is_safe_filename is constructed inside the endpoint
    # It should be `task_id + "_" + file_type_key + extension`
    # For now, we assume it's called with the `expected_filename` that serve_file returns
    mock_is_safe_filename.assert_called_once_with(expected_filename)
    mock_exists.assert_called_once_with(mock_local_path)
    mock_is_file_in_directory.assert_called_once() # Check args if specific base path is known


@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename') # Added to complete mocks for this path
def test_get_file_successful_cloud_redirect(
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file
):
    task_id = "valid-uuid-for-cloud"
    file_type_key = "audio" # Any valid key
    mock_cloud_url = "http://mock-cloud-storage.com/file.mp3"
    expected_filename = EXPECTED_FILENAMES[file_type_key]


    mock_is_valid_uuid.return_value = True
    mock_is_safe_filename.return_value = True # Filename must be safe before returning local path or URL
    mock_serve_file.return_value = {"url": mock_cloud_url, "filename": expected_filename}

    response = client.get(f"/api/files/{task_id}/{file_type_key}", allow_redirects=False) # Important for 307

    assert response.status_code == 307
    assert response.headers["location"] == mock_cloud_url
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)
    # is_safe_filename might be called if the endpoint logic tries to build a local filename first
    # or if it validates the filename part of the URL.
    # Based on the description, serve_file returns filename, then it's checked.
    mock_is_safe_filename.assert_called_once_with(expected_filename)


# Error scenarios for GET /api/files/{task_id}/{file_type_key}

@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_file_invalid_task_id_format(mock_is_valid_uuid):
    mock_is_valid_uuid.return_value = False
    task_id = "not-a-uuid"
    file_type_key = "audio" # Any valid key

    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 400
    assert "Invalid task_id format" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)


@patch('viralStoryGenerator.utils.security.is_valid_uuid') # Must be valid to reach file_type_key check
def test_get_file_invalid_file_type_key(mock_is_valid_uuid):
    mock_is_valid_uuid.return_value = True
    task_id = "valid-uuid"
    invalid_file_type_key = "invalidtype"

    response = client.get(f"/api/files/{task_id}/{invalid_file_type_key}")

    assert response.status_code == 400 # Or 422 if using Pydantic Enum for file_type_key in path
    assert "Invalid file type key" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)


@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename')
def test_get_file_unsafe_filename(
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file
):
    task_id = "valid-uuid-unsafe"
    file_type_key = "story"
    # serve_file returns a filename that is_safe_filename will deem unsafe
    # This filename is generated internally by serve_file or passed through it.
    unsafe_filename = "../../../etc/passwd" # Example of an unsafe filename
    
    mock_is_valid_uuid.return_value = True
    # serve_file returns a dict that includes the filename to be checked
    mock_serve_file.return_value = {"local_path": f"/tmp/{unsafe_filename}", "filename": unsafe_filename}
    mock_is_safe_filename.return_value = False # This is the crucial mock for this test

    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 400
    assert "Unsafe filename detected" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)
    mock_is_safe_filename.assert_called_once_with(unsafe_filename)


@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_file_not_found_by_storage_manager(mock_is_valid_uuid, mock_serve_file):
    task_id = "valid-uuid-no-file"
    file_type_key = "metadata"
    
    mock_is_valid_uuid.return_value = True
    mock_serve_file.return_value = None # Simulate storage_manager not finding the file entry

    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 404
    assert "File record not found or not accessible" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)


@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename')
@patch('os.path.exists')
def test_get_file_local_path_not_exists(
    mock_exists,
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file
):
    task_id = "valid-uuid-local-miss"
    file_type_key = "storyboard"
    mock_local_path = f"/tmp/mock_storage/{task_id}/story_board.json"
    expected_filename = EXPECTED_FILENAMES[file_type_key]

    mock_is_valid_uuid.return_value = True
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}
    mock_is_safe_filename.return_value = True # Filename itself is safe
    mock_exists.return_value = False # But the file doesn't exist on disk

    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 404
    assert "Local file not found" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)
    mock_is_safe_filename.assert_called_once_with(expected_filename)
    mock_exists.assert_called_once_with(mock_local_path)


@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename')
@patch('os.path.exists')
@patch('viralStoryGenerator.utils.security.is_file_in_directory')
def test_get_file_path_traversal_security(
    mock_is_file_in_directory,
    mock_exists,
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file
):
    task_id = "valid-uuid-traversal"
    file_type_key = "audio"
    mock_local_path = f"/tmp/mock_storage/{task_id}/story_audio.mp3" # Path may look okay initially
    expected_filename = EXPECTED_FILENAMES[file_type_key]

    mock_is_valid_uuid.return_value = True
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}
    mock_is_safe_filename.return_value = True # Filename itself is safe
    mock_exists.return_value = True # File exists
    mock_is_file_in_directory.return_value = False # Crucial: path is outside designated storage area

    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 403
    assert "File access forbidden" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)
    mock_is_safe_filename.assert_called_once_with(expected_filename)
    mock_exists.assert_called_once_with(mock_local_path)
    # The actual base path used in is_file_in_directory depends on app_config.storage.LOCAL_STORAGE_PATH
    # We can mock app_config or ensure the call happens.
    # For now, just checking it's called. A more precise test would check its args.
    mock_is_file_in_directory.assert_called_once()


# Audio Range Request Tests
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename')
@patch('viralStoryGenerator.utils.security.is_file_in_directory')
@patch('os.path.exists')
@patch('os.path.getsize')
@patch('aiofiles.open', new_callable=MagicMock)
def test_get_file_audio_range_request_valid(
    mock_aio_open,
    mock_getsize,
    mock_exists,
    mock_is_file_in_directory,
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file
):
    task_id = "valid-uuid-audio-range"
    file_type_key = "audio"
    expected_filename = EXPECTED_FILENAMES[file_type_key]
    mock_local_path = f"/tmp/mock_storage/{task_id}/{expected_filename}"
    mock_content_bytes = MOCK_FILE_CONTENT[file_type_key] # dummy audio content * 10
    total_size = len(mock_content_bytes) # 190

    mock_is_valid_uuid.return_value = True
    mock_is_safe_filename.return_value = True
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}
    mock_exists.return_value = True
    mock_getsize.return_value = total_size
    mock_is_file_in_directory.return_value = True

    # Configure aiofiles.open mock for range request
    range_start, range_end = 0, 99 # Requesting first 100 bytes
    requested_length = (range_end - range_start) + 1
    
    async def async_magic_mock_open_range(*args, **kwargs):
        file_mock = MagicMock()
        file_mock.seek.return_value = None # Mock seek if it's used by StreamingResponse logic
        # Simulate reading only the requested range
        file_mock.read.return_value = mock_content_bytes[range_start : range_end + 1]
        
        async_context_mock = MagicMock()
        async_context_mock.__aenter__.return_value = file_mock
        async_context_mock.__aexit__ = MagicMock(return_value=False)
        return async_context_mock

    mock_aio_open.side_effect = async_magic_mock_open_range

    headers = {"Range": f"bytes={range_start}-{range_end}"}
    response = client.get(f"/api/files/{task_id}/{file_type_key}", headers=headers)

    assert response.status_code == 206
    assert response.headers["content-type"].startswith(EXPECTED_CONTENT_TYPES[file_type_key])
    assert response.headers["content-length"] == str(requested_length)
    assert response.headers["content-range"] == f"bytes {range_start}-{range_end}/{total_size}"
    assert response.content == mock_content_bytes[range_start : range_end + 1]

    mock_aio_open.assert_called_once_with(mock_local_path, "rb")


@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename')
@patch('viralStoryGenerator.utils.security.is_file_in_directory')
@patch('os.path.exists')
@patch('os.path.getsize')
def test_get_file_audio_range_request_invalid(
    mock_getsize,
    mock_exists,
    mock_is_file_in_directory,
    mock_is_safe_filename,
    mock_is_valid_uuid,
    mock_serve_file
):
    task_id = "valid-uuid-audio-range-invalid"
    file_type_key = "audio"
    expected_filename = EXPECTED_FILENAMES[file_type_key]
    mock_local_path = f"/tmp/mock_storage/{task_id}/{expected_filename}"
    total_size = 190 # From MOCK_FILE_CONTENT["audio"]

    mock_is_valid_uuid.return_value = True
    mock_is_safe_filename.return_value = True
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}
    mock_exists.return_value = True
    mock_getsize.return_value = total_size
    mock_is_file_in_directory.return_value = True

    # Range is unsatisfiable (e.g., starts beyond the file size)
    headers = {"Range": f"bytes={total_size+10}-{total_size+100}"}
    response = client.get(f"/api/files/{task_id}/{file_type_key}", headers=headers)

    assert response.status_code == 416 # Range Not Satisfiable
    assert response.headers.get("content-range") == f"bytes */{total_size}"


# Authentication Tests for GET /api/files endpoint
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey_files")
@patch('viralStoryGenerator.utils.security.is_valid_uuid') # To ensure it's not called if auth fails
def test_get_file_auth_no_api_key(mock_is_valid_uuid):
    task_id = "some-uuid-no-auth-file"
    file_type_key = "story"
    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_is_valid_uuid.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey_files")
@patch('viralStoryGenerator.utils.security.is_valid_uuid') # To ensure it's not called
def test_get_file_auth_invalid_api_key(mock_is_valid_uuid):
    task_id = "some-uuid-invalid-auth-file"
    file_type_key = "metadata"
    headers = {"X-API-Key": "wrongapikey_files"}
    response = client.get(f"/api/files/{task_id}/{file_type_key}", headers=headers)

    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_is_valid_uuid.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey_files")
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file') # Mocked to avoid actual file ops
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
# Add other necessary mocks for a "successful" call after auth
@patch('viralStoryGenerator.utils.security.is_safe_filename', return_value=True)
@patch('os.path.exists', return_value=True)
@patch('os.path.getsize', return_value=100)
@patch('viralStoryGenerator.utils.security.is_file_in_directory', return_value=True)
@patch('aiofiles.open', new_callable=MagicMock)
def test_get_file_auth_valid_api_key(
    mock_aio_open, mock_is_file_in_dir, mock_getsize, mock_exists, mock_is_safe_filename,
    mock_is_valid_uuid, mock_serve_file
):
    task_id = "valid-uuid-authed-file"
    file_type_key = "story"
    expected_filename = EXPECTED_FILENAMES[file_type_key]
    mock_local_path = f"/tmp/{expected_filename}"
    mock_content = MOCK_FILE_CONTENT[file_type_key].encode('utf-8')

    mock_is_valid_uuid.return_value = True
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}
    
    async def async_magic_mock_open_valid(*args, **kwargs):
        amock = MagicMock()
        amock.read.return_value = mock_content
        async_context_mock = MagicMock()
        async_context_mock.__aenter__.return_value = amock
        async_context_mock.__aexit__ = MagicMock(return_value=False)
        return async_context_mock
    mock_aio_open.side_effect = async_magic_mock_open_valid
    
    headers = {"X-API-Key": "testapikey_files"}
    response = client.get(f"/api/files/{task_id}/{file_type_key}", headers=headers)

    assert response.status_code == 200 # Assuming it would serve the file
    assert response.text == MOCK_FILE_CONTENT[file_type_key]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.serve_file') # Mocked
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('viralStoryGenerator.utils.security.is_safe_filename', return_value=True)
@patch('os.path.exists', return_value=True)
@patch('os.path.getsize', return_value=100)
@patch('viralStoryGenerator.utils.security.is_file_in_directory', return_value=True)
@patch('aiofiles.open', new_callable=MagicMock)
def test_get_file_auth_disabled_no_key(
    mock_aio_open, mock_is_file_in_dir, mock_getsize, mock_exists, mock_is_safe_filename,
    mock_is_valid_uuid, mock_serve_file
):
    task_id = "valid-uuid-auth-disabled-file"
    file_type_key = "metadata"
    expected_filename = EXPECTED_FILENAMES[file_type_key]
    mock_local_path = f"/tmp/{expected_filename}"
    mock_content_json = MOCK_FILE_CONTENT[file_type_key]
    mock_content_bytes = os.fsencode(str(mock_content_json))


    mock_is_valid_uuid.return_value = True
    mock_serve_file.return_value = {"local_path": mock_local_path, "filename": expected_filename}

    async def async_magic_mock_open_disabled(*args, **kwargs):
        amock = MagicMock()
        amock.read.return_value = mock_content_bytes
        async_context_mock = MagicMock()
        async_context_mock.__aenter__.return_value = amock
        async_context_mock.__aexit__ = MagicMock(return_value=False)
        return async_context_mock
    mock_aio_open.side_effect = async_magic_mock_open_disabled

    response = client.get(f"/api/files/{task_id}/{file_type_key}")

    assert response.status_code == 200
    assert response.json() == mock_content_json
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_serve_file.assert_called_once_with(task_id, file_type_key)


# Tests for POST /api/generate
# Mock for the message broker
mock_message_broker = MagicMock()
async def mock_publish_message_async(payload, routing_key):
    # This function will be the side_effect for the mocked publish_message
    # It can be customized per test to return a message_id or raise an exception
    if hasattr(mock_publish_message_async, 'custom_side_effect'):
        return mock_publish_message_async.custom_side_effect(payload, routing_key)
    return "mock_message_id_" + str(payload.get("job_id", "default"))

mock_message_broker.publish_message = MagicMock(side_effect=mock_publish_message_async)


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id')
@patch('viralStoryGenerator.utils.security.sanitize_input')
@patch('uuid.uuid4') # To control job_id generation
def test_generate_job_successful_all_params(
    mock_uuid4,
    mock_sanitize_input,
    mock_is_valid_voice_id,
    mock_get_broker # This captures the get_message_broker mock
):
    # Reset publish_message mock for this test
    mock_message_broker.publish_message.reset_mock()
    mock_publish_message_async.custom_side_effect = lambda p, rk: "msg_id_all_params_" + str(p.get("job_id"))

    mock_job_id = "test-job-id-123"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id) # mock uuid4().hex
    
    mock_is_valid_voice_id.return_value = True
    # sanitize_input will be called for topic, custom_prompt
    # For simplicity, assume it returns the value as is for this test.
    mock_sanitize_input.side_effect = lambda x, default_on_empty=False: x 

    payload = {
        "topic": "Test Topic",
        "urls": ["http://example.com/source1", "http://example.com/source2"],
        "voice_id": "valid_voice",
        "include_storyboard": True,
        "custom_prompt": "Make it extra engaging.",
        "output_format": "1080p",
        "temperature": 0.8,
        "chunk_size": 250,
        "job_id": mock_job_id # Provide job_id
    }

    response = client.post("/api/generate", json=payload)

    assert response.status_code == 202
    response_data = response.json()
    assert response_data["job_id"] == mock_job_id
    assert "Job accepted for processing" in response_data["message"]
    assert response_data["message_id"].startswith("msg_id_all_params_")

    mock_is_valid_voice_id.assert_called_once_with("valid_voice")
    # sanitize_input called for topic and custom_prompt
    mock_sanitize_input.assert_any_call("Test Topic", default_on_empty=False)
    mock_sanitize_input.assert_any_call("Make it extra engaging.", default_on_empty=True)
    
    mock_message_broker.publish_message.assert_called_once()
    published_payload = mock_message_broker.publish_message.call_args[0][0]
    
    assert published_payload["job_id"] == mock_job_id
    assert published_payload["topic"] == "Test Topic"
    assert published_payload["urls"] == ["http://example.com/source1", "http://example.com/source2"]
    assert published_payload["voice_id"] == "valid_voice"
    assert published_payload["include_storyboard"] is True
    assert published_payload["custom_prompt"] == "Make it extra engaging."
    assert published_payload["settings"]["output_format"] == "1080p"
    assert published_payload["settings"]["temperature"] == 0.8
    assert published_payload["settings"]["chunk_size"] == 250
    assert published_payload["settings"]["source_retention_days"] == app_config.processing.SOURCE_RETENTION_DAYS # Default
    assert "task_type" in published_payload # Should be added by the endpoint


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id') # Not used here, but good to have consistent mocks
@patch('viralStoryGenerator.utils.security.sanitize_input')
@patch('uuid.uuid4')
def test_generate_job_successful_minimal_params(
    mock_uuid4,
    mock_sanitize_input,
    mock_is_valid_voice_id, # Not strictly needed for this test path, but good for consistency
    mock_get_broker
):
    mock_message_broker.publish_message.reset_mock()
    generated_job_id = "generated-job-id-456"
    mock_uuid4.return_value = MagicMock(hex=generated_job_id)
    mock_publish_message_async.custom_side_effect = lambda p, rk: "msg_id_min_params_" + str(p.get("job_id"))


    mock_sanitize_input.side_effect = lambda x, default_on_empty=False: x # Return as is

    payload = {
        "topic": "Minimal Topic",
        "urls": ["http://example.com/minimal"]
    }

    response = client.post("/api/generate", json=payload)

    assert response.status_code == 202
    response_data = response.json()
    assert response_data["job_id"] == generated_job_id # UUID generated by endpoint
    assert "Job accepted for processing" in response_data["message"]
    assert response_data["message_id"].startswith("msg_id_min_params_")

    mock_sanitize_input.assert_called_once_with("Minimal Topic", default_on_empty=False)
    
    mock_message_broker.publish_message.assert_called_once()
    published_payload = mock_message_broker.publish_message.call_args[0][0]

    assert published_payload["job_id"] == generated_job_id
    assert published_payload["topic"] == "Minimal Topic"
    assert published_payload["urls"] == ["http://example.com/minimal"]
    assert published_payload["voice_id"] == app_config.elevenlabs.DEFAULT_VOICE_ID # Default
    assert published_payload["include_storyboard"] is False # Default
    assert published_payload["custom_prompt"] == "" # Default (empty after sanitize with default_on_empty=True)
    assert published_payload["settings"]["output_format"] == app_config.video.DEFAULT_OUTPUT_FORMAT # Default
    assert published_payload["settings"]["temperature"] == app_config.openai.DEFAULT_TEMPERATURE # Default
    assert published_payload["settings"]["chunk_size"] == app_config.processing.DEFAULT_CHUNK_SIZE # Default
    assert "task_type" in published_payload


# Invalid input tests for POST /api/generate
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.sanitize_input')
def test_generate_job_invalid_topic_empty(mock_sanitize_input, mock_get_broker):
    # This test assumes Pydantic model `GenerateRequest` has `topic: constr(min_length=1)`
    # or similar. If `topic: str`, this would not be a 422 at Pydantic level.
    # The problem description mentions "Assert 400 status code (or 422 ...)"
    # So, we test for 422 (Pydantic validation).
    
    # sanitize_input returning empty for a non-empty input is one way to trigger this,
    # if sanitization happens before Pydantic model creation.
    # However, the endpoint uses the Pydantic model directly.
    # So, an empty topic in the request itself should trigger this.
    mock_sanitize_input.return_value = "" # To simulate if sanitize_input was used before validation

    payload = {
        "topic": "", # Empty topic directly
        "urls": ["http://example.com/someurl"]
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422 # Pydantic validation error for empty topic


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.sanitize_input') # Mocked, but not key to this validation
def test_generate_job_invalid_urls_empty_list(mock_sanitize_input, mock_get_broker):
    # Assumes Pydantic model `GenerateRequest` has `urls: List[HttpUrl] = Field(..., min_items=1)`
    mock_sanitize_input.side_effect = lambda x, default_on_empty=False: x 
    payload = {
        "topic": "Valid Topic",
        "urls": [] # Empty URL list
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422 # Pydantic validation error for empty list


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.sanitize_input')
def test_generate_job_invalid_urls_not_http(mock_sanitize_input, mock_get_broker):
    # Assumes Pydantic model `GenerateRequest` has `urls: List[HttpUrl]`
    mock_sanitize_input.side_effect = lambda x, default_on_empty=False: x
    payload = {
        "topic": "Valid Topic",
        "urls": ["not-a-url", "http://validurl.com"] 
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422 # Pydantic validation error for invalid URL format


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id')
@patch('viralStoryGenerator.utils.security.sanitize_input')
def test_generate_job_invalid_voice_id(
    mock_sanitize_input,
    mock_is_valid_voice_id,
    mock_get_broker
):
    mock_sanitize_input.side_effect = lambda x, default_on_empty=False: x
    mock_is_valid_voice_id.return_value = False # Simulate invalid voice_id

    payload = {
        "topic": "Another Valid Topic",
        "urls": ["http://example.com/valid"],
        "voice_id": "invalid_voice_id_format"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 400
    assert "Invalid voice_id" in response.json()["detail"]
    mock_is_valid_voice_id.assert_called_once_with("invalid_voice_id_format")
    mock_message_broker.publish_message.assert_not_called() # Should not publish


# Test for publish_message failure
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id', return_value=True) # Assume valid voice
@patch('viralStoryGenerator.utils.security.sanitize_input', side_effect=lambda x, default_on_empty=False: x)
@patch('uuid.uuid4')
def test_generate_job_publish_message_fails(
    mock_uuid4,
    mock_sanitize_input,
    mock_is_valid_voice_id,
    mock_get_broker
):
    mock_message_broker.publish_message.reset_mock()
    # Configure publish_message to simulate failure (e.g., return None or raise specific error)
    mock_publish_message_async.custom_side_effect = lambda p, rk: None # Simulate failure to get message_id

    mock_job_id = "job-id-publish-fail"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id)

    payload = {
        "topic": "Publish Fail Topic",
        "urls": ["http://example.com/pfail"]
    }
    response = client.post("/api/generate", json=payload)

    assert response.status_code == 500
    assert "Failed to publish job to message broker" in response.json()["detail"]
    mock_message_broker.publish_message.assert_called_once() # Attempted to publish


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id', return_value=True)
@patch('viralStoryGenerator.utils.security.sanitize_input', side_effect=lambda x, default_on_empty=False: x)
@patch('uuid.uuid4')
def test_generate_job_publish_message_exception(
    mock_uuid4,
    mock_sanitize_input,
    mock_is_valid_voice_id,
    mock_get_broker
):
    mock_message_broker.publish_message.reset_mock()
    # Configure publish_message to raise an exception
    mock_publish_message_async.custom_side_effect = MagicMock(side_effect=Exception("Broker connection error"))

    mock_job_id = "job-id-publish-exception"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id)

    payload = {
        "topic": "Publish Exception Topic",
        "urls": ["http://example.com/pexception"]
    }
    response = client.post("/api/generate", json=payload)

    assert response.status_code == 500
    assert "Failed to publish job to message broker" in response.json()["detail"]
    # Ensure the detail of the original exception is not exposed unless intended
    assert "Broker connection error" not in response.json()["detail"] # Check if error is generic
    mock_message_broker.publish_message.assert_called_once()


# Authentication tests for POST /api/generate
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_generate")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker) # So it's not called
def test_generate_job_auth_no_api_key(mock_get_broker):
    mock_message_broker.publish_message.reset_mock()
    payload = {"topic": "Auth Test", "urls": ["http://example.com/auth"]}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_message_broker.publish_message.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_generate")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
def test_generate_job_auth_invalid_api_key(mock_get_broker):
    mock_message_broker.publish_message.reset_mock()
    payload = {"topic": "Auth Test Invalid", "urls": ["http://example.com/auth_invalid"]}
    headers = {"X-API-Key": "wrong_key"}
    response = client.post("/api/generate", json=payload, headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_message_broker.publish_message.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_generate")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id', return_value=True)
@patch('viralStoryGenerator.utils.security.sanitize_input', side_effect=lambda x, default_on_empty=False: x)
@patch('uuid.uuid4')
def test_generate_job_auth_valid_api_key(
    mock_uuid4, mock_sanitize, mock_valid_voice_id, mock_get_broker
):
    mock_message_broker.publish_message.reset_mock()
    mock_publish_message_async.custom_side_effect = lambda p, rk: "msg_id_auth_valid"
    
    mock_job_id = "job-id-auth-valid"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id)

    payload = {"topic": "Auth Test Valid", "urls": ["http://example.com/auth_valid"]}
    headers = {"X-API-Key": "test_api_key_generate"}
    response = client.post("/api/generate", json=payload, headers=headers)
    
    assert response.status_code == 202
    assert response.json()["job_id"] == mock_job_id
    mock_message_broker.publish_message.assert_called_once()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_voice_id', return_value=True)
@patch('viralStoryGenerator.utils.security.sanitize_input', side_effect=lambda x, default_on_empty=False: x)
@patch('uuid.uuid4')
def test_generate_job_auth_disabled_no_key(
    mock_uuid4, mock_sanitize, mock_valid_voice_id, mock_get_broker
):
    mock_message_broker.publish_message.reset_mock()
    mock_publish_message_async.custom_side_effect = lambda p, rk: "msg_id_auth_disabled"

    mock_job_id = "job-id-auth-disabled"
    mock_uuid4.return_value = MagicMock(hex=mock_job_id)

    payload = {"topic": "Auth Test Disabled", "urls": ["http://example.com/auth_disabled"]}
    response = client.post("/api/generate", json=payload)

    assert response.status_code == 202
    assert response.json()["job_id"] == mock_job_id
    mock_message_broker.publish_message.assert_called_once()


# Tests for GET /api/status/{job_id}

# New mock broker for /api/status endpoint
mock_status_message_broker = MagicMock()

async def mock_get_job_progress_async(job_id):
    if hasattr(mock_get_job_progress_async, 'custom_side_effect'):
        return mock_get_job_progress_async.custom_side_effect(job_id)
    # Default behavior if not customized
    if job_id == "known-job-in-redis":
        return {"status": "processing", "job_id": job_id, "message": "Job is processing in Redis."}
    return None

async def mock_track_job_progress_async(job_id, status_data, publish_to_live=False):
    if hasattr(mock_track_job_progress_async, 'custom_side_effect'):
        return mock_track_job_progress_async.custom_side_effect(job_id, status_data, publish_to_live)
    # Default behavior: can record calls or simulate success/failure
    mock_track_job_progress_async.calls.append((job_id, status_data, publish_to_live))
    return True # Simulate successful tracking

mock_status_message_broker.get_job_progress = MagicMock(side_effect=mock_get_job_progress_async)
mock_status_message_broker.track_job_progress = MagicMock(side_effect=mock_track_job_progress_async)
mock_track_job_progress_async.calls = [] # To store calls for assertion

# Mock for storage_manager instance if it's used as an object
# If storage_manager functions are called directly as module functions, patch those instead.
# Assuming `storage_manager` is an instance of a class available in `api.py` or `api_handlers.py`
# For this example, let's assume `api.py` has `from viralStoryGenerator.utils import storage_manager`
# and calls `storage_manager.retrieve_file_content_as_json` etc.

# Test Scenario 1: Invalid job_id format
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_job_status_invalid_job_id_format(mock_is_valid_uuid, mock_get_broker):
    job_id = "not-a-uuid"
    mock_is_valid_uuid.return_value = False

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 400
    assert "Invalid job_id format" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_not_called()


# Test Scenario 2: Job found in Redis (simple cases)
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
def test_get_job_status_found_in_redis_processing(mock_is_valid_uuid, mock_get_broker):
    job_id = "processing-job-in-redis"
    redis_data = {
        "status": "processing", "job_id": job_id, "message": "Currently processing video.",
        "progress": 50, "created_at": "2023-01-01T00:00:00Z", "updated_at": "2023-01-01T00:05:00Z"
    }
    mock_get_job_progress_async.custom_side_effect = lambda jid: redis_data if jid == job_id else None
    mock_status_message_broker.get_job_progress.reset_mock()


    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200
    assert response.json() == redis_data
    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)


@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
def test_get_job_status_found_in_redis_failed(mock_is_valid_uuid, mock_get_broker):
    job_id = "failed-job-in-redis"
    redis_data = {
        "status": "failed", "job_id": job_id, "error": "Video generation failed due to encoding error.",
        "created_at": "2023-01-02T00:00:00Z", "updated_at": "2023-01-02T00:10:00Z"
    }
    mock_get_job_progress_async.custom_side_effect = lambda jid: redis_data if jid == job_id else None
    mock_status_message_broker.get_job_progress.reset_mock()

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200
    assert response.json() == redis_data
    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)


# Test Scenario 3: Job found in Redis (completed, sufficiently detailed)
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.retrieve_file_content_as_json') # Should not be called
def test_get_job_status_redis_completed_detailed(mock_retrieve_json, mock_is_valid_uuid, mock_get_broker):
    job_id = "completed-detailed-redis"
    detailed_redis_data = {
        "job_id": job_id, "status": "completed",
        "created_at": "2023-03-01T10:00:00Z", "updated_at": "2023-03-01T10:30:00Z",
        "topic": "Test Story",
        "story_script": "This is the story script.", # Direct content
        "storyboard_elements": [{"type": "scene", "description": "Scene 1"}], # Direct content
        "final_video_url": f"{app_config.storage.S3_PUBLIC_URL_PREFIX}/{job_id}/final_video.mp4",
        "audio_url": f"{app_config.storage.S3_PUBLIC_URL_PREFIX}/{job_id}/audio.mp3",
        "metadata_url": f"{app_config.storage.S3_PUBLIC_URL_PREFIX}/{job_id}/metadata.json",
        "story_script_url": f"{app_config.storage.S3_PUBLIC_URL_PREFIX}/{job_id}/story.txt",
        "storyboard_url": f"{app_config.storage.S3_PUBLIC_URL_PREFIX}/{job_id}/storyboard.json",
        # Ensure all 'content' fields are present (story_script, storyboard_elements)
        # If these fields are present and non-empty, storage manager should not be called.
    }
    mock_get_job_progress_async.custom_side_effect = lambda jid: detailed_redis_data if jid == job_id else None
    mock_status_message_broker.get_job_progress.reset_mock()

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert response.json()["status"] == "completed"
    assert response.json()["story_script"] == "This is the story script."
    assert response.json()["storyboard_elements"] == [{"type": "scene", "description": "Scene 1"}]
    mock_retrieve_json.assert_not_called() # Crucial check
    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)


# Test Scenario 4: Job in Redis (completed, needs storage lookup)
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.retrieve_file_content_as_json')
@patch('viralStoryGenerator.src.api_handlers._fetch_story_content_from_storage_sync') # Mock the sync helper
@patch('viralStoryGenerator.src.api_handlers._fetch_storyboard_content_from_storage_sync') # Mock the sync helper
@patch('asyncio.to_thread') # Mock asyncio.to_thread
def test_get_job_status_redis_completed_needs_storage_lookup(
    mock_asyncio_to_thread,
    mock_fetch_storyboard_sync,
    mock_fetch_story_sync,
    mock_retrieve_metadata_json,
    mock_is_valid_uuid,
    mock_get_broker
):
    job_id = "completed-needs-lookup"
    # Minimal Redis data: status completed, but no direct content or detailed URLs
    redis_data = {
        "job_id": job_id, "status": "completed",
        "created_at": "2023-04-01T10:00:00Z", "updated_at": "2023-04-01T10:30:00Z",
        "topic": "Test Story from Redis", # Topic might be in Redis
        # Assume metadata_url is present, but story/storyboard content fields are missing/empty
        "metadata_url": f"s3://bucket/{job_id}/metadata.json", # URL might be in Redis
        "story_script_url": f"s3://bucket/{job_id}/story.txt", # URL might be in Redis
        "storyboard_url": f"s3://bucket/{job_id}/storyboard.json", # URL might be in Redis
        "final_video_url": f"s3://bucket/{job_id}/final_video.mp4", # URL might be in Redis
        "audio_url": f"s3://bucket/{job_id}/audio.mp3", # URL might be in Redis
        "story_script": None, # Explicitly None or missing
        "storyboard_elements": None, # Explicitly None or missing
    }
    mock_get_job_progress_async.custom_side_effect = lambda jid: redis_data if jid == job_id else None
    mock_status_message_broker.get_job_progress.reset_mock()
    mock_track_job_progress_async.calls = [] # Reset calls for track_job_progress

    # Mock storage responses
    mock_metadata_content = {
        "job_id": job_id, "topic": "Test Story from Metadata", "status": "completed", # from worker
        "story_script_url": f"s3://bucket/{job_id}/story.txt", # from worker
        "storyboard_url": f"s3://bucket/{job_id}/storyboard.json", # from worker
        "final_video_url": f"s3://bucket/{job_id}/final_video.mp4", # from worker
        "audio_url": f"s3://bucket/{job_id}/audio.mp3", # from worker
        "metadata_url": f"s3://bucket/{job_id}/metadata.json", # from worker
        "created_at": "2023-04-01T09:59:00Z", # Potentially different from Redis initial timestamp
        "updated_at": "2023-04-01T10:29:00Z",
    }
    mock_retrieve_metadata_json.return_value = mock_metadata_content

    mock_story_content = "This is the fetched story script."
    mock_storyboard_content = [{"scene": 1, "text": "Fetched storyboard scene."}]

    # Configure asyncio.to_thread to return the results of the sync functions
    # It will be called with the sync function and its args.
    # We need to check which function is being called.
    def to_thread_side_effect(func, *args, **kwargs):
        if func == mock_fetch_story_sync:
            return mock_story_content
        elif func == mock_fetch_storyboard_sync:
            return mock_storyboard_content
        raise ValueError(f"Unexpected function passed to asyncio.to_thread: {func}")
    mock_asyncio_to_thread.side_effect = to_thread_side_effect
    
    # Mock the sync helper functions directly if they are called by to_thread
    # This is not strictly necessary if to_thread itself is returning the content,
    # but good for clarity if the sync functions themselves have logic to test.
    # For this test, the side_effect on to_thread is sufficient.

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200
    response_data = response.json()

    assert response_data["job_id"] == job_id
    assert response_data["status"] == "completed"
    # Topic might come from Redis first, then potentially overridden by metadata if logic dictates
    assert response_data["topic"] == "Test Story from Metadata" # Assuming metadata is more authoritative
    assert response_data["story_script"] == mock_story_content
    assert response_data["storyboard_elements"] == mock_storyboard_content
    assert response_data["final_video_url"] == mock_metadata_content["final_video_url"] # From metadata
    
    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)
    
    # Check that storage_manager.retrieve_file_content_as_json was called for metadata.json
    # The path for metadata would be constructed using job_id.
    # Assuming the handler constructs a path like f"{job_id}/metadata.json"
    # or uses a method on storage_manager that takes job_id and file_key.
    # For now, assume it's called once for metadata.
    mock_retrieve_metadata_json.assert_called_once()
    
    # Check that asyncio.to_thread was called for story and storyboard
    mock_asyncio_to_thread.assert_any_call(mock_fetch_story_sync, mock_metadata_content["story_script_url"], job_id)
    mock_asyncio_to_thread.assert_any_call(mock_fetch_storyboard_sync, mock_metadata_content["storyboard_url"], job_id)

    # Verify that track_job_progress was called to cache the enriched data
    assert len(mock_track_job_progress_async.calls) == 1
    tracked_call_args = mock_track_job_progress_async.calls[0]
    assert tracked_call_args[0] == job_id # job_id
    tracked_data = tracked_call_args[1] # status_data
    assert tracked_data["status"] == "completed"
    assert tracked_data["story_script"] == mock_story_content
    assert tracked_data["storyboard_elements"] == mock_storyboard_content
    assert tracked_call_args[2] is False # publish_to_live should be False for this internal update


# Test Scenario 5: Job NOT in Redis, metadata in storage (completed)
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.retrieve_file_content_as_json')
@patch('viralStoryGenerator.src.api_handlers._fetch_story_content_from_storage_sync')
@patch('viralStoryGenerator.src.api_handlers._fetch_storyboard_content_from_storage_sync')
@patch('asyncio.to_thread')
def test_get_job_status_not_in_redis_metadata_found(
    mock_asyncio_to_thread,
    mock_fetch_storyboard_sync,
    mock_fetch_story_sync,
    mock_retrieve_metadata_json,
    mock_is_valid_uuid,
    mock_get_broker
):
    job_id = "not-in-redis-metadata-found"
    # Redis returns None
    mock_get_job_progress_async.custom_side_effect = lambda jid: None
    mock_status_message_broker.get_job_progress.reset_mock()
    mock_track_job_progress_async.calls = []

    # Mock storage responses
    mock_metadata_content = {
        "job_id": job_id, "topic": "Story from Storage Metadata", "status": "completed",
        "story_script_url": f"s3://bucket/{job_id}/story.txt",
        "storyboard_url": f"s3://bucket/{job_id}/storyboard.json",
        "final_video_url": f"s3://bucket/{job_id}/final_video.mp4",
        "audio_url": f"s3://bucket/{job_id}/audio.mp3",
        "metadata_url": f"s3://bucket/{job_id}/metadata.json",
        "created_at": "2023-05-01T10:00:00Z",
        "updated_at": "2023-05-01T10:30:00Z",
    }
    mock_retrieve_metadata_json.return_value = mock_metadata_content

    mock_story_content = "Fetched story from storage (no redis)."
    mock_storyboard_content = [{"scene": "S1", "desc": "Fetched storyboard from storage (no redis)."}]

    def to_thread_side_effect(func, *args, **kwargs):
        if func == mock_fetch_story_sync:
            return mock_story_content
        elif func == mock_fetch_storyboard_sync:
            return mock_storyboard_content
        return None
    mock_asyncio_to_thread.side_effect = to_thread_side_effect

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200
    response_data = response.json()

    assert response_data["job_id"] == job_id
    assert response_data["status"] == "completed"
    assert response_data["topic"] == "Story from Storage Metadata"
    assert response_data["story_script"] == mock_story_content
    assert response_data["storyboard_elements"] == mock_storyboard_content
    assert response_data["final_video_url"] == mock_metadata_content["final_video_url"]

    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)
    mock_retrieve_metadata_json.assert_called_once() # Called because Redis returned None
    
    mock_asyncio_to_thread.assert_any_call(mock_fetch_story_sync, mock_metadata_content["story_script_url"], job_id)
    mock_asyncio_to_thread.assert_any_call(mock_fetch_storyboard_sync, mock_metadata_content["storyboard_url"], job_id)

    # Verify caching to Redis
    assert len(mock_track_job_progress_async.calls) == 1
    tracked_call_args = mock_track_job_progress_async.calls[0]
    assert tracked_call_args[0] == job_id
    assert tracked_call_args[1]["status"] == "completed"
    assert tracked_call_args[1]["story_script"] == mock_story_content


# Test Scenario 6: Not in Redis, metadata found, but story/storyboard files missing
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.retrieve_file_content_as_json')
@patch('viralStoryGenerator.src.api_handlers._fetch_story_content_from_storage_sync')
@patch('viralStoryGenerator.src.api_handlers._fetch_storyboard_content_from_storage_sync')
@patch('asyncio.to_thread')
def test_get_job_status_not_in_redis_files_missing(
    mock_asyncio_to_thread,
    mock_fetch_storyboard_sync, # Mocked sync helper
    mock_fetch_story_sync,     # Mocked sync helper
    mock_retrieve_metadata_json,
    mock_is_valid_uuid,
    mock_get_broker
):
    job_id = "not-in-redis-files-missing"
    mock_get_job_progress_async.custom_side_effect = lambda jid: None # Not in Redis
    mock_status_message_broker.get_job_progress.reset_mock()
    mock_track_job_progress_async.calls = []

    mock_metadata_content = {
        "job_id": job_id, "topic": "Files Missing Story", "status": "completed",
        "story_script_url": f"s3://bucket/{job_id}/story.txt",
        "storyboard_url": f"s3://bucket/{job_id}/storyboard.json",
        "final_video_url": f"s3://bucket/{job_id}/final_video.mp4",
        "audio_url": f"s3://bucket/{job_id}/audio.mp3",
        "metadata_url": f"s3://bucket/{job_id}/metadata.json",
        "created_at": "2023-06-01T10:00:00Z", "updated_at": "2023-06-01T10:30:00Z",
    }
    mock_retrieve_metadata_json.return_value = mock_metadata_content

    # Simulate storage methods returning None (file not found)
    def to_thread_side_effect_files_missing(func, *args, **kwargs):
        if func == mock_fetch_story_sync:
            return None # Story file missing
        elif func == mock_fetch_storyboard_sync:
            return None # Storyboard file missing
        return None
    mock_asyncio_to_thread.side_effect = to_thread_side_effect_files_missing

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200 # Still 200 as metadata was found
    response_data = response.json()

    assert response_data["job_id"] == job_id
    assert response_data["status"] == "completed" # From metadata
    assert response_data["topic"] == "Files Missing Story"
    assert response_data["story_script"] is None # Should be None as file was missing
    assert response_data["storyboard_elements"] is None # Should be None
    assert response_data["final_video_url"] == mock_metadata_content["final_video_url"]

    mock_retrieve_metadata_json.assert_called_once()
    mock_asyncio_to_thread.assert_any_call(mock_fetch_story_sync, mock_metadata_content["story_script_url"], job_id)
    mock_asyncio_to_thread.assert_any_call(mock_fetch_storyboard_sync, mock_metadata_content["storyboard_url"], job_id)

    # Verify caching to Redis, even with missing content
    assert len(mock_track_job_progress_async.calls) == 1
    tracked_call_args = mock_track_job_progress_async.calls[0]
    assert tracked_call_args[0] == job_id
    assert tracked_call_args[1]["status"] == "completed"
    assert tracked_call_args[1]["story_script"] is None
    assert tracked_call_args[1]["storyboard_elements"] is None


# Test Scenario 7: Job not found anywhere (Redis and Storage miss)
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.retrieve_file_content_as_json')
def test_get_job_status_not_found_anywhere(
    mock_retrieve_metadata_json,
    mock_is_valid_uuid,
    mock_get_broker
):
    job_id = "job-not-found-anywhere"
    mock_get_job_progress_async.custom_side_effect = lambda jid: None # Not in Redis
    mock_status_message_broker.get_job_progress.reset_mock()
    
    # Simulate metadata not found in storage
    # Option 1: return None
    mock_retrieve_metadata_json.return_value = None
    # Option 2: raise FileNotFoundError (depends on storage_manager implementation)
    # mock_retrieve_metadata_json.side_effect = FileNotFoundError("Metadata not found in storage")

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 404
    assert "Job status not found" in response.json()["detail"] # Or similar message

    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)
    mock_retrieve_metadata_json.assert_called_once() # Attempted to retrieve metadata


# Test Scenario 8: Storage access error during fallback
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
@patch('viralStoryGenerator.utils.storage_manager.StorageManager.retrieve_file_content_as_json')
def test_get_job_status_storage_error_fallback(
    mock_retrieve_metadata_json,
    mock_is_valid_uuid,
    mock_get_broker
):
    job_id = "job-storage-error"
    mock_get_job_progress_async.custom_side_effect = lambda jid: None # Not in Redis
    mock_status_message_broker.get_job_progress.reset_mock()
    
    # Simulate storage_manager raising an unexpected error
    mock_retrieve_metadata_json.side_effect = Exception("S3 connection timeout")

    response = client.get(f"/api/status/{job_id}")

    # Depending on error handling, this could be 500 or 404 if the error is caught
    # and treated as "not found". Given it's an unexpected S3 error, 500 is more likely.
    assert response.status_code == 500 
    assert "Error retrieving job status from storage" in response.json()["detail"]
    # Check if the original error detail is hidden for security
    assert "S3 connection timeout" not in response.json()["detail"]


    mock_is_valid_uuid.assert_called_once_with(job_id)
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)
    mock_retrieve_metadata_json.assert_called_once()


# Test Scenario 9: Authentication
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_status")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_job_status_auth_no_api_key(mock_get_broker):
    job_id = "some-job-id-no-auth"
    mock_status_message_broker.get_job_progress.reset_mock() # Ensure not called
    
    response = client.get(f"/api/status/{job_id}")
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_status_message_broker.get_job_progress.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_status")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_job_status_auth_invalid_api_key(mock_get_broker):
    job_id = "some-job-id-invalid-auth"
    mock_status_message_broker.get_job_progress.reset_mock()
    
    headers = {"X-API-Key": "wrong_key_for_status"}
    response = client.get(f"/api/status/{job_id}", headers=headers)
    
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_status_message_broker.get_job_progress.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_status")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True) # Assume valid UUID after auth
def test_get_job_status_auth_valid_api_key(mock_is_valid_uuid, mock_get_broker):
    job_id = "job-id-valid-auth"
    # Simulate a basic Redis response after successful auth
    redis_data = {"status": "pending", "job_id": job_id}
    mock_get_job_progress_async.custom_side_effect = lambda jid: redis_data if jid == job_id else None
    mock_status_message_broker.get_job_progress.reset_mock()

    headers = {"X-API-Key": "test_api_key_status"}
    response = client.get(f"/api/status/{job_id}", headers=headers)

    assert response.status_code == 200
    assert response.json() == redis_data
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid', return_value=True)
def test_get_job_status_auth_disabled_no_key(mock_is_valid_uuid, mock_get_broker):
    job_id = "job-id-auth-disabled"
    redis_data = {"status": "queued", "job_id": job_id}
    mock_get_job_progress_async.custom_side_effect = lambda jid: redis_data if jid == job_id else None
    mock_status_message_broker.get_job_progress.reset_mock()

    response = client.get(f"/api/status/{job_id}")

    assert response.status_code == 200
    assert response.json() == redis_data
    mock_status_message_broker.get_job_progress.assert_called_once_with(job_id)


# Tests for GET /api/queue/status

# Augment mock_status_message_broker with get_queue_information
async def mock_get_queue_information_async():
    if hasattr(mock_get_queue_information_async, 'custom_side_effect'):
        return mock_get_queue_information_async.custom_side_effect()
    # Default success response structure (can be customized per test)
    return {
        "status": "ok",
        "error_message": None,
        "stream_name": app_config.redis.STREAM_NAME,
        "stream_length": 100,
        "pending_messages_count": 5,
        "consumer_groups": [
            {
                "name": "group1",
                "consumers": [
                    {"name": "consumer1-1", "pending": 2, "inactive_ms": 12345},
                    {"name": "consumer1-2", "pending": 0, "inactive_ms": 500}
                ],
                "pending_messages": 2
            }
        ],
        "dead_letter_queue_length": 0,
        "recent_messages": [
            {"id": "1700000000000-0", "payload": {"job_id": "job1", "topic": "Topic 1"}, "delivered_count": 1, "last_delivered_ms": 1700000001000},
            {"id": "1700000002000-0", "payload": {"job_id": "job2", "topic": "Topic 2"}, "delivered_count": 2, "last_delivered_ms": 1700000003000}
        ]
    }

mock_status_message_broker.get_queue_information = MagicMock(side_effect=mock_get_queue_information_async)


# Scenario 1: Successful queue status retrieval
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_successful(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    # Use default successful mock_get_queue_information_async behavior
    mock_get_queue_information_async.custom_side_effect = None # Ensure default behavior

    response = client.get("/api/queue/status")

    assert response.status_code == 200
    response_data = response.json()

    assert response_data["status"] == "ok"
    assert response_data["error_message"] is None
    assert response_data["stream_name"] == app_config.redis.STREAM_NAME
    assert response_data["stream_length"] == 100
    assert response_data["pending_messages_count"] == 5
    assert len(response_data["consumer_groups"]) == 1
    cg = response_data["consumer_groups"][0]
    assert cg["name"] == "group1"
    assert len(cg["consumers"]) == 2
    assert cg["consumers"][0]["name"] == "consumer1-1"
    assert cg["consumers"][0]["pending_messages"] == 2 # Schema uses 'pending_messages'
    assert cg["consumers"][0]["inactive_time_ms"] == 12345 # Schema uses 'inactive_time_ms'
    assert len(response_data["recent_messages"]) == 2
    rm = response_data["recent_messages"][0]
    assert rm["message_id"] == "1700000000000-0" # Schema uses 'message_id'
    assert rm["data"]["job_id"] == "job1" # Schema uses 'data'
    
    mock_status_message_broker.get_queue_information.assert_called_once()


# Scenario 2: get_queue_information reports an error
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_broker_reports_error(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    error_response_from_broker = {
        "status": "error",
        "error_message": "Redis connection failed",
        "stream_name": app_config.redis.STREAM_NAME,
        "stream_length": 0, # Default/error values
        "pending_messages_count": 0,
        "consumer_groups": [],
        "dead_letter_queue_length": -1, # Example error indicator
        "recent_messages": []
    }
    mock_get_queue_information_async.custom_side_effect = lambda: error_response_from_broker

    response = client.get("/api/queue/status")

    assert response.status_code == 200 # Endpoint handles it gracefully
    response_data = response.json()
    
    assert response_data["status"] == "error"
    assert response_data["error_message"] == "Redis connection failed"
    assert response_data["stream_length"] == 0
    assert len(response_data["consumer_groups"]) == 0
    mock_status_message_broker.get_queue_information.assert_called_once()


# Scenario 3: get_queue_information raises an exception
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_broker_raises_exception(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    mock_get_queue_information_async.custom_side_effect = MagicMock(side_effect=Exception("Unexpected broker error"))

    response = client.get("/api/queue/status")

    assert response.status_code == 500
    response_data = response.json()
    assert "Internal server error" in response_data["detail"]
    assert "Unexpected broker error" not in response_data["detail"] # Hide internal error details
    mock_status_message_broker.get_queue_information.assert_called_once()


# Scenario 4: Authentication
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_queuestatus")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_auth_no_api_key(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    response = client.get("/api/queue/status")
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_status_message_broker.get_queue_information.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_queuestatus")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_auth_invalid_api_key(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    headers = {"X-API-Key": "wrong_key_for_queue_status"}
    response = client.get("/api/queue/status", headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_status_message_broker.get_queue_information.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_queuestatus")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_auth_valid_api_key(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    # Use default successful mock_get_queue_information_async behavior
    mock_get_queue_information_async.custom_side_effect = None 

    headers = {"X-API-Key": "test_api_key_queuestatus"}
    response = client.get("/api/queue/status", headers=headers)
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok" # Check if successful call was made
    mock_status_message_broker.get_queue_information.assert_called_once()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_status_message_broker)
def test_get_queue_status_auth_disabled_no_key(mock_get_broker):
    mock_status_message_broker.get_queue_information.reset_mock()
    mock_get_queue_information_async.custom_side_effect = None

    response = client.get("/api/queue/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_status_message_broker.get_queue_information.assert_called_once()


# Tests for POST /api/queue/clear-stalled

# New mock broker for /api/queue/clear-stalled endpoint
mock_clear_stalled_broker = MagicMock()

async def mock_cs_initialize_async():
    if hasattr(mock_cs_initialize_async, 'custom_side_effect'):
        return mock_cs_initialize_async.custom_side_effect()
    return None # Default successful init

async def mock_cs_get_queue_info_async():
    if hasattr(mock_cs_get_queue_info_async, 'custom_side_effect'):
        return mock_cs_get_queue_info_async.custom_side_effect()
    # Default successful response
    return [ # Corresponds to XINFO GROUPS output structure
        {"name": "group1", "consumers": 2, "pending": 0, "last-delivered-id": "id1"},
        {"name": "group2", "consumers": 1, "pending": 0, "last-delivered-id": "id2"},
    ]

async def mock_cs_get_pending_messages_info_async(group_name, consumer_name, min_idle_time_ms, count):
    if hasattr(mock_cs_get_pending_messages_info_async, 'custom_side_effect'):
        return mock_cs_get_pending_messages_info_async.custom_side_effect(group_name, consumer_name, min_idle_time_ms, count)
    # Default: no pending messages for any group/consumer
    return [] # Corresponds to XPENDING <stream> <group> - IDLE <min_idle_time> <start_id> <end_id> <count> [<consumer_name>]

mock_clear_stalled_broker.initialize = MagicMock(side_effect=mock_cs_initialize_async)
mock_clear_stalled_broker.get_queue_information = MagicMock(side_effect=mock_cs_get_queue_info_async) # This should be get_consumer_groups_info based on endpoint
mock_clear_stalled_broker.get_consumer_groups_info = MagicMock(side_effect=mock_cs_get_queue_info_async) # Correcting based on typical Redis broker interface for XINFO GROUPS
mock_clear_stalled_broker.get_pending_messages_info = MagicMock(side_effect=mock_cs_get_pending_messages_info_async)


# Scenario 1: Successful call, no stalled jobs found
@patch('viralStoryGenerator.src.api._logger') # Patch the logger in api.py
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_no_stalled_found(mock_get_broker, mock_api_logger):
    # Reset mocks
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True) # Reset side_effect too
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_pending_messages_info.reset_mock(side_effect=True)
    mock_api_logger.info.reset_mock()

    # Configure default successful behavior for this test
    mock_cs_initialize_async.custom_side_effect = None
    mock_cs_get_queue_info_async.custom_side_effect = lambda: [
        {"name": "group1", "consumers": 1, "pending": 0, "last-delivered-id": "id1"},
    ]
    # This means get_pending_messages_info will return [] (its default) for group1/any-consumer
    mock_cs_get_pending_messages_info_async.custom_side_effect = None 


    response = client.post("/api/queue/clear-stalled")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Stalled job check completed."
    assert response_data["claimed_jobs"] == 0
    assert response_data["failed_to_claim_jobs"] == 0
    assert response_data["reprocessed_jobs"] == 0
    assert response_data["failed_to_reprocess_jobs"] == 0
    
    mock_clear_stalled_broker.initialize.assert_called_once()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_called_once()
    # get_pending_messages_info would be called for each consumer in each group.
    # If a group has 0 pending overall, the inner loop for consumers might be skipped by endpoint logic.
    # Based on current api.py, it iterates groups, then for each group, it tries to get pending for *each consumer*.
    # The provided `clear_stalled_jobs` endpoint code does not iterate consumers within a group for XPENDING.
    # It seems to iterate groups and then calls get_pending_messages_info for the group itself, not per consumer.
    # Let's adjust the mock setup for get_pending_messages_info if needed.
    # The endpoint code: `pending_info = await broker.get_pending_messages_info(group_info["name"], min_idle_time_ms=STALLED_JOB_THRESHOLD_MS, count=10)`
    # So it's called per group, not per consumer.
    mock_clear_stalled_broker.get_pending_messages_info.assert_called_once_with(
        group_name="group1", 
        min_idle_time_ms=app_config.processing.STALLED_JOB_THRESHOLD_MS, 
        count=10 # Default count in endpoint
    )
    # Verify logger calls (no stalled jobs found)
    mock_api_logger.info.assert_any_call("Starting stalled job check...")
    mock_api_logger.info.assert_any_call(f"Checking group: group1, Stalled threshold: {app_config.processing.STALLED_JOB_THRESHOLD_MS}ms")
    mock_api_logger.info.assert_any_call("No stalled jobs found in group group1.")
    mock_api_logger.info.assert_any_call("Stalled job check finished. Summary: {'claimed': 0, 'failed_to_claim': 0, 'reprocessed': 0, 'failed_to_reprocess': 0}")


# Scenario 2: Successful call, stalled jobs found (logging only)
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_found_logging_only(mock_get_broker, mock_api_logger):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_pending_messages_info.reset_mock(side_effect=True)
    mock_api_logger.info.reset_mock()
    mock_api_logger.warning.reset_mock() # For "TODO" log messages

    # Configure mocks
    mock_cs_initialize_async.custom_side_effect = None
    mock_cs_get_queue_info_async.custom_side_effect = lambda: [ # For get_consumer_groups_info
        {"name": "groupA", "consumers": 2, "pending": 5, "last-delivered-id": "idA"},
        {"name": "groupB", "consumers": 1, "pending": 0, "last-delivered-id": "idB"}, # No pending here
    ]
    
    # Simulate pending messages for groupA, none for groupB
    # The actual structure of pending_messages_info output from broker method:
    # [{'message_id': '123-0', 'consumer': 'consumer-1', 'idle_time_ms': 70000, 'delivery_count': 1}, ...]
    stalled_messages_groupA = [
        {'message_id': 'msg1', 'consumer': 'consumerA1', 'idle_time_ms': app_config.processing.STALLED_JOB_THRESHOLD_MS + 1000, 'delivery_count': 1},
        {'message_id': 'msg2', 'consumer': 'consumerA2', 'idle_time_ms': app_config.processing.STALLED_JOB_THRESHOLD_MS + 2000, 'delivery_count': 2},
    ]

    def get_pending_side_effect(group_name, consumer_name, min_idle_time_ms, count): # consumer_name arg is for XPENDING per consumer, endpoint uses XPENDING per group
        if group_name == "groupA":
            # Filter by min_idle_time_ms if the mock needs to be that precise,
            # or just return the list assuming the endpoint's min_idle_time_ms is used.
            # The endpoint calls get_pending_messages_info with group_name, min_idle_time_ms, count.
            # So, the mock should expect these.
            return [msg for msg in stalled_messages_groupA if msg['idle_time_ms'] >= min_idle_time_ms]
        elif group_name == "groupB":
            return [] # No pending for groupB
        return []
        
    # The endpoint calls: await broker.get_pending_messages_info(group_info["name"], min_idle_time_ms=STALLED_JOB_THRESHOLD_MS, count=10)
    # So, the mock needs to accept group_name, min_idle_time_ms, count. The consumer_name arg in my mock was an error for this specific call.
    async def mock_cs_get_pending_messages_info_per_group_async(group_name, min_idle_time_ms, count):
         if hasattr(mock_cs_get_pending_messages_info_per_group_async, 'custom_side_effect'):
            return await mock_cs_get_pending_messages_info_per_group_async.custom_side_effect(group_name, min_idle_time_ms, count)
         # Default behavior
         if group_name == "groupA":
            return [msg for msg in stalled_messages_groupA if msg['idle_time_ms'] >= min_idle_time_ms]
         return []

    mock_clear_stalled_broker.get_pending_messages_info.side_effect = mock_cs_get_pending_messages_info_per_group_async
    mock_cs_get_pending_messages_info_per_group_async.custom_side_effect = None # Use the default logic above for this test


    response = client.post("/api/queue/clear-stalled")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Stalled job check completed."
    assert response_data["claimed_jobs"] == 0 # Logging only
    assert response_data["failed_to_claim_jobs"] == 0
    assert response_data["reprocessed_jobs"] == 0
    assert response_data["failed_to_reprocess_jobs"] == 0

    mock_clear_stalled_broker.initialize.assert_called_once()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_called_once()
    
    # get_pending_messages_info called for each group
    mock_clear_stalled_broker.get_pending_messages_info.assert_any_call(
        group_name="groupA", min_idle_time_ms=app_config.processing.STALLED_JOB_THRESHOLD_MS, count=10
    )
    mock_clear_stalled_broker.get_pending_messages_info.assert_any_call(
        group_name="groupB", min_idle_time_ms=app_config.processing.STALLED_JOB_THRESHOLD_MS, count=10
    )
    
    # Verify logger calls
    mock_api_logger.info.assert_any_call("Starting stalled job check...")
    mock_api_logger.info.assert_any_call(f"Checking group: groupA, Stalled threshold: {app_config.processing.STALLED_JOB_THRESHOLD_MS}ms")
    mock_api_logger.info.assert_any_call(f"Found 2 stalled jobs in group groupA: {stalled_messages_groupA}")
    mock_api_logger.warning.assert_any_call("TODO: Implement actual claiming and reprocessing for groupA if needed.") # Check for the TODO message
    
    mock_api_logger.info.assert_any_call(f"Checking group: groupB, Stalled threshold: {app_config.processing.STALLED_JOB_THRESHOLD_MS}ms")
    mock_api_logger.info.assert_any_call("No stalled jobs found in group groupB.")
    mock_api_logger.info.assert_any_call("Stalled job check finished. Summary: {'claimed': 0, 'failed_to_claim': 0, 'reprocessed': 0, 'failed_to_reprocess': 0}")


# Scenario 3: Error during get_consumer_groups_info
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_error_get_consumer_groups(mock_get_broker, mock_api_logger):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_pending_messages_info.reset_mock(side_effect=True) # Should not be called
    mock_api_logger.error.reset_mock()

    mock_cs_initialize_async.custom_side_effect = None # Initialize succeeds
    # get_consumer_groups_info raises an exception
    mock_cs_get_queue_info_async.custom_side_effect = MagicMock(side_effect=Exception("Redis XINFO GROUPS error")) 

    response = client.post("/api/queue/clear-stalled")

    assert response.status_code == 500
    response_data = response.json()
    assert "Internal server error while processing queue" in response_data["detail"]
    
    mock_clear_stalled_broker.initialize.assert_called_once()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_called_once()
    mock_clear_stalled_broker.get_pending_messages_info.assert_not_called()
    
    mock_api_logger.error.assert_called_once()
    # Check that the log message contains the exception type or message if possible/desired
    # For example:
    args, _ = mock_api_logger.error.call_args
    assert "Error during stalled job check: Redis XINFO GROUPS error" in args[0]


# Scenario 4: Error during get_pending_messages_info
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_error_get_pending_info(mock_get_broker, mock_api_logger):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_pending_messages_info.reset_mock(side_effect=True)
    mock_api_logger.error.reset_mock()

    mock_cs_initialize_async.custom_side_effect = None # Initialize succeeds
    # get_consumer_groups_info succeeds, returns one group
    mock_cs_get_queue_info_async.custom_side_effect = lambda: [
        {"name": "groupX", "consumers": 1, "pending": 1, "last-delivered-id": "idX"},
    ]
    # get_pending_messages_info raises an exception for groupX
    async def mock_pending_info_exception_side_effect(group_name, min_idle_time_ms, count):
        if group_name == "groupX":
            raise Exception("Redis XPENDING error")
        return []
    
    # Use the more specific mock for get_pending_messages_info if it was defined as such
    # (as in test_clear_stalled_jobs_found_logging_only)
    if hasattr(mock_clear_stalled_broker.get_pending_messages_info, 'side_effect') and \
       hasattr(mock_clear_stalled_broker.get_pending_messages_info.side_effect, 'custom_side_effect'):
        mock_clear_stalled_broker.get_pending_messages_info.side_effect.custom_side_effect = mock_pending_info_exception_side_effect
    else: # Fallback to directly setting side_effect on the main mock method
        mock_clear_stalled_broker.get_pending_messages_info.side_effect = mock_pending_info_exception_side_effect


    response = client.post("/api/queue/clear-stalled")

    assert response.status_code == 500
    response_data = response.json()
    assert "Internal server error while processing queue" in response_data["detail"]

    mock_clear_stalled_broker.initialize.assert_called_once()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_called_once()
    mock_clear_stalled_broker.get_pending_messages_info.assert_called_once_with(
        group_name="groupX", min_idle_time_ms=app_config.processing.STALLED_JOB_THRESHOLD_MS, count=10
    )
    
    mock_api_logger.error.assert_called_once()
    args, _ = mock_api_logger.error.call_args
    assert "Error during stalled job check: Redis XPENDING error" in args[0]


# Scenario 5: Authentication
@patch('viralStoryGenerator.src.api._logger') # To prevent its output during auth tests unless specifically testing logs
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_clearstalled")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_auth_no_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True) # Ensure not called
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    
    response = client.post("/api/queue/clear-stalled")
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_clear_stalled_broker.initialize.assert_not_called()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_clearstalled")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_auth_invalid_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)

    headers = {"X-API-Key": "wrong_key_for_clearstalled"}
    response = client.post("/api/queue/clear-stalled", headers=headers)
    
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_clear_stalled_broker.initialize.assert_not_called()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_clearstalled")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_auth_valid_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_pending_messages_info.reset_mock(side_effect=True) # Ensure it's also reset

    # Configure default successful behavior for broker calls after auth
    mock_cs_initialize_async.custom_side_effect = None
    mock_cs_get_queue_info_async.custom_side_effect = lambda: [] # No groups, simplest success case
    mock_cs_get_pending_messages_info_async.custom_side_effect = None # No pending messages

    headers = {"X-API-Key": "test_api_key_clearstalled"}
    response = client.post("/api/queue/clear-stalled", headers=headers)
    
    assert response.status_code == 200 # Successful call after auth
    assert response.json()["claimed_jobs"] == 0 # From default mock behavior
    
    mock_clear_stalled_broker.initialize.assert_called_once()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_called_once()
    # get_pending_messages_info might not be called if get_consumer_groups_info returns empty list
    # Adjust assertion based on actual logic for empty groups.
    # If groups is empty, the loop for get_pending_messages_info is not entered.
    mock_clear_stalled_broker.get_pending_messages_info.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_clear_stalled_broker)
def test_clear_stalled_jobs_auth_disabled_no_key(mock_get_broker, mock_api_logger_ignored):
    mock_clear_stalled_broker.initialize.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_consumer_groups_info.reset_mock(side_effect=True)
    mock_clear_stalled_broker.get_pending_messages_info.reset_mock(side_effect=True)

    mock_cs_initialize_async.custom_side_effect = None
    mock_cs_get_queue_info_async.custom_side_effect = lambda: [] # No groups
    mock_cs_get_pending_messages_info_async.custom_side_effect = None

    response = client.post("/api/queue/clear-stalled")

    assert response.status_code == 200
    assert response.json()["claimed_jobs"] == 0
    
    mock_clear_stalled_broker.initialize.assert_called_once()
    mock_clear_stalled_broker.get_consumer_groups_info.assert_called_once()
    mock_clear_stalled_broker.get_pending_messages_info.assert_not_called() # Due to no groups


# Tests for DELETE /api/queue/purge

# New mock broker for /api/queue/purge endpoint
mock_purge_broker = MagicMock()

async def mock_purge_initialize_async():
    if hasattr(mock_purge_initialize_async, 'custom_side_effect'):
        return mock_purge_initialize_async.custom_side_effect()
    return None # Default successful init

async def mock_purge_stream_messages_async():
    if hasattr(mock_purge_stream_messages_async, 'custom_side_effect'):
        return mock_purge_stream_messages_async.custom_side_effect()
    return 10 # Default: 10 messages purged

mock_purge_broker.initialize = MagicMock(side_effect=mock_purge_initialize_async)
mock_purge_broker.purge_stream_messages = MagicMock(side_effect=mock_purge_stream_messages_async)

CONFIRMATION_CODE = "CONFIRM_PURGE_ALL_JOBS_SERIOUSLY"

# Scenario 1: Successful purge with correct confirmation
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_successful(mock_get_broker, mock_api_logger):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)
    mock_api_logger.critical.reset_mock()
    mock_api_logger.info.reset_mock() # For other log levels if needed

    mock_purge_initialize_async.custom_side_effect = None # Success
    mock_purge_stream_messages_async.custom_side_effect = lambda: 15 # Purged 15 messages

    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}")

    assert response.status_code == 200
    response_data = response.json()
    assert "Queue purged successfully." in response_data["message"]
    assert response_data["purged_messages_count"] == 15

    mock_purge_broker.initialize.assert_called_once()
    mock_purge_broker.purge_stream_messages.assert_called_once()
    
    mock_api_logger.critical.assert_any_call(f"Purge queue requested with confirmation code: {CONFIRMATION_CODE}")
    mock_api_logger.critical.assert_any_call("Queue purge successful. Messages purged: 15")


# Scenario 2: Purge attempt with incorrect confirmation
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_incorrect_confirmation(mock_get_broker, mock_api_logger):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)
    mock_api_logger.warning.reset_mock()

    wrong_confirmation = "WRONG_CODE_LOL"
    response = client.delete(f"/api/queue/purge?confirmation={wrong_confirmation}")

    assert response.status_code == 400
    response_data = response.json()
    assert "Invalid confirmation code" in response_data["detail"]

    mock_purge_broker.initialize.assert_not_called()
    mock_purge_broker.purge_stream_messages.assert_not_called()
    mock_api_logger.warning.assert_called_once_with(
        f"Queue purge attempt failed due to incorrect confirmation code: {wrong_confirmation}"
    )

# Scenario 3: Purge attempt with missing confirmation
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_missing_confirmation(mock_get_broker, mock_api_logger):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)
    
    response = client.delete("/api/queue/purge") # No confirmation param

    assert response.status_code == 422 # FastAPI validation error for missing query param
    # Detail structure for missing query param can be checked if needed
    # e.g., response.json()["detail"][0]["msg"] == "field required"

    mock_purge_broker.initialize.assert_not_called()
    mock_purge_broker.purge_stream_messages.assert_not_called()
    # No specific log for this from the endpoint code, FastAPI handles it.

# Scenario 4: Error during initialize
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_error_initialize(mock_get_broker, mock_api_logger):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True) # Should not be called
    mock_api_logger.error.reset_mock()

    mock_purge_initialize_async.custom_side_effect = Exception("Redis init failed")

    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}")

    assert response.status_code == 500
    response_data = response.json()
    assert "Internal server error during queue purge" in response_data["detail"]
    
    mock_purge_broker.initialize.assert_called_once()
    mock_purge_broker.purge_stream_messages.assert_not_called()
    mock_api_logger.error.assert_called_once()
    args, _ = mock_api_logger.error.call_args
    assert "Error during queue purge (initialize phase): Redis init failed" in args[0]


# Scenario 5: Error during purge_stream_messages
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_error_purge_messages(mock_get_broker, mock_api_logger):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)
    mock_api_logger.error.reset_mock()

    mock_purge_initialize_async.custom_side_effect = None # Initialize succeeds
    mock_purge_stream_messages_async.custom_side_effect = Exception("XTRIM/XDEL failed")

    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}")

    assert response.status_code == 500
    response_data = response.json()
    assert "Internal server error during queue purge" in response_data["detail"]

    mock_purge_broker.initialize.assert_called_once()
    mock_purge_broker.purge_stream_messages.assert_called_once()
    mock_api_logger.error.assert_called_once()
    args, _ = mock_api_logger.error.call_args
    assert "Error during queue purge (purge_stream_messages phase): XTRIM/XDEL failed" in args[0]


# Scenario 6: Authentication
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_purge")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_auth_no_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)
    
    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}")
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_purge_broker.initialize.assert_not_called()
    mock_purge_broker.purge_stream_messages.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_purge")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_auth_invalid_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)

    headers = {"X-API-Key": "wrong_key_for_purge"}
    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}", headers=headers)
    
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_purge_broker.initialize.assert_not_called()
    mock_purge_broker.purge_stream_messages.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_purge")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_auth_valid_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)

    mock_purge_initialize_async.custom_side_effect = None
    mock_purge_stream_messages_async.custom_side_effect = lambda: 5 # Purged 5

    headers = {"X-API-Key": "test_api_key_purge"}
    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}", headers=headers)
    
    assert response.status_code == 200
    assert response.json()["purged_messages_count"] == 5
    
    mock_purge_broker.initialize.assert_called_once()
    mock_purge_broker.purge_stream_messages.assert_called_once()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_purge_broker)
def test_purge_queue_auth_disabled_no_key(mock_get_broker, mock_api_logger_ignored):
    mock_purge_broker.initialize.reset_mock(side_effect=True)
    mock_purge_broker.purge_stream_messages.reset_mock(side_effect=True)

    mock_purge_initialize_async.custom_side_effect = None
    mock_purge_stream_messages_async.custom_side_effect = lambda: 3 # Purged 3

    response = client.delete(f"/api/queue/purge?confirmation={CONFIRMATION_CODE}")

    assert response.status_code == 200
    assert response.json()["purged_messages_count"] == 3
    
    mock_purge_broker.initialize.assert_called_once()
    mock_purge_broker.purge_stream_messages.assert_called_once()


# Tests for POST /api/queue/job/{job_id}/retry

mock_retry_broker = MagicMock()

async def mock_retry_initialize_async():
    # print("mock_retry_initialize_async called")
    if hasattr(mock_retry_initialize_async, 'custom_side_effect'):
        return mock_retry_initialize_async.custom_side_effect()
    return None

async def mock_retry_get_job_progress_async(job_id):
    # print(f"mock_retry_get_job_progress_async called with {job_id=}")
    if hasattr(mock_retry_get_job_progress_async, 'custom_side_effect'):
        return mock_retry_get_job_progress_async.custom_side_effect(job_id)
    # Default: job not found
    return None

async def mock_retry_publish_message_async(payload, routing_key):
    # print(f"mock_retry_publish_message_async called with {payload=}, {routing_key=}")
    if hasattr(mock_retry_publish_message_async, 'custom_side_effect'):
        return mock_retry_publish_message_async.custom_side_effect(payload, routing_key)
    return f"new_message_id_for_{payload.get('job_id', 'unknown_job')}"

async def mock_retry_track_job_progress_async(job_id, status_data, publish_to_live=False):
    # print(f"mock_retry_track_job_progress_async called with {job_id=}, {status_data=}, {publish_to_live=}")
    if hasattr(mock_retry_track_job_progress_async, 'custom_side_effect'):
        return mock_retry_track_job_progress_async.custom_side_effect(job_id, status_data, publish_to_live)
    mock_retry_track_job_progress_async.calls.append((job_id, status_data, publish_to_live))
    return True

mock_retry_broker.initialize = MagicMock(side_effect=mock_retry_initialize_async)
mock_retry_broker.get_job_progress = MagicMock(side_effect=mock_retry_get_job_progress_async)
mock_retry_broker.publish_message = MagicMock(side_effect=mock_retry_publish_message_async)
mock_retry_broker.track_job_progress = MagicMock(side_effect=mock_retry_track_job_progress_async)
mock_retry_track_job_progress_async.calls = []


# Scenario 1: Successful job retry
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('uuid.uuid4')
def test_retry_job_successful(
    mock_uuid4,
    mock_is_valid_uuid,
    mock_get_broker,
    mock_api_logger
):
    # Reset mocks and their side effects for this test
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True); mock_retry_track_job_progress_async.calls = []
    mock_api_logger.info.reset_mock(); mock_api_logger.error.reset_mock()

    original_job_id = "original-failed-job-123"
    new_job_id = "new-retried-job-456"
    new_message_id = f"new_msg_id_for_{new_job_id}"

    mock_is_valid_uuid.return_value = True
    
    original_job_data = {
        "job_id": original_job_id,
        "status": "failed", # Crucial for retry eligibility
        "topic": "Original Topic",
        "urls": ["http://example.com/original_source"],
        "voice_id": "original_voice",
        "include_storyboard": False,
        "custom_prompt": "Original prompt",
        "settings": {
            "output_format": "720p", "temperature": 0.7, "chunk_size": 200, 
            "source_retention_days": 30
        },
        "task_type": "generate_story_and_video" # Original task type
    }
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: original_job_data if jid == original_job_id else None
    
    # Mock uuid.uuid4().hex to return the new job ID
    mock_uuid4.return_value = MagicMock(hex=new_job_id)
    
    mock_retry_publish_message_async.custom_side_effect = lambda p, rk: new_message_id
    mock_retry_track_job_progress_async.custom_side_effect = None # Use default successful tracking


    response = client.post(f"/api/queue/job/{original_job_id}/retry")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Job successfully retried."
    assert response_data["original_job_id"] == original_job_id
    assert response_data["new_job_id"] == new_job_id
    assert response_data["new_message_id"] == new_message_id

    mock_is_valid_uuid.assert_called_once_with(original_job_id)
    mock_retry_broker.initialize.assert_called_once()
    mock_retry_broker.get_job_progress.assert_called_once_with(original_job_id)
    
    # Verify publish_message call
    mock_retry_broker.publish_message.assert_called_once()
    published_payload = mock_retry_broker.publish_message.call_args[0][0]
    routing_key = mock_retry_broker.publish_message.call_args[0][1]

    assert published_payload["job_id"] == new_job_id
    assert published_payload["retry_of_job_id"] == original_job_id
    assert published_payload["topic"] == original_job_data["topic"]
    assert published_payload["urls"] == original_job_data["urls"]
    assert published_payload["voice_id"] == original_job_data["voice_id"]
    assert published_payload["include_storyboard"] == original_job_data["include_storyboard"]
    assert published_payload["custom_prompt"] == original_job_data["custom_prompt"]
    assert published_payload["settings"] == original_job_data["settings"]
    assert published_payload["task_type"] == original_job_data["task_type"] # Should match original
    assert routing_key == app_config.redis.STREAM_NAME # Default routing key

    # Verify track_job_progress call for the original job
    assert len(mock_retry_track_job_progress_async.calls) == 1
    tracked_call_args = mock_retry_track_job_progress_async.calls[0]
    assert tracked_call_args[0] == original_job_id
    assert tracked_call_args[1]["status"] == "retried"
    assert tracked_call_args[1]["retried_with_job_id"] == new_job_id
    assert tracked_call_args[2] is True # publish_to_live should be True

    mock_api_logger.info.assert_any_call(f"Job {original_job_id} found with status 'failed'. Proceeding with retry.")
    mock_api_logger.info.assert_any_call(f"Successfully published retry job {new_job_id} (originally {original_job_id}). New message ID: {new_message_id}")
    mock_api_logger.info.assert_any_call(f"Successfully marked original job {original_job_id} as 'retried' with new job ID {new_job_id}.")


# Scenario 2: Job not found
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_retry_job_not_found(
    mock_is_valid_uuid,
    mock_get_broker,
    mock_api_logger
):
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True) # Should not be called
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True) # Should not be called
    mock_api_logger.warning.reset_mock()

    job_id_not_found = "job-does-not-exist-404"
    mock_is_valid_uuid.return_value = True
    
    # get_job_progress returns None, simulating job not found
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: None 

    response = client.post(f"/api/queue/job/{job_id_not_found}/retry")

    assert response.status_code == 404
    response_data = response.json()
    assert "Original job not found" in response_data["detail"]
    assert response_data["job_id"] == job_id_not_found

    mock_is_valid_uuid.assert_called_once_with(job_id_not_found)
    mock_retry_broker.initialize.assert_called_once()
    mock_retry_broker.get_job_progress.assert_called_once_with(job_id_not_found)
    mock_retry_broker.publish_message.assert_not_called()
    mock_retry_broker.track_job_progress.assert_not_called()
    
    mock_api_logger.warning.assert_called_once_with(f"Attempted to retry job {job_id_not_found}, but it was not found in the job store.")


# Scenario 3: Job not in "failed" state
@pytest.mark.parametrize("non_failed_status", ["completed", "processing", "queued"])
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_retry_job_not_in_failed_state(
    mock_is_valid_uuid,
    mock_get_broker,
    mock_api_logger,
    non_failed_status
):
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True)
    mock_api_logger.warning.reset_mock()

    job_id_not_failed = "job-not-failed-state-789"
    mock_is_valid_uuid.return_value = True
    
    job_data_not_failed = {
        "job_id": job_id_not_failed,
        "status": non_failed_status, # Not "failed"
        "topic": "Some Topic", "urls": ["http://example.com/some_source"] 
        # Other fields can be minimal as they won't be used for re-queuing
    }
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: job_data_not_failed if jid == job_id_not_failed else None

    response = client.post(f"/api/queue/job/{job_id_not_failed}/retry")

    assert response.status_code == 400
    response_data = response.json()
    assert f"Job {job_id_not_failed} cannot be retried because its status is '{non_failed_status}' (must be 'failed')." in response_data["detail"]
    assert response_data["job_id"] == job_id_not_failed

    mock_is_valid_uuid.assert_called_once_with(job_id_not_failed)
    mock_retry_broker.initialize.assert_called_once()
    mock_retry_broker.get_job_progress.assert_called_once_with(job_id_not_failed)
    mock_retry_broker.publish_message.assert_not_called()
    mock_retry_broker.track_job_progress.assert_not_called()
    
    mock_api_logger.warning.assert_called_once_with(
        f"Job {job_id_not_failed} cannot be retried because its status is '{non_failed_status}' (must be 'failed')."
    )


# Scenario 4: Invalid job_id format
@patch('viralStoryGenerator.src.api._logger') # To check for no error logs, or specific logs if any
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_retry_job_invalid_job_id_format(
    mock_is_valid_uuid,
    mock_get_broker,
    mock_api_logger
):
    mock_retry_broker.initialize.reset_mock(side_effect=True)
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True)
    
    invalid_job_id = "not-a-valid-uuid-format"
    mock_is_valid_uuid.return_value = False # Simulate invalid UUID

    response = client.post(f"/api/queue/job/{invalid_job_id}/retry")

    assert response.status_code == 400
    response_data = response.json()
    assert "Invalid job_id format" in response_data["detail"]
    assert response_data["job_id"] == invalid_job_id

    mock_is_valid_uuid.assert_called_once_with(invalid_job_id)
    # Broker methods should not be called
    mock_retry_broker.initialize.assert_not_called()
    mock_retry_broker.get_job_progress.assert_not_called()
    mock_retry_broker.publish_message.assert_not_called()
    mock_retry_broker.track_job_progress.assert_not_called()


# Scenario 5: Error during publish_message (re-queuing)
@pytest.mark.parametrize("publish_failure_mode", ["return_none", "raise_exception"])
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('uuid.uuid4') # Still need to mock this for new job_id generation
def test_retry_job_error_publish_message(
    mock_uuid4, # Mocked even if not directly used in assertion, as it's part of the flow
    mock_is_valid_uuid,
    mock_get_broker,
    mock_api_logger,
    publish_failure_mode
):
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True) # Should not be called
    mock_api_logger.error.reset_mock()
    mock_api_logger.info.reset_mock() # To check initial log messages

    original_job_id = "job-fail-on-publish-777"
    new_job_id = "new-job-for-failed-publish-888" # Generated before publish attempt
    
    mock_is_valid_uuid.return_value = True
    mock_uuid4.return_value = MagicMock(hex=new_job_id)

    original_job_data = {
        "job_id": original_job_id, "status": "failed",
        "topic": "Topic for Publish Fail", "urls": ["http://example.com/publish_fail"],
        "task_type": "generate_story_and_video", # Ensure this is present
        # Other fields can be minimal
        "settings": {} # Ensure settings is present
    }
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: original_job_data if jid == original_job_id else None

    if publish_failure_mode == "return_none":
        mock_retry_publish_message_async.custom_side_effect = lambda p, rk: None
    else: # raise_exception
        mock_retry_publish_message_async.custom_side_effect = MagicMock(side_effect=Exception("Broker publish error"))

    response = client.post(f"/api/queue/job/{original_job_id}/retry")

    assert response.status_code == 500
    response_data = response.json()
    assert "Failed to publish retry job to message broker" in response_data["detail"]
    assert response_data["original_job_id"] == original_job_id
    assert response_data["new_job_id"] == new_job_id # New job ID was generated

    mock_retry_broker.initialize.assert_called_once()
    mock_retry_broker.get_job_progress.assert_called_once_with(original_job_id)
    mock_retry_broker.publish_message.assert_called_once() # Attempted to publish
    mock_retry_broker.track_job_progress.assert_not_called() # Crucial: original job status not updated
    
    mock_api_logger.info.assert_any_call(f"Job {original_job_id} found with status 'failed'. Proceeding with retry.")
    mock_api_logger.error.assert_called_once()
    args, _ = mock_api_logger.error.call_args
    assert f"Failed to publish retry job {new_job_id} (originally {original_job_id}) to broker." in args[0]
    if publish_failure_mode == "raise_exception":
        assert "Broker publish error" in args[0] # Check if original exception is logged


# Scenario 6: Error during track_job_progress (updating original job)
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('uuid.uuid4')
def test_retry_job_error_track_progress(
    mock_uuid4,
    mock_is_valid_uuid,
    mock_get_broker,
    mock_api_logger
):
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True) # This will be configured to fail
    mock_api_logger.error.reset_mock()
    mock_api_logger.info.reset_mock()

    original_job_id = "job-fail-on-track-111"
    new_job_id = "new-job-for-failed-track-222"
    new_message_id = f"new_msg_id_for_{new_job_id}" # Publish succeeds

    mock_is_valid_uuid.return_value = True
    mock_uuid4.return_value = MagicMock(hex=new_job_id)

    original_job_data = {
        "job_id": original_job_id, "status": "failed",
        "topic": "Topic for Track Fail", "urls": ["http://example.com/track_fail"],
        "task_type": "generate_story_and_video", "settings": {}
    }
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: original_job_data if jid == original_job_id else None
    
    # publish_message succeeds
    mock_retry_publish_message_async.custom_side_effect = lambda p, rk: new_message_id
    
    # track_job_progress raises an exception
    mock_retry_track_job_progress_async.custom_side_effect = MagicMock(side_effect=Exception("Redis track_job_progress error"))

    response = client.post(f"/api/queue/job/{original_job_id}/retry")

    # The job was re-queued, but updating the original job's status failed.
    # The API might still return 200 because the primary action (re-queueing) succeeded.
    # Or it might return 500 if atomicity of the whole operation is desired.
    # Based on the current structure, a 500 seems more appropriate as an error occurred.
    assert response.status_code == 500 
    response_data = response.json()
    assert "Failed to update original job status to 'retried'" in response_data["detail"]
    assert response_data["original_job_id"] == original_job_id
    assert response_data["new_job_id"] == new_job_id
    assert response_data["new_message_id"] == new_message_id # New message ID is still part of the response

    mock_retry_broker.publish_message.assert_called_once() # Publish was successful
    mock_retry_broker.track_job_progress.assert_called_once() # Attempted to track
    
    mock_api_logger.info.assert_any_call(f"Successfully published retry job {new_job_id} (originally {original_job_id}). New message ID: {new_message_id}")
    mock_api_logger.error.assert_called_once()
    args, _ = mock_api_logger.error.call_args
    assert f"Failed to mark original job {original_job_id} as 'retried' after successful re-queue of new job {new_job_id}." in args[0]
    assert "Redis track_job_progress error" in args[0]


# Scenario 7: Authentication
@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_retry")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
def test_retry_job_auth_no_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_retry_broker.initialize.reset_mock(side_effect=True)
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    
    original_job_id = "job-auth-no-key-retry"
    response = client.post(f"/api/queue/job/{original_job_id}/retry")
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_retry_broker.initialize.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_retry")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
def test_retry_job_auth_invalid_api_key(mock_get_broker, mock_api_logger_ignored):
    mock_retry_broker.initialize.reset_mock(side_effect=True)
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)

    original_job_id = "job-auth-invalid-key-retry"
    headers = {"X-API-Key": "wrong_key_for_retry"}
    response = client.post(f"/api/queue/job/{original_job_id}/retry", headers=headers)
    
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_retry_broker.initialize.assert_not_called()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_retry")
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid') # To make it pass this check
@patch('uuid.uuid4') # To mock new job ID generation
def test_retry_job_auth_valid_api_key(
    mock_uuid4, mock_is_valid_uuid, mock_get_broker, mock_api_logger_ignored
):
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True); mock_retry_track_job_progress_async.calls = []

    original_job_id = "job-auth-valid-key-retry"
    new_job_id = "new-job-auth-valid"
    
    mock_is_valid_uuid.return_value = True
    mock_uuid4.return_value = MagicMock(hex=new_job_id)
    # Simulate a "failed" job being found
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: {"status": "failed", "job_id": jid, "task_type": "t", "settings":{}} if jid == original_job_id else None
    # Simulate successful publish and track
    mock_retry_publish_message_async.custom_side_effect = lambda p, rk: "new_msg_id"
    mock_retry_track_job_progress_async.custom_side_effect = None


    headers = {"X-API-Key": "test_api_key_retry"}
    response = client.post(f"/api/queue/job/{original_job_id}/retry", headers=headers)
    
    assert response.status_code == 200 # Successful retry
    assert response.json()["new_job_id"] == new_job_id
    
    mock_retry_broker.initialize.assert_called_once()
    mock_retry_broker.get_job_progress.assert_called_once_with(original_job_id)
    mock_retry_broker.publish_message.assert_called_once()
    mock_retry_broker.track_job_progress.assert_called_once()


@patch('viralStoryGenerator.src.api._logger')
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_retry_broker)
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
@patch('uuid.uuid4')
def test_retry_job_auth_disabled_no_key(
    mock_uuid4, mock_is_valid_uuid, mock_get_broker, mock_api_logger_ignored
):
    mock_retry_broker.initialize.reset_mock(side_effect=True); mock_cs_initialize_async.custom_side_effect = None
    mock_retry_broker.get_job_progress.reset_mock(side_effect=True)
    mock_retry_broker.publish_message.reset_mock(side_effect=True)
    mock_retry_broker.track_job_progress.reset_mock(side_effect=True); mock_retry_track_job_progress_async.calls = []

    original_job_id = "job-auth-disabled-retry"
    new_job_id = "new-job-auth-disabled"

    mock_is_valid_uuid.return_value = True
    mock_uuid4.return_value = MagicMock(hex=new_job_id)
    mock_retry_get_job_progress_async.custom_side_effect = lambda jid: {"status": "failed", "job_id": jid, "task_type": "t", "settings":{}} if jid == original_job_id else None
    mock_retry_publish_message_async.custom_side_effect = lambda p, rk: "new_msg_id"
    mock_retry_track_job_progress_async.custom_side_effect = None

    response = client.post(f"/api/queue/job/{original_job_id}/retry")

    assert response.status_code == 200
    assert response.json()["new_job_id"] == new_job_id

    mock_retry_broker.initialize.assert_called_once()
    mock_retry_broker.get_job_progress.assert_called_once_with(original_job_id)
    mock_retry_broker.publish_message.assert_called_once()
    mock_retry_broker.track_job_progress.assert_called_once()


# Tests for GET /api/queue/all-status

mock_all_queues_broker = MagicMock()

async def mock_aq_initialize_async():
    if hasattr(mock_aq_initialize_async, 'custom_side_effect'):
        return mock_aq_initialize_async.custom_side_effect()
    return None # Default successful init

async def mock_aq_get_queue_information_async(queue_name_arg_not_used_by_broker_method_itself):
    # The actual broker's get_queue_information might not take queue_name if it's configured at init
    # However, the endpoint loop passes queue_name to it.
    # This mock will use a dictionary to store responses per queue_name.
    if hasattr(mock_aq_get_queue_information_async, 'responses_by_queue'):
        # The endpoint calls get_queue_information on a *new broker instance* per queue.
        # So, this mock needs to be configured for the *current* broker instance's queue.
        # This means the test needs to patch the broker instance used for a specific queue.
        # A simpler approach is to have the mock function decide based on some external factor
        # or by patching the return value of get_message_broker for each iteration.
        # For now, let's assume the test will configure this mock's behavior for the specific queue being tested.
        current_queue_name = mock_aq_get_queue_information_async.current_queue_name_for_test
        if current_queue_name in mock_aq_get_queue_information_async.responses_by_queue:
            response_config = mock_aq_get_queue_information_async.responses_by_queue[current_queue_name]
            if isinstance(response_config, Exception):
                raise response_config
            return response_config

    # Default response if not specifically configured for the current_queue_name
    return {
        "status": "ok", "error_message": None, "stream_name": "default_stream",
        "stream_length": 10, "pending_messages_count": 1, "consumer_groups": [],
        "dead_letter_queue_length": 0, "recent_messages": []
    }
mock_aq_get_queue_information_async.current_queue_name_for_test = "" # Set this in tests
mock_aq_get_queue_information_async.responses_by_queue = {}


mock_all_queues_broker.initialize = MagicMock(side_effect=mock_aq_initialize_async)
mock_all_queues_broker.get_queue_information = MagicMock(side_effect=mock_aq_get_queue_information_async)


# Scenario 1: Successful retrieval for a single configured queue
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_all_queues_broker)
@patch('viralStoryGenerator.src.api.app_config.redis.QUEUE_NAME', "primary_test_queue")
@patch('viralStoryGenerator.src.api.app_config.redis.ADDITIONAL_QUEUES_TO_MONITOR', []) # No additional queues for this test
def test_get_all_queue_status_single_successful(mock_get_broker):
    mock_all_queues_broker.initialize.reset_mock(side_effect=True); mock_aq_initialize_async.custom_side_effect = None
    mock_all_queues_broker.get_queue_information.reset_mock(side_effect=True)

    primary_queue_name = "primary_test_queue"
    queue_data = {
        "status": "ok", "error_message": None, "stream_name": primary_queue_name,
        "stream_length": 50, "pending_messages_count": 5, 
        "consumer_groups": [{"name": "cg1", "consumers": [{"name":"c1","pending_messages":5, "inactive_time_ms":100}], "pending_messages": 5}],
        "dead_letter_queue_length": 1, 
        "recent_messages": [{"message_id": "mid1", "data": {"key": "val"}, "delivery_count":1, "last_delivered_ms":123}]
    }
    
    # Configure the mock for the primary queue
    mock_aq_get_queue_information_async.current_queue_name_for_test = primary_queue_name
    mock_aq_get_queue_information_async.responses_by_queue = {primary_queue_name: queue_data}

    response = client.get("/api/queue/all-status")

    assert response.status_code == 200
    response_data = response.json()
    
    assert primary_queue_name in response_data
    q_status = response_data[primary_queue_name]
    
    assert q_status["status"] == "ok"
    assert q_status["error_message"] is None
    assert q_status["stream_name"] == primary_queue_name
    assert q_status["stream_length"] == 50
    assert q_status["pending_messages_count"] == 5
    assert len(q_status["consumer_groups"]) == 1
    assert q_status["consumer_groups"][0]["name"] == "cg1"
    assert len(q_status["recent_messages"]) == 1
    assert q_status["recent_messages"][0]["message_id"] == "mid1"

    mock_all_queues_broker.initialize.assert_called_once() # Called once per broker instance
    # get_queue_information is called on the broker instance for its configured queue.
    # Since the endpoint creates a new broker per queue in the list, and we test one queue,
    # it should be called once on the (single) mocked broker instance returned.
    mock_all_queues_broker.get_queue_information.assert_called_once()


# Scenario 2: Broker returns error for a queue
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_all_queues_broker)
@patch('viralStoryGenerator.src.api.app_config.redis.QUEUE_NAME', "primary_queue_error")
@patch('viralStoryGenerator.src.api.app_config.redis.ADDITIONAL_QUEUES_TO_MONITOR', [])
def test_get_all_queue_status_broker_returns_error(mock_get_broker):
    mock_all_queues_broker.initialize.reset_mock(side_effect=True); mock_aq_initialize_async.custom_side_effect = None
    mock_all_queues_broker.get_queue_information.reset_mock(side_effect=True)

    queue_name = "primary_queue_error"
    error_data = {
        "status": "error", "error_message": "Redis connection refused",
        "stream_name": queue_name, "stream_length": 0, "pending_messages_count": 0,
        "consumer_groups": [], "dead_letter_queue_length": 0, "recent_messages": []
    }
    
    mock_aq_get_queue_information_async.current_queue_name_for_test = queue_name
    mock_aq_get_queue_information_async.responses_by_queue = {queue_name: error_data}

    response = client.get("/api/queue/all-status")

    assert response.status_code == 200
    response_data = response.json()
    
    assert queue_name in response_data
    q_status = response_data[queue_name]
    
    assert q_status["status"] == "error"
    assert q_status["error_message"] == "Redis connection refused"
    assert q_status["stream_name"] == queue_name # Still includes stream name
    assert q_status["stream_length"] == 0 # Defaults or error state values

    mock_all_queues_broker.initialize.assert_called_once()
    mock_all_queues_broker.get_queue_information.assert_called_once()


# Scenario 3: Broker raises an exception for a queue
@patch('viralStoryGenerator.src.api.get_message_broker', return_value=mock_all_queues_broker)
@patch('viralStoryGenerator.src.api.app_config.redis.QUEUE_NAME', "primary_queue_exception")
@patch('viralStoryGenerator.src.api.app_config.redis.ADDITIONAL_QUEUES_TO_MONITOR', [])
def test_get_all_queue_status_broker_raises_exception(mock_get_broker):
    mock_all_queues_broker.initialize.reset_mock(side_effect=True); mock_aq_initialize_async.custom_side_effect = None
    mock_all_queues_broker.get_queue_information.reset_mock(side_effect=True)

    queue_name = "primary_queue_exception"
    exception_message = "Simulated Redis connection timeout"
    
    mock_aq_get_queue_information_async.current_queue_name_for_test = queue_name
    # Configure get_queue_information to raise an exception for this queue
    mock_aq_get_queue_information_async.responses_by_queue = {queue_name: Exception(exception_message)}

    response = client.get("/api/queue/all-status")

    assert response.status_code == 200 # Endpoint handles the exception gracefully per queue
    response_data = response.json()
    
    assert queue_name in response_data
    q_status = response_data[queue_name]
    
    assert q_status["status"] == "error"
    assert q_status["error_message"] == f"Error retrieving queue information: {exception_message}"
    assert q_status["stream_name"] == queue_name # Still includes stream name in the error response
    assert q_status["stream_length"] is None # Or some default error value like 0/None
    assert q_status["pending_messages_count"] is None
    assert q_status["consumer_groups"] == [] # Empty list for error cases
    assert q_status["dead_letter_queue_length"] is None
    assert q_status["recent_messages"] == []


    mock_all_queues_broker.initialize.assert_called_once()
    mock_all_queues_broker.get_queue_information.assert_called_once()


# Scenario 4: Multiple queues
# For this test, we need get_message_broker to return a broker that's "aware" of the queue name
# it's supposed to be for, or we need to make its get_queue_information method respond dynamically.
# The endpoint creates a new broker for each queue name.
# So, we patch get_message_broker to return different mock broker instances,
# or one mock broker whose get_queue_information method is very smart.
# Let's try making get_message_broker return a pre-configured mock for each queue.

@patch('viralStoryGenerator.src.api.app_config.redis.QUEUE_NAME', "q_primary")
@patch('viralStoryGenerator.src.api.app_config.redis.ADDITIONAL_QUEUES_TO_MONITOR', ["q_additional1", "q_additional2"])
@patch('viralStoryGenerator.src.api.get_message_broker') # Patch the function that returns broker instances
def test_get_all_queue_status_multiple_queues(mock_get_message_broker_func):
    
    q_primary_name = "q_primary"
    q_add1_name = "q_additional1"
    q_add2_name = "q_additional2"

    q_primary_data = {
        "status": "ok", "error_message": None, "stream_name": q_primary_name,
        "stream_length": 100, "pending_messages_count": 10, "consumer_groups": [],
        "dead_letter_queue_length": 0, "recent_messages": []
    }
    q_add1_data = {
        "status": "ok", "error_message": None, "stream_name": q_add1_name,
        "stream_length": 20, "pending_messages_count": 2, "consumer_groups": [],
        "dead_letter_queue_length": 0, "recent_messages": []
    }
    q_add2_data_error = { # Simulate one queue having an error
        "status": "error", "error_message": "Specific error for q_additional2", 
        "stream_name": q_add2_name, "stream_length": 0, "pending_messages_count": 0,
        "consumer_groups": [], "dead_letter_queue_length": 0, "recent_messages": []
    }

    # This dictionary will map queue names to their specific mock broker instances
    mock_brokers_map = {}

    # Create and configure a mock broker for each queue
    for q_name, q_data in [(q_primary_name, q_primary_data), 
                           (q_add1_name, q_add1_data), 
                           (q_add2_name, q_add2_data_error)]:
        
        broker_instance_mock = MagicMock()
        
        # Mock initialize for this instance
        async def instance_init_async(): return None
        broker_instance_mock.initialize = MagicMock(side_effect=instance_init_async)
        
        # Mock get_queue_information for this instance to return its specific data
        async def instance_get_info_async(queue_name_arg_ignored=None): return q_data # Return data specific to this instance's queue
        broker_instance_mock.get_queue_information = MagicMock(side_effect=instance_get_info_async)
        
        mock_brokers_map[q_name] = broker_instance_mock

    # Configure get_message_broker to return the correct mock broker instance based on queue_name
    def get_broker_side_effect(queue_name_param, settings=None): # settings is the second arg to get_message_broker
        return mock_brokers_map[queue_name_param]

    mock_get_message_broker_func.side_effect = get_broker_side_effect

    response = client.get("/api/queue/all-status")
    assert response.status_code == 200
    response_data = response.json()

    assert len(response_data) == 3
    assert q_primary_name in response_data
    assert response_data[q_primary_name]["stream_name"] == q_primary_name
    assert response_data[q_primary_name]["stream_length"] == 100

    assert q_add1_name in response_data
    assert response_data[q_add1_name]["stream_name"] == q_add1_name
    assert response_data[q_add1_name]["stream_length"] == 20
    
    assert q_add2_name in response_data
    assert response_data[q_add2_name]["status"] == "error"
    assert response_data[q_add2_name]["error_message"] == "Specific error for q_additional2"

    # get_message_broker was called for each queue
    assert mock_get_message_broker_func.call_count == 3
    mock_get_message_broker_func.assert_any_call(queue_name=q_primary_name, settings=app_config.redis)
    mock_get_message_broker_func.assert_any_call(queue_name=q_add1_name, settings=app_config.redis)
    mock_get_message_broker_func.assert_any_call(queue_name=q_add2_name, settings=app_config.redis)

    # Check calls on individual broker mocks
    mock_brokers_map[q_primary_name].initialize.assert_called_once()
    mock_brokers_map[q_primary_name].get_queue_information.assert_called_once()
    mock_brokers_map[q_add1_name].initialize.assert_called_once()
    mock_brokers_map[q_add1_name].get_queue_information.assert_called_once()
    mock_brokers_map[q_add2_name].initialize.assert_called_once()
    mock_brokers_map[q_add2_name].get_queue_information.assert_called_once()


# Scenario 5: Authentication
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_allstatus")
@patch('viralStoryGenerator.src.api.get_message_broker') # To ensure it's not called
def test_get_all_queue_status_auth_no_api_key(mock_get_broker_func):
    response = client.get("/api/queue/all-status")
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_get_broker_func.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_allstatus")
@patch('viralStoryGenerator.src.api.get_message_broker')
def test_get_all_queue_status_auth_invalid_api_key(mock_get_broker_func):
    headers = {"X-API-Key": "wrong_key_for_allstatus"}
    response = client.get("/api/queue/all-status", headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_get_broker_func.assert_not_called()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "test_api_key_allstatus")
@patch('viralStoryGenerator.src.api.app_config.redis.QUEUE_NAME', "auth_q_primary")
@patch('viralStoryGenerator.src.api.app_config.redis.ADDITIONAL_QUEUES_TO_MONITOR', [])
@patch('viralStoryGenerator.src.api.get_message_broker')
def test_get_all_queue_status_auth_valid_api_key(mock_get_broker_func):
    queue_name = "auth_q_primary"
    queue_data = {"status": "ok", "stream_name": queue_name, "stream_length": 1} # Minimal data
    
    # Setup a mock broker instance for the get_message_broker function to return
    broker_instance_mock = MagicMock()
    async def instance_init_async(): return None
    broker_instance_mock.initialize = MagicMock(side_effect=instance_init_async)
    async def instance_get_info_async(queue_name_arg_ignored=None): return queue_data
    broker_instance_mock.get_queue_information = MagicMock(side_effect=instance_get_info_async)
    
    mock_get_broker_func.return_value = broker_instance_mock # Always return this instance

    headers = {"X-API-Key": "test_api_key_allstatus"}
    response = client.get("/api/queue/all-status", headers=headers)
    
    assert response.status_code == 200
    assert queue_name in response.json()
    assert response.json()[queue_name]["stream_name"] == queue_name
    mock_get_broker_func.assert_called_once_with(queue_name=queue_name, settings=app_config.redis)
    broker_instance_mock.initialize.assert_called_once()
    broker_instance_mock.get_queue_information.assert_called_once()


@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # Auth disabled
@patch('viralStoryGenerator.src.api.app_config.redis.QUEUE_NAME', "no_auth_q_primary")
@patch('viralStoryGenerator.src.api.app_config.redis.ADDITIONAL_QUEUES_TO_MONITOR', [])
@patch('viralStoryGenerator.src.api.get_message_broker')
def test_get_all_queue_status_auth_disabled_no_key(mock_get_broker_func):
    queue_name = "no_auth_q_primary"
    queue_data = {"status": "ok", "stream_name": queue_name, "stream_length": 2}
    
    broker_instance_mock = MagicMock()
    async def instance_init_async(): return None
    broker_instance_mock.initialize = MagicMock(side_effect=instance_init_async)
    async def instance_get_info_async(queue_name_arg_ignored=None): return queue_data
    broker_instance_mock.get_queue_information = MagicMock(side_effect=instance_get_info_async)
    
    mock_get_broker_func.return_value = broker_instance_mock

    response = client.get("/api/queue/all-status")
    
    assert response.status_code == 200
    assert queue_name in response.json()
    assert response.json()[queue_name]["stream_name"] == queue_name
    mock_get_broker_func.assert_called_once_with(queue_name=queue_name, settings=app_config.redis)


# Tests for Configuration Endpoints

# POST /api/config/storyboard
@pytest.mark.parametrize("initial_state", [True, False])
@pytest.mark.parametrize("target_state", [True, False])
@patch('viralStoryGenerator.src.api._logger')
def test_config_storyboard_enable_disable(mock_api_logger, initial_state, target_state, monkeypatch):
    # Ensure app_config is in a known state before the test and restored after
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', initial_state)
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False) # Assuming auth is tested separately or disabled for this

    response = client.post(f"/api/config/storyboard?enabled={str(target_state).lower()}")

    assert response.status_code == 200
    response_data = response.json()
    expected_message = f"Storyboard generation has been {'enabled' if target_state else 'disabled'}."
    assert response_data["message"] == expected_message
    assert response_data["new_status"] == target_state
    assert app_config.storyboard.ENABLE_STORYBOARD_GENERATION == target_state
    
    mock_api_logger.info.assert_called_once_with(
        f"ENABLE_STORYBOARD_GENERATION set to {target_state}. Previous: {initial_state}"
    )

def test_config_storyboard_missing_param(monkeypatch):
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)
    response = client.post("/api/config/storyboard") # No 'enabled' param
    assert response.status_code == 422 # FastAPI validation for missing query param

def test_config_storyboard_invalid_param_value(monkeypatch):
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)
    response = client.post("/api/config/storyboard?enabled=notabool")
    assert response.status_code == 422 # FastAPI validation for incorrect type

@patch('viralStoryGenerator.src.api._logger')
def test_config_storyboard_auth_no_api_key(mock_api_logger, monkeypatch):
    initial_config_value = app_config.storyboard.ENABLE_STORYBOARD_GENERATION
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "config_test_key")
    
    response = client.post("/api/config/storyboard?enabled=true") # No API key in header
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    # Ensure config was not changed
    assert app_config.storyboard.ENABLE_STORYBOARD_GENERATION == initial_config_value
    mock_api_logger.info.assert_not_called()

@patch('viralStoryGenerator.src.api._logger')
def test_config_storyboard_auth_invalid_api_key(mock_api_logger, monkeypatch):
    initial_config_value = app_config.storyboard.ENABLE_STORYBOARD_GENERATION
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "config_test_key")
    
    headers = {"X-API-Key": "wrong_config_key"}
    response = client.post("/api/config/storyboard?enabled=true", headers=headers)
    
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    assert app_config.storyboard.ENABLE_STORYBOARD_GENERATION == initial_config_value
    mock_api_logger.info.assert_not_called()

@patch('viralStoryGenerator.src.api._logger')
def test_config_storyboard_auth_valid_api_key(mock_api_logger, monkeypatch):
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_STORYBOARD_GENERATION', False) # Start from known state
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "config_test_key")
    
    headers = {"X-API-Key": "config_test_key"}
    response = client.post("/api/config/storyboard?enabled=true", headers=headers)
    
    assert response.status_code == 200
    assert app_config.storyboard.ENABLE_STORYBOARD_GENERATION is True
    mock_api_logger.info.assert_called_once_with(
        "ENABLE_STORYBOARD_GENERATION set to True. Previous: False"
    )

# POST /api/config/image-generation
@pytest.mark.parametrize("initial_state", [True, False])
@pytest.mark.parametrize("target_state", [True, False])
@patch('viralStoryGenerator.src.api._logger')
def test_config_image_generation_enable_disable(mock_api_logger, initial_state, target_state, monkeypatch):
    monkeypatch.setattr(app_config, 'ENABLE_IMAGE_GENERATION', initial_state)
    monkeypatch.setattr(app_config.openai, 'ENABLED', initial_state) # Assuming these are toggled together
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)

    response = client.post(f"/api/config/image-generation?enabled={str(target_state).lower()}")

    assert response.status_code == 200
    response_data = response.json()
    expected_message = f"Image generation (OpenAI) has been {'enabled' if target_state else 'disabled'}."
    assert response_data["message"] == expected_message
    assert response_data["new_status"] == target_state
    assert app_config.ENABLE_IMAGE_GENERATION == target_state
    assert app_config.openai.ENABLED == target_state # Check both are set
    
    mock_api_logger.info.assert_called_once_with(
        f"Image generation (app_config.ENABLE_IMAGE_GENERATION and app_config.openai.ENABLED) set to {target_state}. Previous: {initial_state}"
    )

def test_config_image_generation_missing_param(monkeypatch):
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)
    response = client.post("/api/config/image-generation")
    assert response.status_code == 422

def test_config_image_generation_invalid_param_value(monkeypatch):
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)
    response = client.post("/api/config/image-generation?enabled=badvalue")
    assert response.status_code == 422

@patch('viralStoryGenerator.src.api._logger')
def test_config_image_generation_auth_no_api_key(mock_api_logger, monkeypatch):
    initial_app_config_val = app_config.ENABLE_IMAGE_GENERATION
    initial_openai_val = app_config.openai.ENABLED
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "img_config_key")
    
    response = client.post("/api/config/image-generation?enabled=true")
    
    assert response.status_code == 401
    assert app_config.ENABLE_IMAGE_GENERATION == initial_app_config_val
    assert app_config.openai.ENABLED == initial_openai_val
    mock_api_logger.info.assert_not_called()

@patch('viralStoryGenerator.src.api._logger')
def test_config_image_generation_auth_invalid_api_key(mock_api_logger, monkeypatch):
    initial_app_config_val = app_config.ENABLE_IMAGE_GENERATION
    initial_openai_val = app_config.openai.ENABLED
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "img_config_key")
    
    headers = {"X-API-Key": "wrong_img_key"}
    response = client.post("/api/config/image-generation?enabled=true", headers=headers)
    
    assert response.status_code == 401
    assert app_config.ENABLE_IMAGE_GENERATION == initial_app_config_val
    assert app_config.openai.ENABLED == initial_openai_val
    mock_api_logger.info.assert_not_called()

@patch('viralStoryGenerator.src.api._logger')
def test_config_image_generation_auth_valid_api_key(mock_api_logger, monkeypatch):
    monkeypatch.setattr(app_config, 'ENABLE_IMAGE_GENERATION', False)
    monkeypatch.setattr(app_config.openai, 'ENABLED', False)
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "img_config_key")
    
    headers = {"X-API-Key": "img_config_key"}
    response = client.post("/api/config/image-generation?enabled=true", headers=headers)
    
    assert response.status_code == 200
    assert app_config.ENABLE_IMAGE_GENERATION is True
    assert app_config.openai.ENABLED is True
    mock_api_logger.info.assert_called_once_with(
        "Image generation (app_config.ENABLE_IMAGE_GENERATION and app_config.openai.ENABLED) set to True. Previous: False"
    )

# POST /api/config/audio-generation
@pytest.mark.parametrize("initial_state", [True, False])
@pytest.mark.parametrize("target_state", [True, False])
@patch('viralStoryGenerator.src.api._logger')
def test_config_audio_generation_enable_disable(mock_api_logger, initial_state, target_state, monkeypatch):
    monkeypatch.setattr(app_config, 'ENABLE_AUDIO_GENERATION', initial_state)
    monkeypatch.setattr(app_config.elevenlabs, 'ENABLED', initial_state) # Assuming these are toggled together
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)

    response = client.post(f"/api/config/audio-generation?enabled={str(target_state).lower()}")

    assert response.status_code == 200
    response_data = response.json()
    expected_message = f"Audio generation (ElevenLabs) has been {'enabled' if target_state else 'disabled'}."
    assert response_data["message"] == expected_message
    assert response_data["new_status"] == target_state
    assert app_config.ENABLE_AUDIO_GENERATION == target_state
    assert app_config.elevenlabs.ENABLED == target_state # Check both are set
    
    mock_api_logger.info.assert_called_once_with(
        f"Audio generation (app_config.ENABLE_AUDIO_GENERATION and app_config.elevenlabs.ENABLED) set to {target_state}. Previous: {initial_state}"
    )

def test_config_audio_generation_missing_param(monkeypatch):
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)
    response = client.post("/api/config/audio-generation")
    assert response.status_code == 422

def test_config_audio_generation_invalid_param_value(monkeypatch):
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', False)
    response = client.post("/api/config/audio-generation?enabled=badvalue")
    assert response.status_code == 422

@patch('viralStoryGenerator.src.api._logger')
def test_config_audio_generation_auth_no_api_key(mock_api_logger, monkeypatch):
    initial_app_config_val = app_config.ENABLE_AUDIO_GENERATION
    initial_elevenlabs_val = app_config.elevenlabs.ENABLED
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "audio_config_key")
    
    response = client.post("/api/config/audio-generation?enabled=true")
    
    assert response.status_code == 401
    assert app_config.ENABLE_AUDIO_GENERATION == initial_app_config_val
    assert app_config.elevenlabs.ENABLED == initial_elevenlabs_val
    mock_api_logger.info.assert_not_called()

@patch('viralStoryGenerator.src.api._logger')
def test_config_audio_generation_auth_invalid_api_key(mock_api_logger, monkeypatch):
    initial_app_config_val = app_config.ENABLE_AUDIO_GENERATION
    initial_elevenlabs_val = app_config.elevenlabs.ENABLED
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "audio_config_key")
    
    headers = {"X-API-Key": "wrong_audio_key"}
    response = client.post("/api/config/audio-generation?enabled=true", headers=headers)
    
    assert response.status_code == 401
    assert app_config.ENABLE_AUDIO_GENERATION == initial_app_config_val
    assert app_config.elevenlabs.ENABLED == initial_elevenlabs_val
    mock_api_logger.info.assert_not_called()

@patch('viralStoryGenerator.src.api._logger')
def test_config_audio_generation_auth_valid_api_key(mock_api_logger, monkeypatch):
    monkeypatch.setattr(app_config, 'ENABLE_AUDIO_GENERATION', False)
    monkeypatch.setattr(app_config.elevenlabs, 'ENABLED', False)
    monkeypatch.setattr(app_config.http, 'API_KEY_ENABLED', True)
    monkeypatch.setattr(app_config.http, 'API_KEY', "audio_config_key")
    
    headers = {"X-API-Key": "audio_config_key"}
    response = client.post("/api/config/audio-generation?enabled=true", headers=headers)
    
    assert response.status_code == 200
    assert app_config.ENABLE_AUDIO_GENERATION is True
    assert app_config.elevenlabs.ENABLED is True
    mock_api_logger.info.assert_called_once_with(
        "Audio generation (app_config.ENABLE_AUDIO_GENERATION and app_config.elevenlabs.ENABLED) set to True. Previous: False"
    )


@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_not_found(mock_is_valid_uuid, mock_get_task_status):
    mock_is_valid_uuid.return_value = True
    task_id = "valid-uuid-404"
    mock_get_task_status.return_value = None # Task not found by handler

    response = client.get(f"/api/stories/{task_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_get_task_status.assert_called_once_with(task_id)

@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_invalid_uuid_format(mock_is_valid_uuid, mock_get_task_status):
    mock_is_valid_uuid.return_value = False
    task_id = "not-a-valid-uuid"

    response = client.get(f"/api/stories/{task_id}")

    assert response.status_code == 400
    assert "Invalid task_id format" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_get_task_status.assert_not_called()

@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_handler_exception(mock_is_valid_uuid, mock_get_task_status):
    mock_is_valid_uuid.return_value = True
    task_id = "valid-uuid-500"
    mock_get_task_status.side_effect = Exception("Handler crashed")

    response = client.get(f"/api/stories/{task_id}")

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_get_task_status.assert_called_once_with(task_id)


# Authentication Tests for GET /api/stories/{task_id}
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey_get")
@patch('viralStoryGenerator.src.api_handlers.get_task_status') # To ensure it's not called
@patch('viralStoryGenerator.utils.security.is_valid_uuid') # To ensure it's not called if auth fails first
def test_get_story_status_auth_no_api_key(mock_is_valid_uuid, mock_get_task_status):
    task_id = "some-uuid-no-auth"
    # No headers provided
    response = client.get(f"/api/stories/{task_id}")

    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]
    mock_is_valid_uuid.assert_not_called()
    mock_get_task_status.assert_not_called()

@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey_get")
@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_auth_invalid_api_key(mock_is_valid_uuid, mock_get_task_status):
    task_id = "some-uuid-invalid-auth"
    headers = {"X-API-Key": "wrongapikey_get"}
    response = client.get(f"/api/stories/{task_id}", headers=headers)

    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
    mock_is_valid_uuid.assert_not_called()
    mock_get_task_status.assert_not_called()

@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', True)
@patch('viralStoryGenerator.src.api.app_config.http.API_KEY', "testapikey_get")
@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_auth_valid_api_key(mock_is_valid_uuid, mock_get_task_status):
    mock_is_valid_uuid.return_value = True # Auth passes, so this will be called
    task_id = "valid-uuid-authed"
    expected_status = {"task_id": task_id, "status": "processing"}
    mock_get_task_status.return_value = expected_status
    
    headers = {"X-API-Key": "testapikey_get"}
    response = client.get(f"/api/stories/{task_id}", headers=headers)

    assert response.status_code == 200
    assert response.json() == expected_status
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_get_task_status.assert_called_once_with(task_id)

@patch('viralStoryGenerator.src.api.app_config.http.API_KEY_ENABLED', False) # API key auth disabled
@patch('viralStoryGenerator.src.api_handlers.get_task_status')
@patch('viralStoryGenerator.utils.security.is_valid_uuid')
def test_get_story_status_auth_disabled_no_key(mock_is_valid_uuid, mock_get_task_status):
    mock_is_valid_uuid.return_value = True
    task_id = "valid-uuid-auth-disabled"
    expected_status = {"task_id": task_id, "status": "queued"}
    mock_get_task_status.return_value = expected_status

    response = client.get(f"/api/stories/{task_id}")

    assert response.status_code == 200
    assert response.json() == expected_status
    mock_is_valid_uuid.assert_called_once_with(task_id)
    mock_get_task_status.assert_called_once_with(task_id)
