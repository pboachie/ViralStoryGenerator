import pytest
import asyncio
import json
import os
import time
import shutil
import uuid
from unittest.mock import patch, MagicMock, mock_open, AsyncMock

# Assuming the module is viralStoryGenerator.utils.health_check
from viralStoryGenerator.utils import health_check as hc_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.models import ServiceStatus, ServiceStatusDetail # For type hints/obj creation

# --- Global Mocks & Fixtures ---

MOCK_STATUS_FILE_PATH = "/tmp/mock_service_status_history.json"

@pytest.fixture(autouse=True)
def mock_appconfig_health_check_defaults(monkeypatch):
    """Set default app_config values for health_check tests."""
    # Service enabled flags
    monkeypatch.setattr(app_config.redis, 'ENABLED', True)
    monkeypatch.setattr(app_config.elevenlabs, 'ENABLED', True)
    monkeypatch.setattr(app_config.llm, 'ENABLED', True) # Assuming generic LLM enabled flag
    monkeypatch.setattr(app_config.dalle, 'ENABLED', True) # For image generation checks if any

    # API Keys (can be dummy for most tests unless specific auth failure is tested)
    monkeypatch.setattr(app_config.elevenlabs, 'API_KEY', "dummy_eleven_key")
    monkeypatch.setattr(app_config.llm, 'API_KEY', "dummy_llm_key") # If LLM uses a key directly

    # Endpoints
    monkeypatch.setattr(app_config.elevenlabs, 'API_ENDPOINT_USER_INFO', "http://mock.elevenlabs/user/info")
    monkeypatch.setattr(app_config.llm, 'ENDPOINT', "http://mock.llm/api/status") # Example status endpoint

    # Redis
    monkeypatch.setattr(app_config.redis, 'HOST', "mock_redis_hc_host")
    monkeypatch.setattr(app_config.redis, 'PORT', 6379)
    monkeypatch.setattr(app_config.redis, 'PASSWORD', None)
    
    # Disk status
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "local") # Default to local for disk checks
    monkeypatch.setattr(app_config.storage, 'LOCAL_STORAGE_PATH', "/tmp/mock_local_storage_hc")
    monkeypatch.setattr(app_config.health_check, 'MIN_DISK_SPACE_GB_WARNING', 10)
    monkeypatch.setattr(app_config.health_check, 'MIN_DISK_SPACE_GB_CRITICAL', 2)

    # ServiceTracker settings
    monkeypatch.setattr(app_config.health_check, 'STATUS_HISTORY_FILE', MOCK_STATUS_FILE_PATH)
    monkeypatch.setattr(app_config.health_check, 'MAX_RESPONSE_TIME_SAMPLES', 10)
    monkeypatch.setattr(app_config.health_check, 'MAX_STATUS_HISTORY_ENTRIES', 50)
    
    # General
    monkeypatch.setattr(app_config, 'ENVIRONMENT', "test_env")
    monkeypatch.setattr(app_config, 'VERSION', "hc_test_v1")
    
    # Ensure the mock local storage path exists for relevant tests
    if not os.path.exists(app_config.storage.LOCAL_STORAGE_PATH):
        os.makedirs(app_config.storage.LOCAL_STORAGE_PATH, exist_ok=True)
    
    yield
    
    # Cleanup mock status file if created
    if os.path.exists(MOCK_STATUS_FILE_PATH):
        os.remove(MOCK_STATUS_FILE_PATH)
    # Cleanup mock storage path if created by this fixture (more robust cleanup might be needed if tests write here)
    # if os.path.exists(app_config.storage.LOCAL_STORAGE_PATH) and "mock_local_storage_hc" in app_config.storage.LOCAL_STORAGE_PATH:
    #     shutil.rmtree(app_config.storage.LOCAL_STORAGE_PATH)


@pytest.fixture
def mock_health_check_logger():
    """Fixture to mock the _logger in health_check.py."""
    with patch('viralStoryGenerator.utils.health_check._logger') as mock_logger:
        yield mock_logger

@pytest.fixture
def mock_time_time():
    """Fixture to mock time.time() for controlling timestamps."""
    # Use a class to allow advancing time
    class MockTime:
        def __init__(self, start_time=1000.0):
            self.current_time = start_time
        
        def time(self):
            return self.current_time
        
        def advance(self, seconds):
            self.current_time += seconds

    mock_time_obj = MockTime()
    with patch('time.time', side_effect=mock_time_obj.time) as mock_time:
        # Attach the controller object to the mock so tests can advance time
        mock_time.controller = mock_time_obj 
        yield mock_time


# --- Tests for ServiceTracker (Scenario 1: Initialization) ---

def test_service_tracker_init_no_file(mock_health_check_logger, mock_appconfig_health_check_defaults):
    # Ensure file does not exist
    if os.path.exists(MOCK_STATUS_FILE_PATH):
        os.remove(MOCK_STATUS_FILE_PATH)
    
    tracker = hc_module.ServiceTracker()
    
    assert tracker.status_history == {}
    mock_health_check_logger.info.assert_any_call(
        f"Service status history file not found at {MOCK_STATUS_FILE_PATH}. Starting with empty history."
    )

@patch('builtins.open', new_callable=mock_open)
def test_service_tracker_init_invalid_json_file(
    mock_file_open, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    # Simulate file exists but contains invalid JSON
    with patch('os.path.exists', return_value=True): # Mock os.path.exists for this scope
        mock_file_open.return_value.read.return_value = "this is not valid json"
        # json.load will raise JSONDecodeError
        with patch('json.load', side_effect=json.JSONDecodeError("err", "doc", 0)):
            tracker = hc_module.ServiceTracker()
            
    assert tracker.status_history == {} # Should default to empty
    mock_health_check_logger.error.assert_any_call(
        f"Error loading service status history from {MOCK_STATUS_FILE_PATH}: Expecting value: line 1 column 1 (char 0). Starting with empty history."
    )


@patch('builtins.open', new_callable=mock_open)
def test_service_tracker_init_valid_json_file(
    mock_file_open, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    valid_history_data = {
        "redis": {
            "current_status": "healthy",
            "last_status_change": 900.0,
            "uptime_start": 800.0,
            "total_uptime": 100.0,
            "response_times": [0.1, 0.2],
            "status_log": [{"timestamp": 800.0, "status": "unhealthy"}, {"timestamp": 900.0, "status": "healthy"}]
        }
    }
    mock_file_open.return_value.read.return_value = json.dumps(valid_history_data)
    
    with patch('os.path.exists', return_value=True):
        with patch('json.load', return_value=valid_history_data) as mock_json_load_valid:
             tracker = hc_module.ServiceTracker()

    assert tracker.status_history == valid_history_data
    mock_json_load_valid.assert_called_once()
    mock_health_check_logger.info.assert_any_call(
        f"Successfully loaded service status history from {MOCK_STATUS_FILE_PATH}."
    )


@patch('builtins.open', side_effect=IOError("Cannot open file")) # Generic IOError
def test_service_tracker_init_file_permission_error(
    mock_file_open_io_error, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    with patch('os.path.exists', return_value=True):
        tracker = hc_module.ServiceTracker()
            
    assert tracker.status_history == {} # Should default to empty
    mock_health_check_logger.error.assert_any_call(
        f"Error loading service status history from {MOCK_STATUS_FILE_PATH}: Cannot open file. Starting with empty history."
    )


# --- Tests for ServiceTracker._save_status_history & _load_status_history (Scenario 1.x, covered by init) ---
# The init tests already cover _load_status_history implicitly.
# _save_status_history will be tested via update_service_status.

# --- Tests for ServiceTracker.update_service_status (Scenario 2) ---

@pytest.fixture
def tracker_with_mocked_save(mock_appconfig_health_check_defaults):
    """Provides a ServiceTracker instance with _save_status_history mocked."""
    with patch.object(hc_module.ServiceTracker, '_save_status_history', MagicMock()) as mock_save:
        tracker = hc_module.ServiceTracker()
        # Attach the mock_save to the instance for assertions if needed, though usually not directly
        tracker._save_status_history_mock = mock_save 
        yield tracker


def test_update_service_status_new_service_healthy(
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger
):
    service_name = "new_service_1"
    current_time = mock_time_time.controller.current_time
    
    tracker_with_mocked_save.update_service_status(service_name, "healthy", message="First check", response_time_ms=100.0)
    
    assert service_name in tracker_with_mocked_save.status_history
    service_info = tracker_with_mocked_save.status_history[service_name]
    
    assert service_info["current_status"] == "healthy"
    assert service_info["message"] == "First check"
    assert service_info["last_checked"] == current_time
    assert service_info["last_status_change"] == current_time # Changed from None to healthy
    assert service_info["uptime_start"] == current_time # Started as healthy
    assert service_info["total_uptime"] == 0 # No previous uptime
    assert service_info["response_times"] == [100.0]
    assert len(service_info["status_log"]) == 1
    assert service_info["status_log"][0] == {"timestamp": current_time, "status": "healthy", "message": "First check"}
    
    tracker_with_mocked_save._save_status_history_mock.assert_called_once()


def test_update_service_status_existing_service_status_change(
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger
):
    service_name = "existing_service_1"
    initial_time = mock_time_time.controller.current_time
    
    # Initial healthy status
    tracker_with_mocked_save.update_service_status(service_name, "healthy", response_time_ms=50.0)
    
    # Advance time and change status to unhealthy
    mock_time_time.controller.advance(100) # Advance 100 seconds
    unhealthy_time = mock_time_time.controller.current_time
    
    tracker_with_mocked_save.update_service_status(service_name, "unhealthy", message="Went down", response_time_ms=None) # No response time for failure
    
    service_info = tracker_with_mocked_save.status_history[service_name]
    assert service_info["current_status"] == "unhealthy"
    assert service_info["message"] == "Went down"
    assert service_info["last_checked"] == unhealthy_time
    assert service_info["last_status_change"] == unhealthy_time # Time of this status change
    assert service_info["uptime_start"] is None # No longer healthy, so uptime_start is None
    assert service_info["total_uptime"] == 100.0 # Was healthy for 100 seconds (unhealthy_time - initial_time)
    assert service_info["response_times"] == [50.0] # Only one successful response time
    assert len(service_info["status_log"]) == 2
    assert service_info["status_log"][1]["status"] == "unhealthy"
    
    # Change back to healthy
    mock_time_time.controller.advance(200) # Down for 200 seconds
    healthy_again_time = mock_time_time.controller.current_time
    tracker_with_mocked_save.update_service_status(service_name, "healthy", message="Back up!", response_time_ms=70.0)
    
    service_info = tracker_with_mocked_save.status_history[service_name]
    assert service_info["current_status"] == "healthy"
    assert service_info["uptime_start"] == healthy_again_time # New uptime_start
    assert service_info["last_status_change"] == healthy_again_time
    assert service_info["total_uptime"] == 100.0 # total_uptime only accumulates completed healthy periods
    assert service_info["response_times"] == [50.0, 70.0] # Added new response time
    assert len(service_info["status_log"]) == 3
    assert service_info["status_log"][2]["status"] == "healthy"


def test_update_service_status_response_times_limited(
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger, monkeypatch
):
    service_name = "service_response_times"
    max_samples = 3
    monkeypatch.setattr(app_config.health_check, 'MAX_RESPONSE_TIME_SAMPLES', max_samples)
    
    for i in range(max_samples + 2): # Add more samples than limit
        mock_time_time.controller.advance(1)
        tracker_with_mocked_save.update_service_status(service_name, "healthy", response_time_ms=float(i * 10))
        
    service_info = tracker_with_mocked_save.status_history[service_name]
    assert len(service_info["response_times"]) == max_samples
    # Should contain the *last* max_samples items
    assert service_info["response_times"] == [20.0, 30.0, 40.0] # Samples for i=2,3,4


def test_update_service_status_status_log_limited(
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger, monkeypatch
):
    service_name = "service_status_log"
    max_log_entries = 3
    monkeypatch.setattr(app_config.health_check, 'MAX_STATUS_HISTORY_ENTRIES', max_log_entries)
    
    base_time = mock_time_time.controller.current_time
    for i in range(max_log_entries + 2): # Add more log entries than limit
        mock_time_time.controller.current_time = base_time + i # Ensure distinct timestamps
        status = "healthy" if i % 2 == 0 else "degraded"
        tracker_with_mocked_save.update_service_status(service_name, status, message=f"Update {i}")
        
    service_info = tracker_with_mocked_save.status_history[service_name]
    assert len(service_info["status_log"]) == max_log_entries
    # Should contain the *last* max_log_entries items
    # Updates are i=0 (h), i=1 (d), i=2 (h), i=3 (d), i=4 (h)
    # Log should have i=2, i=3, i=4
    assert service_info["status_log"][0]["message"] == "Update 2"
    assert service_info["status_log"][1]["message"] == "Update 3"
    assert service_info["status_log"][2]["message"] == "Update 4"


def test_update_service_status_no_status_change_no_log_or_uptime_start_reset(
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger
):
    service_name = "service_no_change"
    initial_time = mock_time_time.controller.current_time
    
    tracker_with_mocked_save.update_service_status(service_name, "healthy", message="Initial", response_time_ms=10.0)
    initial_log_count = len(tracker_with_mocked_save.status_history[service_name]["status_log"])
    initial_uptime_start = tracker_with_mocked_save.status_history[service_name]["uptime_start"]
    initial_last_status_change = tracker_with_mocked_save.status_history[service_name]["last_status_change"]

    mock_time_time.controller.advance(10)
    # Update with same status
    tracker_with_mocked_save.update_service_status(service_name, "healthy", message="Still fine", response_time_ms=12.0)
    
    service_info = tracker_with_mocked_save.status_history[service_name]
    assert len(service_info["status_log"]) == initial_log_count # No new log entry for same status
    assert service_info["uptime_start"] == initial_uptime_start # Uptime start should not change
    assert service_info["last_status_change"] == initial_last_status_change # Last status change time should not change
    assert service_info["response_times"] == [10.0, 12.0] # Response time still updated
    assert service_info["last_checked"] == mock_time_time.controller.current_time # Last checked is updated
    assert service_info["message"] == "Still fine" # Message is updated

# Scenario 3: ServiceTracker._save_status_history

@patch('builtins.open', new_callable=mock_open)
@patch('os.replace') # For atomic write
@patch('json.dump')
@patch('tempfile.NamedTemporaryFile')
def test_save_status_history_successful_write(
    mock_named_temp_file, mock_json_dump, mock_os_replace, mock_file_open_save,
    mock_health_check_logger, mock_appconfig_health_check_defaults
):
    # Setup a ServiceTracker instance without mocking _save_status_history itself
    tracker = hc_module.ServiceTracker() 
    
    # Populate some history
    tracker.status_history = {"service1": {"status": "test"}}
    
    # Mock the behavior of NamedTemporaryFile
    # It needs to act like a context manager and provide a 'name' attribute for the temp file
    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.name = "/tmp/mock_service_status_history.json.tmp" # Example temp file name
    # Ensure the file object itself is returned by __enter__ for the 'with' statement
    mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_obj
    # Ensure it's treated as a file stream for json.dump
    # json.dump expects a file-like object with a `write` method.
    # The NamedTemporaryFile instance itself is usually the file object in write mode.
    # So, mock_named_temp_file.return_value should be what json.dump gets.
    
    # Call the method (it's called internally by update_service_status, or can be called directly if made public for testing)
    # For this test, let's assume we can call it directly for focused testing.
    # If it's strictly private, we test it via update_service_status.
    # The prompt implies testing _save_status_history.
    
    # Make _save_status_history public for this test or test via update_service_status.
    # Let's test via update_service_status as that's its natural call path.
    
    # Re-patch _save_status_history for other tests in tracker_with_mocked_save,
    # but for this one, we want the real method to run.
    
    # To test the real _save_status_history, we need an instance where it's NOT mocked.
    # The `tracker` instance created above is suitable.

    tracker._save_status_history() # Call it directly

    mock_named_temp_file.assert_called_once_with(mode='w', delete=False, dir=os.path.dirname(MOCK_STATUS_FILE_PATH))
    # json.dump is called with the history and the file object from NamedTemporaryFile
    mock_json_dump.assert_called_once_with(tracker.status_history, mock_temp_file_obj, indent=4)
    mock_os_replace.assert_called_once_with(mock_temp_file_obj.name, MOCK_STATUS_FILE_PATH)
    mock_health_check_logger.debug.assert_any_call(f"Service status history saved to {MOCK_STATUS_FILE_PATH}")


@patch('tempfile.NamedTemporaryFile', side_effect=IOError("Disk full making temp file"))
def test_save_status_history_temp_file_creation_fails(
    mock_named_temp_file_io_error, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    tracker = hc_module.ServiceTracker()
    tracker.status_history = {"service_temp_fail": {"status": "error"}}

    tracker._save_status_history()

    mock_health_check_logger.error.assert_any_call(
        f"Error saving service status history (temp file stage) to {MOCK_STATUS_FILE_PATH}: Disk full making temp file"
    )


@patch('tempfile.NamedTemporaryFile')
@patch('json.dump', side_effect=TypeError("Data not serializable"))
def test_save_status_history_json_dump_fails(
    mock_json_dump_type_error, mock_named_temp_file_jd_fail, 
    mock_health_check_logger, mock_appconfig_health_check_defaults
):
    tracker = hc_module.ServiceTracker()
    tracker.status_history = {"service_json_fail": {"status": "error"}}

    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.name = "/tmp/mock_service_status_history.json.tmp_jd_fail"
    mock_named_temp_file_jd_fail.return_value.__enter__.return_value = mock_temp_file_obj
    
    tracker._save_status_history()

    mock_health_check_logger.error.assert_any_call(
        f"Error saving service status history (JSON dump stage) to {MOCK_STATUS_FILE_PATH}: Data not serializable"
    )


@patch('tempfile.NamedTemporaryFile')
@patch('json.dump') # Assume dump succeeds
@patch('os.replace', side_effect=OSError("Permission denied for replace"))
def test_save_status_history_os_replace_fails(
    mock_os_replace_os_error, mock_json_dump_replace_fail, mock_named_temp_file_replace_fail,
    mock_health_check_logger, mock_appconfig_health_check_defaults
):
    tracker = hc_module.ServiceTracker()
    tracker.status_history = {"service_replace_fail": {"status": "error"}}

    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.name = "/tmp/mock_service_status_history.json.tmp_replace_fail"
    mock_named_temp_file_replace_fail.return_value.__enter__.return_value = mock_temp_file_obj
    
    tracker._save_status_history()

    mock_health_check_logger.error.assert_any_call(
        f"Error saving service status history (file replace stage) to {MOCK_STATUS_FILE_PATH}: Permission denied for replace"
    )
    # Check if the temporary file was attempted to be cleaned up if os.replace fails
    # This depends on the implementation of _save_status_history's finally block for the temp file.
    # Assuming it tries to os.remove(temp_file_path) in a finally.
    with patch('os.remove') as mock_os_remove_temp:
        # We need to re-run with the specific error to test the finally block's os.remove
        # This means the side_effect for os.replace should be active.
        # The previous call to _save_status_history already happened with the error.
        # To specifically test the finally clause of the tempfile, we need to ensure
        # an error happens *after* tempfile creation but *before* or *during* os.replace.
        # The current test structure for os.replace failure covers this.
        # The temp file should be removed if os.replace fails.
        mock_os_remove_temp.assert_not_called() # os.remove on the *original* file shouldn't be called.
        # The test for os.remove on the temp file in case of os.replace error is implicitly hard
        # to isolate here without refactoring _save_status_history.
        # However, the NamedTemporaryFile(delete=False) means it's our responsibility.
        # The code *should* have a try/finally for the temp_file_path.
        # `with tempfile.NamedTemporaryFile(...) as tmpfile:` handles this automatically if delete=True.
        # With delete=False, manual cleanup is needed on error.
        # The current _save_status_history has a finally clause for os.remove(temp_file_path)
        # only if an exception occurs *during the `with open(tmpfile.name, 'w')` block* or *during os.replace*.

        # Let's refine the test for os.replace failure and check temp file removal:
        tracker_replace_fail = hc_module.ServiceTracker()
        tracker_replace_fail.status_history = {"service_replace_fail_cleanup": {"status": "testing_cleanup"}}
        
        mock_temp_file_obj_cleanup = MagicMock()
        temp_file_path_for_cleanup = "/tmp/temp_for_cleanup_test.tmp"
        mock_temp_file_obj_cleanup.name = temp_file_path_for_cleanup
        
        # Reset and re-configure mocks for this specific sub-test
        mock_named_temp_file_replace_fail.reset_mock(return_value=True, side_effect=None) # Clear previous side_effect
        mock_named_temp_file_replace_fail.return_value.__enter__.return_value = mock_temp_file_obj_cleanup
        mock_json_dump_replace_fail.reset_mock(side_effect=None) # Clear previous side_effect
        mock_os_replace_os_error.reset_mock(side_effect=OSError("Permission denied for replace")) # Re-apply specific error

        with patch('os.remove') as mock_os_remove_temp_explicit:
            tracker_replace_fail._save_status_history()
            mock_os_remove_temp_explicit.assert_called_once_with(temp_file_path_for_cleanup)


# Scenario 4: ServiceTracker.get_service_uptime
def test_get_service_uptime_calculations(tracker_with_mocked_save, mock_time_time):
    tracker = tracker_with_mocked_save # Uses the fixture that mocks _save_status_history
    service_name = "uptime_svc"
    
    # Case 1: Service never recorded (should be 0)
    assert tracker.get_service_uptime(service_name) == 0.0

    # Case 2: Service is new and healthy
    mock_time_time.controller.current_time = 1000.0
    tracker.update_service_status(service_name, "healthy", response_time_ms=10)
    # Current time is 1000, uptime_start is 1000. Current session uptime = 0. Total_uptime = 0.
    assert tracker.get_service_uptime(service_name) == 0.0 
    
    mock_time_time.controller.current_time = 1050.0 # Advance time by 50s
    # Current session uptime = 50. Total_uptime = 0.
    assert tracker.get_service_uptime(service_name) == 50.0

    # Case 3: Service was healthy, went unhealthy, now healthy again
    # state after previous: current_status="healthy", uptime_start=1000, total_uptime=0, last_checked=1000 (from update)
    # let's re-update to set last_checked
    tracker.update_service_status(service_name, "healthy", response_time_ms=10) # last_checked=1050
    
    mock_time_time.controller.current_time = 1100.0 # Healthy for 100s (1100-1000)
    tracker.update_service_status(service_name, "unhealthy", response_time_ms=None) # Goes down
    # total_uptime becomes 100. uptime_start is None.
    assert tracker.status_history[service_name]["total_uptime"] == 100.0
    assert tracker.status_history[service_name]["uptime_start"] is None
    assert tracker.get_service_uptime(service_name) == 100.0 # Should return total_uptime

    mock_time_time.controller.current_time = 1150.0 # Unhealthy for 50s
    tracker.update_service_status(service_name, "healthy", response_time_ms=20) # Back to healthy
    # uptime_start is 1150. total_uptime is still 100.
    assert tracker.status_history[service_name]["uptime_start"] == 1150.0
    assert tracker.get_service_uptime(service_name) == 100.0 # Current session uptime is 0, so returns total_uptime

    mock_time_time.controller.current_time = 1200.0 # Healthy for 50s in current session
    # Current session is 50s. total_uptime is 100.
    assert tracker.get_service_uptime(service_name) == 150.0 # 100 (total) + 50 (current session)

    # Case 4: Service is currently unhealthy
    mock_time_time.controller.current_time = 1250.0 # Healthy for another 50s (total 100s in this session)
    tracker.update_service_status(service_name, "unhealthy", response_time_ms=None)
    # total_uptime becomes 100 (previous) + 100 (session just ended) = 200
    # uptime_start becomes None.
    assert tracker.status_history[service_name]["total_uptime"] == 200.0
    assert tracker.get_service_uptime(service_name) == 200.0

    # Case 5: Service started unhealthy, then became healthy
    service_name_2 = "uptime_svc_starts_unhealthy"
    mock_time_time.controller.current_time = 2000.0
    tracker.update_service_status(service_name_2, "unhealthy")
    assert tracker.get_service_uptime(service_name_2) == 0.0
    
    mock_time_time.controller.current_time = 2050.0 # Unhealthy for 50s
    tracker.update_service_status(service_name_2, "healthy", response_time_ms=10) # Becomes healthy
    # uptime_start = 2050, total_uptime = 0
    assert tracker.get_service_uptime(service_name_2) == 0.0 # Current session uptime is 0
    
    mock_time_time.controller.current_time = 2100.0 # Healthy for 50s
    assert tracker.get_service_uptime(service_name_2) == 50.0


# Scenario 5: ServiceTracker.get_average_response_time
def test_get_average_response_time(tracker_with_mocked_save):
    tracker = tracker_with_mocked_save
    service_name = "avg_rt_svc"

    # Case 1: No response times recorded
    assert tracker.get_average_response_time(service_name) == 0.0

    # Case 2: Some response times
    tracker.update_service_status(service_name, "healthy", response_time_ms=100.0)
    tracker.update_service_status(service_name, "healthy", response_time_ms=150.0)
    tracker.update_service_status(service_name, "healthy", response_time_ms=200.0)
    # Avg = (100+150+200)/3 = 150
    assert tracker.get_average_response_time(service_name) == 150.0

    # Case 3: Response time is None (should be ignored)
    tracker.update_service_status(service_name, "unhealthy", response_time_ms=None)
    assert tracker.get_average_response_time(service_name) == 150.0 # Still 150, None ignored

    # Case 4: Max samples reached (uses only current samples)
    # MAX_RESPONSE_TIME_SAMPLES is 10 by default fixture. Let's add more.
    for i in range(12): # Add 12 more, total 3 + 12 = 15. Will keep last 10.
        tracker.update_service_status(service_name, "healthy", response_time_ms=float(i * 10)) # 0, 10, ..., 110
    
    # Expected last 10: 20,30,40,50,60,70,80,90,100,110
    # Sum = (20+110)*10/2 = 130*5 = 650. Avg = 65.0
    # The original [100,150,200] are pushed out. New list [0,10,20,30,40,50,60,70,80,90,100,110] -> last 10 are [20...110]
    # Original samples: [100, 150, 200]
    # New samples: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110
    # After adding 0: [150, 200, 0] (if MAX_SAMPLES=3)
    # After adding 10: [200, 0, 10]
    # After adding 20: [0, 10, 20]
    # ...
    # After adding 110 (last of 12 new): [90, 100, 110] if MAX_SAMPLES=3
    # Avg = (90+100+110)/3 = 300/3 = 100
    # If MAX_SAMPLES is 10 (default from fixture):
    # Original: [100, 150, 200]
    # Adding 0..110 (12 samples).
    # Response times list becomes: [100,150,200,0,10,20,30,40,50,60] (len 10) -> [70,80,90,100,110] (last 5 of 12 new ones)
    # The list becomes [20,30,40,50,60,70,80,90,100,110]
    # Sum = (20+110)*10/2 = 130*5 = 650. Avg = 65.0
    assert tracker.get_average_response_time(service_name) == 65.0


# --- Tests for Individual Check Functions ---

# Scenario: check_redis_status
@pytest.mark.asyncio
@patch('redis.asyncio.Redis', new_callable=AsyncMock) # Mock the Redis class from redis.asyncio
async def test_check_redis_status_healthy(
    MockRedisClass, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    MockRedisClass.from_url.return_value = mock_redis_instance # Mock the class method
    
    # Simulate time passing during the ping
    start_time = mock_time_time.controller.current_time
    async def ping_side_effect():
        mock_time_time.controller.advance(0.05) # 50ms for ping
        return True
    mock_redis_instance.ping.side_effect = ping_side_effect

    await hc_module.check_redis_status(tracker_with_mocked_save)
    
    MockRedisClass.from_url.assert_called_once_with(
        f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}", password=None, decode_responses=True
    )
    mock_redis_instance.ping.assert_called_once()
    
    status_entry = tracker_with_mocked_save.status_history.get("redis")
    assert status_entry is not None
    assert status_entry["current_status"] == "healthy"
    assert status_entry["message"] == "Ping successful."
    assert status_entry["response_times"][-1] == pytest.approx(50.0) # 50ms


@pytest.mark.asyncio
@patch('redis.asyncio.Redis', new_callable=AsyncMock)
async def test_check_redis_status_unhealthy_ping_false(
    MockRedisClass, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=False) # Ping returns False
    MockRedisClass.from_url.return_value = mock_redis_instance
    
    await hc_module.check_redis_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("redis")
    assert status_entry["current_status"] == "unhealthy"
    assert "Ping failed." in status_entry["message"]


@pytest.mark.asyncio
@patch('redis.asyncio.Redis', new_callable=AsyncMock)
async def test_check_redis_status_unhealthy_exception(
    MockRedisClass, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    redis_conn_error = ConnectionRefusedError("Redis connection refused")
    MockRedisClass.from_url.side_effect = redis_conn_error # Connection itself fails
    
    await hc_module.check_redis_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("redis")
    assert status_entry["current_status"] == "unhealthy"
    assert f"Connection error: {redis_conn_error}" in status_entry["message"]


def test_check_redis_status_disabled(
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.redis, 'ENABLED', False)
    # This function is async, but if it returns early due to disabled, it might not need full async test
    # However, to be safe and consistent with its signature:
    
    async def run_check_redis_disabled():
        await hc_module.check_redis_status(tracker_with_mocked_save)

    asyncio.run(run_check_redis_disabled()) # Run the async function

    status_entry = tracker_with_mocked_save.status_history.get("redis")
    assert status_entry["current_status"] == "disabled"
    assert status_entry["message"] == "Redis is disabled in configuration."


# --- Tests for check_elevenlabs_status ---
@pytest.mark.asyncio
@patch('requests.get')
async def test_check_elevenlabs_status_healthy(
    mock_requests_get_el, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"subscription": {"status": "active"}}
    mock_requests_get_el.return_value = mock_response
    
    start_time = mock_time_time.controller.current_time
    async def get_side_effect_el(*args, **kwargs):
        mock_time_time.controller.advance(0.12) # 120ms
        return mock_response
    mock_requests_get_el.side_effect = get_side_effect_el

    await hc_module.check_elevenlabs_status(tracker_with_mocked_save)

    mock_requests_get_el.assert_called_once_with(
        app_config.elevenlabs.API_ENDPOINT_USER_INFO,
        headers={"xi-api-key": app_config.elevenlabs.API_KEY},
        timeout=app_config.httpOptions.TIMEOUT
    )
    status_entry = tracker_with_mocked_save.status_history.get("elevenlabs")
    assert status_entry["current_status"] == "healthy"
    assert status_entry["message"] == "API key is valid. Subscription status: active."
    assert status_entry["response_times"][-1] == pytest.approx(120.0)


@pytest.mark.asyncio
@patch('requests.get')
async def test_check_elevenlabs_status_unhealthy_api_error(
    mock_requests_get_el, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 401 # Unauthorized
    mock_response.reason = "Unauthorized"
    mock_response.text = "Invalid API key provided."
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error", response=mock_response)
    mock_requests_get_el.return_value = mock_response # This mock will be used by raise_for_status

    await hc_module.check_elevenlabs_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("elevenlabs")
    assert status_entry["current_status"] == "unhealthy"
    assert "API error: 401 Unauthorized. Detail: Invalid API key provided." in status_entry["message"]


@pytest.mark.asyncio
@patch('requests.get', side_effect=requests.exceptions.Timeout("ElevenLabs timeout"))
async def test_check_elevenlabs_status_unhealthy_timeout(
    mock_requests_get_el_timeout, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    await hc_module.check_elevenlabs_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("elevenlabs")
    assert status_entry["current_status"] == "unhealthy"
    assert "Request failed: ElevenLabs timeout" in status_entry["message"]


def test_check_elevenlabs_status_disabled(
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.elevenlabs, 'ENABLED', False)
    async def run_check_el_disabled():
        await hc_module.check_elevenlabs_status(tracker_with_mocked_save)
    asyncio.run(run_check_el_disabled())
    
    status_entry = tracker_with_mocked_save.status_history.get("elevenlabs")
    assert status_entry["current_status"] == "disabled"
    assert status_entry["message"] == "ElevenLabs is disabled in configuration."


def test_check_elevenlabs_status_no_api_key(
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.elevenlabs, 'API_KEY', None) # API Key not set
    async def run_check_el_no_key():
        await hc_module.check_elevenlabs_status(tracker_with_mocked_save)
    asyncio.run(run_check_el_no_key())

    status_entry = tracker_with_mocked_save.status_history.get("elevenlabs")
    assert status_entry["current_status"] == "unhealthy" # Or "degraded" depending on desired strictness
    assert status_entry["message"] == "ElevenLabs API Key not configured."

# --- Tests for check_llm_status ---
@pytest.mark.asyncio
@patch('requests.get') # Assuming LLM status check uses GET, adjust if POST
async def test_check_llm_status_healthy(
    mock_requests_get_llm, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults
):
    # Mock LLM API response for status check
    # This depends on the actual LLM API structure. For OpenAI-like, it might not have a dedicated status endpoint.
    # The code uses app_config.llm.ENDPOINT, which is for chat/completions.
    # Let's assume a successful call to the main ENDPOINT (e.g., a lightweight query or if it has a status path)
    # For now, assume a simple 200 OK means healthy if no specific status path is used.
    # If it's OpenAI, there's no standard status endpoint. The code might just try a minimal request.
    # The current code in `check_llm_status` uses `requests.get(app_config.llm.ENDPOINT + "/models")` for OpenAI.
    # Let's adjust the mock_appconfig_health_check_defaults if needed or use a specific endpoint for test.
    
    monkeypatch = pytest.MonkeyPatch() # To adjust config for this test if needed
    llm_models_endpoint = app_config.llm.ENDPOINT + "/models" # Assuming this is what's called
    monkeypatch.setattr(app_config.llm, 'API_KEY', "llm_test_key_for_hc")


    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "some-model", "object": "model"}]} # Example OpenAI /models response
    mock_requests_get_llm.return_value = mock_response
    
    start_time = mock_time_time.controller.current_time
    async def get_side_effect_llm(*args, **kwargs):
        mock_time_time.controller.advance(0.25) # 250ms
        return mock_response
    mock_requests_get_llm.side_effect = get_side_effect_llm

    await hc_module.check_llm_status(tracker_with_mocked_save)

    expected_headers = {"Authorization": f"Bearer {app_config.llm.API_KEY}"} if app_config.llm.API_KEY else {}
    mock_requests_get_llm.assert_called_once_with(
        llm_models_endpoint, headers=expected_headers, timeout=app_config.httpOptions.TIMEOUT
    )
    status_entry = tracker_with_mocked_save.status_history.get("llm")
    assert status_entry["current_status"] == "healthy"
    assert "LLM API is responsive. Models found: 1" in status_entry["message"]
    assert status_entry["response_times"][-1] == pytest.approx(250.0)


@pytest.mark.asyncio
@patch('requests.get')
async def test_check_llm_status_unhealthy_api_error(
    mock_requests_get_llm, tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.llm, 'API_KEY', "llm_test_key_for_hc_err")
    llm_models_endpoint = app_config.llm.ENDPOINT + "/models"

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 503 # Service Unavailable
    mock_response.reason = "Service Unavailable"
    mock_response.text = "LLM service is down for maintenance."
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("LLM Error", response=mock_response)
    mock_requests_get_llm.return_value = mock_response

    await hc_module.check_llm_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("llm")
    assert status_entry["current_status"] == "unhealthy"
    assert "LLM API error: 503 Service Unavailable. Detail: LLM service is down for maintenance." in status_entry["message"]


def test_check_llm_status_disabled(
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.llm, 'ENABLED', False)
    async def run_check_llm_disabled():
        await hc_module.check_llm_status(tracker_with_mocked_save)
    asyncio.run(run_check_llm_disabled())
    
    status_entry = tracker_with_mocked_save.status_history.get("llm")
    assert status_entry["current_status"] == "disabled"
    assert status_entry["message"] == "LLM service is disabled in configuration."


# --- Tests for check_disk_status ---
# Note: shutil.disk_usage might not be available on all test platforms (e.g., some CI without full disk utils)
# We will mock it thoroughly.

@patch('shutil.disk_usage')
@patch('os.path.exists')
@patch('os.makedirs') # For the case where LOCAL_STORAGE_PATH might not exist
@patch('asyncio.to_thread', side_effect=lambda func, *args: asyncio.coroutine(func)(*args)) # Simulate to_thread
async def test_check_disk_status_healthy(
    mock_async_to_thread, mock_os_makedirs, mock_os_exists, mock_shutil_disk_usage,
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults, monkeypatch
):
    storage_path = app_config.storage.LOCAL_STORAGE_PATH
    mock_os_exists.return_value = True # Assume path exists

    # Simulate disk usage: total = 100GB, used = 50GB, free = 50GB
    # shutil.disk_usage returns a tuple (total, used, free) in bytes
    total_bytes = 100 * (1024**3)
    free_bytes = 50 * (1024**3)
    used_bytes = total_bytes - free_bytes
    mock_shutil_disk_usage.return_value = (total_bytes, used_bytes, free_bytes)
    
    await hc_module.check_disk_status(tracker_with_mocked_save)

    mock_shutil_disk_usage.assert_called_once_with(storage_path)
    status_entry = tracker_with_mocked_save.status_history.get("disk_space")
    assert status_entry["current_status"] == "healthy"
    assert "Available disk space is sufficient." in status_entry["message"]
    assert status_entry["details"]["path_checked"] == storage_path
    assert status_entry["details"]["free_gb"] == pytest.approx(50.0)


@pytest.mark.parametrize("free_gb_mock, expected_status, expected_level", [
    (app_config.health_check.MIN_DISK_SPACE_GB_WARNING - 1, "degraded", "WARNING"), # Warning level
    (app_config.health_check.MIN_DISK_SPACE_GB_CRITICAL - 1, "unhealthy", "CRITICAL"), # Critical level
    (app_config.health_check.MIN_DISK_SPACE_GB_CRITICAL, "degraded", "WARNING"), # At critical threshold, still warning (as >= critical is fine)
                                                                                # Code says: free_gb < CRITICAL -> unhealthy. free_gb < WARNING -> degraded
                                                                                # So, if free_gb = CRITICAL, it's not < CRITICAL, so not unhealthy.
                                                                                # If free_gb = CRITICAL and CRITICAL < WARNING, then it's also < WARNING, so degraded.
                                                                                # Let's adjust test values based on app_config fixture for clarity.
])
@patch('shutil.disk_usage')
@patch('os.path.exists', return_value=True)
@patch('asyncio.to_thread', side_effect=lambda func, *args: asyncio.coroutine(func)(*args))
async def test_check_disk_status_warning_critical(
    mock_async_to_thread, mock_os_exists, mock_shutil_disk_usage,
    tracker_with_mocked_save, mock_time_time, mock_health_check_logger, 
    mock_appconfig_health_check_defaults, monkeypatch,
    free_gb_mock, expected_status, expected_level # From parametrize
):
    storage_path = app_config.storage.LOCAL_STORAGE_PATH
    warning_gb = app_config.health_check.MIN_DISK_SPACE_GB_WARNING # e.g. 10
    critical_gb = app_config.health_check.MIN_DISK_SPACE_GB_CRITICAL # e.g. 2

    # Adjust actual free_gb based on parameterization
    if free_gb_mock == warning_gb -1: # e.g. 9GB
        actual_free_bytes = (warning_gb - 1) * (1024**3)
    elif free_gb_mock == critical_gb -1: # e.g. 1GB
        actual_free_bytes = (critical_gb -1) * (1024**3)
    elif free_gb_mock == critical_gb : # e.g. 2GB
         actual_free_bytes = critical_gb * (1024**3)


    total_bytes = 100 * (1024**3) # Assume 100GB total for simplicity
    used_bytes = total_bytes - actual_free_bytes
    mock_shutil_disk_usage.return_value = (total_bytes, used_bytes, actual_free_bytes)
    
    await hc_module.check_disk_status(tracker_with_mocked_save)

    status_entry = tracker_with_mocked_save.status_history.get("disk_space")
    assert status_entry["current_status"] == expected_status
    assert f"Disk space is low ({expected_level})" in status_entry["message"]
    assert status_entry["details"]["free_gb"] == pytest.approx(actual_free_bytes / (1024**3))


def test_check_disk_status_provider_not_local(
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "s3") # Not local
    
    async def run_check_disk_not_local():
        await hc_module.check_disk_status(tracker_with_mocked_save)
    asyncio.run(run_check_disk_not_local())

    status_entry = tracker_with_mocked_save.status_history.get("disk_space")
    assert status_entry["current_status"] == "not_applicable"
    assert status_entry["message"] == "Disk space check not applicable for non-local storage provider (s3)."


@patch('os.path.exists', return_value=False)
@patch('os.makedirs', side_effect=OSError("Cannot create dir"))
@patch('asyncio.to_thread', side_effect=lambda func, *args: asyncio.coroutine(func)(*args))
async def test_check_disk_status_path_creation_fails(
    mock_async_to_thread, mock_os_makedirs, mock_os_exists,
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    storage_path = app_config.storage.LOCAL_STORAGE_PATH
    await hc_module.check_disk_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("disk_space")
    assert status_entry["current_status"] == "unhealthy"
    assert f"Storage path {storage_path} does not exist and could not be created." in status_entry["message"]


@patch('shutil.disk_usage', side_effect=Exception("shutil error"))
@patch('os.path.exists', return_value=True)
@patch('asyncio.to_thread', side_effect=lambda func, *args: asyncio.coroutine(func)(*args))
async def test_check_disk_status_shutil_exception(
    mock_async_to_thread, mock_os_exists, mock_shutil_disk_usage_exc,
    tracker_with_mocked_save, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    await hc_module.check_disk_status(tracker_with_mocked_save)
    
    status_entry = tracker_with_mocked_save.status_history.get("disk_space")
    assert status_entry["current_status"] == "unknown"
    assert "Error checking disk space: shutil error" in status_entry["message"]

# --- Tests for get_service_status (aggregator) ---

@pytest.fixture
def mock_individual_check_functions(monkeypatch):
    """Mocks all individual check_..._status functions."""
    mocks = {
        'redis': AsyncMock(),
        'elevenlabs': AsyncMock(),
        'llm': AsyncMock(),
        'disk_space': AsyncMock() # Assuming check_disk_status is also async or wrapped
    }
    monkeypatch.setattr(hc_module, 'check_redis_status', mocks['redis'])
    monkeypatch.setattr(hc_module, 'check_elevenlabs_status', mocks['elevenlabs'])
    monkeypatch.setattr(hc_module, 'check_llm_status', mocks['llm'])
    monkeypatch.setattr(hc_module, 'check_disk_status', mocks['disk_space'])
    return mocks

@pytest.mark.asyncio
async def test_get_service_status_aggregator_all_healthy(
    mock_individual_check_functions, tracker_with_mocked_save, # Use the tracker with mocked save
    mock_time_time, mock_health_check_logger, mock_appconfig_health_check_defaults
):
    # Configure individual checks to do nothing (they will update the tracker passed to them)
    # The tracker will then be queried. We need to simulate the tracker's state *after* checks.
    
    # Instead of mocking the checks to do nothing, we'll mock the tracker's get_service_status_detail
    # method, which is what the aggregator uses to build its final list.
    # Or, more simply, pre-populate the tracker's history as if checks ran.
    
    current_time_val = 1000.0
    mock_time_time.controller.current_time = current_time_val
    
    # Pre-populate tracker history as if all checks ran and were healthy
    tracker_with_mocked_save.status_history = {
        "redis": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val, "response_times": [10], "total_uptime": 3600, "uptime_start": current_time_val - 3600, "last_status_change": current_time_val - 3600},
        "elevenlabs": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val, "response_times": [100], "total_uptime": 7200, "uptime_start": current_time_val - 7200, "last_status_change": current_time_val - 7200},
        "llm": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val, "response_times": [200], "total_uptime": 10800, "uptime_start": current_time_val - 10800, "last_status_change": current_time_val - 10800},
        "disk_space": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val, "details": {"free_gb": 50}, "total_uptime": 86400, "uptime_start": current_time_val - 86400, "last_status_change": current_time_val - 86400},
    }
    # Ensure the global tracker instance used by get_service_status is our mocked one
    with patch('viralStoryGenerator.utils.health_check.tracker_instance', tracker_with_mocked_save):
        # Set API start time for uptime calculation
        hc_module.api_start_time = current_time_val - (24 * 3600) # API up for 1 day

        service_status_response = await hc_module.get_service_status()

    assert service_status_response.overall_status == "healthy"
    assert service_status_response.version == app_config.VERSION
    assert service_status_response.environment == app_config.ENVIRONMENT
    assert service_status_response.api_uptime == pytest.approx(24 * 3600) # 1 day in seconds
    
    assert len(service_status_response.services) == 4 # Based on pre-populated history
    for service_detail in service_status_response.services:
        assert service_detail.status == "healthy"
    
    # Individual check functions should have been called
    for check_mock in mock_individual_check_functions.values():
        check_mock.assert_called_once_with(tracker_with_mocked_save)


@pytest.mark.asyncio
@pytest.mark.parametrize("unhealthy_service, degraded_service, expected_overall_status", [
    ("redis", None, "unhealthy"),             # One unhealthy -> overall unhealthy
    (None, "llm", "degraded"),                # One degraded -> overall degraded
    ("redis", "llm", "unhealthy"),            # Unhealthy overrides degraded
    ("disk_space", "elevenlabs", "unhealthy"),# Another unhealthy example
])
async def test_get_service_status_aggregator_mixed_statuses(
    mock_individual_check_functions, tracker_with_mocked_save, mock_time_time,
    unhealthy_service, degraded_service, expected_overall_status, 
    mock_appconfig_health_check_defaults # Ensure app_config is available for version/env
):
    current_time_val = 2000.0
    mock_time_time.controller.current_time = current_time_val
    
    # Base healthy statuses
    tracker_with_mocked_save.status_history = {
        s: {"current_status": "healthy", "message": "OK", "last_checked": current_time_val} 
        for s in ["redis", "elevenlabs", "llm", "disk_space"]
    }
    if unhealthy_service:
        tracker_with_mocked_save.status_history[unhealthy_service]["current_status"] = "unhealthy"
        tracker_with_mocked_save.status_history[unhealthy_service]["message"] = "Service is down"
    if degraded_service:
        tracker_with_mocked_save.status_history[degraded_service]["current_status"] = "degraded"
        tracker_with_mocked_save.status_history[degraded_service]["message"] = "Performance issue"

    with patch('viralStoryGenerator.utils.health_check.tracker_instance', tracker_with_mocked_save):
        hc_module.api_start_time = current_time_val - 1000 # API up for 1000s
        service_status_response = await hc_module.get_service_status()

    assert service_status_response.overall_status == expected_overall_status
    for service_detail in service_status_response.services:
        if service_detail.service_name == unhealthy_service:
            assert service_detail.status == "unhealthy"
        elif service_detail.service_name == degraded_service:
            assert service_detail.status == "degraded"
        else:
            assert service_detail.status == "healthy"

    for check_mock in mock_individual_check_functions.values():
        check_mock.assert_called_once_with(tracker_with_mocked_save)


@pytest.mark.asyncio
async def test_get_service_status_aggregator_unknown_overrides_degraded(
    mock_individual_check_functions, tracker_with_mocked_save, mock_time_time,
    mock_appconfig_health_check_defaults
):
    current_time_val = 3000.0
    mock_time_time.controller.current_time = current_time_val
    
    tracker_with_mocked_save.status_history = {
        "redis": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val},
        "elevenlabs": {"current_status": "degraded", "message": "Slow response", "last_checked": current_time_val},
        "llm": {"current_status": "unknown", "message": "Status check failed", "last_checked": current_time_val}, # Unknown
        "disk_space": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val},
    }
    with patch('viralStoryGenerator.utils.health_check.tracker_instance', tracker_with_mocked_save):
        service_status_response = await hc_module.get_service_status()

    # "unknown" is treated as more severe than "degraded" for overall status if no "unhealthy"
    assert service_status_response.overall_status == "unhealthy" 


@pytest.mark.asyncio
async def test_get_service_status_aggregator_disabled_services_not_unhealthy(
    mock_individual_check_functions, tracker_with_mocked_save, mock_time_time,
    mock_appconfig_health_check_defaults
):
    current_time_val = 4000.0
    mock_time_time.controller.current_time = current_time_val
    
    tracker_with_mocked_save.status_history = {
        "redis": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val},
        "elevenlabs": {"current_status": "disabled", "message": "Disabled by config", "last_checked": current_time_val}, # Disabled
        "llm": {"current_status": "healthy", "message": "OK", "last_checked": current_time_val},
        "disk_space": {"current_status": "not_applicable", "message": "S3 storage", "last_checked": current_time_val}, # N/A
    }
    with patch('viralStoryGenerator.utils.health_check.tracker_instance', tracker_with_mocked_save):
        service_status_response = await hc_module.get_service_status()

    # "disabled" and "not_applicable" should not make overall status unhealthy/degraded
    assert service_status_response.overall_status == "healthy"
    
    found_disabled = False
    found_not_applicable = False
    for service in service_status_response.services:
        if service.service_name == "elevenlabs" and service.status == "disabled":
            found_disabled = True
        if service.service_name == "disk_space" and service.status == "not_applicable":
            found_not_applicable = True
    assert found_disabled
    assert found_not_applicable
