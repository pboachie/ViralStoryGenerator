import pytest
import time
import datetime
import threading
from unittest.mock import patch, MagicMock, call

# Assuming the module is viralStoryGenerator.utils.scheduled_cleanup
from viralStoryGenerator.utils import scheduled_cleanup as sc_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_cleanup_defaults(monkeypatch):
    """Set default app_config values for scheduled_cleanup tests."""
    monkeypatch.setattr(app_config.storage, 'CLEANUP_INTERVAL_HOURS', 24)
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 7)
    # Base dir is needed by storage_manager.cleanup_old_files, but that will be mocked.
    # However, if ScheduledCleanupTask itself reads it for some reason:
    monkeypatch.setattr(app_config.storage, 'BASE_DIR', "/mock/base/storage")


@pytest.fixture
def mock_cleanup_logger():
    """Fixture to mock the _logger in scheduled_cleanup.py."""
    with patch('viralStoryGenerator.utils.scheduled_cleanup._logger') as mock_logger:
        yield mock_logger

@pytest.fixture
def mock_schedule_lib():
    """Fixture to mock the entire schedule library's relevant functions."""
    with patch('schedule.every') as mock_every, \
         patch('schedule.run_pending') as mock_run_pending, \
         patch('schedule.idle_seconds') as mock_idle_seconds, \
         patch('schedule.next_run') as mock_next_run, \
         patch('schedule.clear') as mock_clear:
        
        # Configure mock_every to return a chainable mock
        mock_hour_job = MagicMock()
        mock_hours_obj = MagicMock()
        mock_hours_obj.do.return_value = mock_hour_job
        mock_every_interval_obj = MagicMock()
        mock_every_interval_obj.hours = mock_hours_obj
        mock_every_interval_obj.hour = mock_hours_obj # Handle singular too
        
        # If schedule.every(X) is called, it returns an object that then has .hours, .minutes etc.
        # We need every(X) to return something that has .hours
        # And that .hours needs to have .do
        # A bit complex to mock the full chain. Let's simplify:
        # mock_every().hours.do()
        
        mock_do_job = MagicMock()
        mock_every_chain = MagicMock()
        mock_every_chain.hours.do.return_value = mock_do_job
        mock_every.return_value = mock_every_chain
        
        yield {
            'every': mock_every,
            'run_pending': mock_run_pending,
            'idle_seconds': mock_idle_seconds,
            'next_run': mock_next_run,
            'clear': mock_clear,
            '_mock_do_job': mock_do_job # To check job registration
        }
        # schedule.clear() is called by the fixture itself if we want to ensure clean state for each test
        # For now, tests will call it if they modify schedule's state.

@pytest.fixture
def mock_threading_thread():
    """Fixture to mock threading.Thread."""
    with patch('threading.Thread') as mock_thread_class:
        mock_thread_instance = MagicMock(spec=threading.Thread)
        mock_thread_class.return_value = mock_thread_instance
        yield mock_thread_class, mock_thread_instance # Return class and instance

@pytest.fixture
def mock_storage_manager_cleanup():
    """Fixture to mock storage_manager.cleanup_old_files."""
    with patch('viralStoryGenerator.utils.storage_manager.cleanup_old_files') as mock_cleanup:
        yield mock_cleanup

@pytest.fixture
def mock_time_control():
    """Fixture to mock time.time and time.sleep, and datetime.datetime.now."""
    # Using a class to manage current_time for time.time and advance it
    class TimeController:
        def __init__(self, start_time=1000.0):
            self.current_time = start_time
            self._actual_sleep = time.sleep # Keep a reference to real sleep if needed

        def time(self):
            return self.current_time
        
        def sleep(self, seconds): # Mock for time.sleep
            self.current_time += seconds
            # self._actual_sleep(0.00001) # Tiny sleep to allow context switches if testing threads intensely
            return None

        def datetime_now(self, tz=None): # Mock for datetime.datetime.now
            return datetime.datetime.fromtimestamp(self.current_time, tz=tz)

        def advance(self, seconds):
            self.current_time += seconds

    controller = TimeController()
    
    with patch('time.time', side_effect=controller.time) as mock_time_dot_time, \
         patch('time.sleep', side_effect=controller.sleep) as mock_time_dot_sleep, \
         patch('datetime.datetime', MagicMock()) as mock_datetime_class:
        
        mock_datetime_class.now.side_effect = controller.datetime_now
        # Attach controller to one of the mocks for access in tests
        mock_time_dot_time.controller = controller
        yield mock_time_dot_time, mock_time_dot_sleep, mock_datetime_class


# --- Tests for ScheduledCleanupTask Initialization (Scenario 1) ---

def test_scheduled_cleanup_task_init_retention_enabled(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_schedule_lib
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 7)
    
    task = sc_module.ScheduledCleanupTask()
    
    assert task.retention_days == 7
    assert task.cleanup_interval_hours == 24 # From fixture
    assert not task.is_running()
    assert task._stop_event.is_set() is False # Stop event should be clear initially
    mock_cleanup_logger.info.assert_any_call(
        f"ScheduledCleanupTask initialized. Retention: 7 days, Interval: 24 hours. Task not started."
    )
    mock_schedule_lib['clear'].assert_called_once() # Ensure schedule is cleared on init


def test_scheduled_cleanup_task_init_retention_disabled_negative_days(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_schedule_lib
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', -1) # Disabled
    
    task = sc_module.ScheduledCleanupTask()
    
    assert task.retention_days == -1
    assert task._retention_enabled is False
    mock_cleanup_logger.info.assert_any_call(
        "File retention is disabled (FILE_RETENTION_DAYS <= 0). Cleanup task will not run."
    )
    mock_schedule_lib['clear'].assert_called_once()


def test_scheduled_cleanup_task_init_retention_disabled_zero_days(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_schedule_lib
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 0) # Disabled
    
    task = sc_module.ScheduledCleanupTask()
    
    assert task.retention_days == 0
    assert task._retention_enabled is False
    mock_cleanup_logger.info.assert_any_call(
        "File retention is disabled (FILE_RETENTION_DAYS <= 0). Cleanup task will not run."
    )

# --- Tests for ScheduledCleanupTask.start() (Scenario 2) ---

def test_start_task_retention_disabled(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_schedule_lib, mock_threading_thread
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 0) # Retention disabled
    task = sc_module.ScheduledCleanupTask()
    
    task.start()
    
    assert not task.is_running()
    mock_threading_thread[0].assert_not_called() # Thread class not instantiated
    mock_cleanup_logger.info.assert_any_call(
        "Cleanup task not started because file retention is disabled."
    )


@patch.object(sc_module.ScheduledCleanupTask, '_run_cleanup_job') # Mock the method on the class
def test_start_task_retention_enabled_not_running(
    mock_run_cleanup_job_method, mock_cleanup_logger, mock_appconfig_cleanup_defaults, 
    monkeypatch, mock_schedule_lib, mock_threading_thread
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 5) # Retention enabled
    task = sc_module.ScheduledCleanupTask()
    
    # Ensure _scheduler_loop is not running if it was by chance from another test context (though fixtures should handle)
    task._thread = None 
    task._stop_event.set() # Ensure it's initially "not running" in a way start() would proceed
    task._stop_event.clear() # Clear it for the start() call

    # Mock the thread instance that will be created
    mock_thread_instance = mock_threading_thread[1]

    task.start()

    assert task.is_running()
    
    # Check threading.Thread was called to create the thread
    mock_threading_thread[0].assert_called_once_with(target=task._scheduler_loop, daemon=True)
    mock_thread_instance.start.assert_called_once() # Thread was started
    
    # Check _run_cleanup_job was called immediately
    mock_run_cleanup_job_method.assert_called_once()
    
    # Check schedule.every().hours.do()
    mock_schedule_lib['every'].assert_called_once_with(task.cleanup_interval_hours)
    # The chain `every(X).hours.do(job)` needs to be asserted.
    # mock_schedule_lib['every'] returns a mock that has a `.hours` attribute (MagicMock)
    # that `.hours` attribute has a `.do` method (MagicMock)
    mock_schedule_lib['every'].return_value.hours.do.assert_called_once_with(task._run_cleanup_job)
    
    mock_cleanup_logger.info.assert_any_call(f"Scheduled cleanup task started. Interval: {task.cleanup_interval_hours} hours.")

    # Cleanup: stop the task to allow thread to exit if it actually started (though it's mocked)
    task.stop() # This will set the event and join the mocked thread


def test_start_task_already_running(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_threading_thread
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 3)
    task = sc_module.ScheduledCleanupTask()
    
    # Simulate task is already running
    task._thread = MagicMock(spec=threading.Thread) # A mock thread
    task._thread.is_alive.return_value = True # Thread is alive
    # _stop_event is clear when running
    task._stop_event.clear() 

    # Store original thread mock calls to ensure no new thread is started
    original_thread_class_call_count = mock_threading_thread[0].call_count
    original_thread_instance_start_call_count = mock_threading_thread[1].start.call_count

    task.start() # Attempt to start again

    assert task.is_running() # Should still be considered running with the original thread
    assert mock_threading_thread[0].call_count == original_thread_class_call_count # No new Thread()
    assert mock_threading_thread[1].start.call_count == original_thread_instance_start_call_count # No new start()
    
    mock_cleanup_logger.warning.assert_called_once_with(
        "Cleanup task is already running or start was called without a full stop."
    )

# --- Tests for ScheduledCleanupTask.stop() (Scenario 3) ---

def test_stop_task_when_running(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_schedule_lib, mock_threading_thread
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 7) # Ensure retention is enabled
    task = sc_module.ScheduledCleanupTask()

    # Simulate task is running
    mock_thread_instance = mock_threading_thread[1] # Get the instance from the fixture
    task._thread = mock_thread_instance
    task._thread.is_alive.return_value = True # Simulate thread is alive
    task._stop_event.clear() # Running means event is clear

    task.stop()

    assert not task.is_running() # is_running should be False
    assert task._stop_event.is_set() # Stop event should be set
    mock_thread_instance.join.assert_called_once() # Thread should be joined
    mock_schedule_lib['clear'].assert_called_once() # Schedule should be cleared
    mock_cleanup_logger.info.assert_any_call("Scheduled cleanup task stopped.")


def test_stop_task_when_not_running(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, mock_schedule_lib, mock_threading_thread
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 7)
    task = sc_module.ScheduledCleanupTask()
    
    # Ensure task is not running (initial state after init, or explicitly set)
    task._thread = None
    task._stop_event.set() # Stop event is set when not running typically

    # Store original call counts for join and clear to ensure they are not called
    original_join_call_count = 0 # Assuming thread is None, join won't be on our mock_thread_instance
    if mock_threading_thread[1].join.called: # if some other test left it called
        original_join_call_count = mock_threading_thread[1].join.call_count
        
    original_clear_call_count = mock_schedule_lib['clear'].call_count


    task.stop()

    assert not task.is_running()
    # If thread was None, join shouldn't be called on the mock_thread_instance from fixture
    # If it was some other thread object, we can't easily check.
    # For this test, _thread is None, so join is definitely not called.
    assert mock_threading_thread[1].join.call_count == original_join_call_count
    
    # schedule.clear() might be called even if not running, to be safe.
    # Based on current code: it only clears if self._thread is not None.
    # If _thread is None, schedule.clear() is not called.
    assert mock_schedule_lib['clear'].call_count == original_clear_call_count
    
    # Check for a specific log, or no log if that's the behavior
    # Current code logs "Cleanup task is not running."
    mock_cleanup_logger.info.assert_any_call("Cleanup task is not running.")

# --- Tests for ScheduledCleanupTask._run_cleanup_job() (Scenario 4) ---

@patch('viralStoryGenerator.utils.storage_manager.cleanup_old_files')
def test_run_cleanup_job_successful(
    mock_cleanup_old_files, mock_cleanup_logger, mock_appconfig_cleanup_defaults, 
    monkeypatch, mock_time_control # mock_time_control for time.time
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 3)
    task = sc_module.ScheduledCleanupTask()
    
    mock_cleanup_old_files.return_value = 15 # 15 files deleted
    
    # Simulate calling the job (it's usually scheduled, but we call it directly)
    # _run_cleanup_job is a protected method.
    # We can call it directly for testing.
    
    start_time = mock_time_control[0].controller.current_time # time.time()
    task._run_cleanup_job()
    end_time = mock_time_control[0].controller.current_time # time.time() after potential sleep in job
                                                            # cleanup_old_files is sync, so time advance is minimal
                                                            # unless time is advanced *inside* the mocked function.

    mock_cleanup_old_files.assert_called_once_with(task.retention_days)
    assert task.last_run_time is not None
    # last_run_time is set to datetime.now().timestamp() at the end of the job.
    # So, it should be very close to `end_time`.
    assert task.last_run_time == pytest.approx(end_time) 
    
    assert task.last_deleted_count == 15
    assert task.total_deleted_count == 15 # First run
    
    mock_cleanup_logger.info.assert_any_call(
        f"Cleanup job completed. Deleted {15} files. Next run scheduled based on interval."
    )

    # Run again to check total_deleted_count
    mock_cleanup_old_files.return_value = 5
    mock_time_control[0].controller.advance(100) # Advance time
    next_run_start_time = mock_time_control[0].controller.current_time
    task._run_cleanup_job()
    next_run_end_time = mock_time_control[0].controller.current_time

    assert task.last_deleted_count == 5
    assert task.total_deleted_count == 20 # 15 (previous) + 5 (current)
    assert task.last_run_time == pytest.approx(next_run_end_time)


@patch('viralStoryGenerator.utils.storage_manager.cleanup_old_files')
def test_run_cleanup_job_cleanup_raises_exception(
    mock_cleanup_old_files, mock_cleanup_logger, mock_appconfig_cleanup_defaults, 
    monkeypatch, mock_time_control
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 2)
    task = sc_module.ScheduledCleanupTask()

    cleanup_exception = Exception("Disk read error during cleanup")
    mock_cleanup_old_files.side_effect = cleanup_exception
    
    initial_total_deleted_count = task.total_deleted_count # Should be 0
    
    start_time = mock_time_control[0].controller.current_time
    task._run_cleanup_job()
    end_time_after_exception = mock_time_control[0].controller.current_time

    mock_cleanup_old_files.assert_called_once_with(task.retention_days)
    
    # last_run_time should still be updated to show an attempt was made
    assert task.last_run_time == pytest.approx(end_time_after_exception)
    
    # Counts should not change, or last_deleted_count might be 0 or None
    assert task.last_deleted_count == 0 # Set to 0 on failure
    assert task.total_deleted_count == initial_total_deleted_count # Not incremented
    
    mock_cleanup_logger.error.assert_called_once_with(
        f"Error during scheduled cleanup job: {cleanup_exception}", exc_info=True
    )

# --- Tests for ScheduledCleanupTask._scheduler_loop() & get_status() (Scenario 5 & 6) ---

@patch.object(sc_module.ScheduledCleanupTask, '_run_cleanup_job') # Mock the job itself
def test_scheduler_loop_runs_pending_and_stops(
    mock_run_job_method, mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, 
    mock_schedule_lib, mock_threading_thread, mock_time_control
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 1)
    task = sc_module.ScheduledCleanupTask()

    # Mock schedule.idle_seconds() to control loop iterations
    # Let it run a few times then indicate no more jobs soon, then stop_event is set.
    idle_seconds_side_effects = [0.01, 0.01, None] # Loop twice, then idle_seconds returns None (stops loop)
    mock_schedule_lib['idle_seconds'].side_effect = idle_seconds_side_effects
    
    # Mock schedule.run_pending()
    # mock_schedule_lib['run_pending'] is already a MagicMock

    # Mock thread instance for is_alive checks
    mock_thread_instance = mock_threading_thread[1]
    
    # Start the task (this also calls _run_cleanup_job once immediately)
    task.start() 
    
    # Check that _run_cleanup_job was called by start()
    mock_run_job_method.assert_called_once() 
    mock_run_job_method.reset_mock() # Reset for checking calls from scheduler loop

    # Simulate the scheduler loop running a few times
    # The loop itself is in task._scheduler_loop, run by the thread.
    # We need to control the thread's execution or mock what it calls.
    # The actual test of the loop is tricky without real threading or more complex mocks.
    
    # Instead of directly testing the loop, we verify its effects, which are
    # calls to schedule.run_pending() and respecting _stop_event.
    
    # Let's simulate the thread's target function (_scheduler_loop) being called
    # and see if it calls run_pending and respects stop_event.
    # This means we don't call task.start() which creates a real thread.
    # We call _scheduler_loop directly.
    
    task_for_direct_loop_test = sc_module.ScheduledCleanupTask()
    # It needs to be "running" for _scheduler_loop to do its work
    task_for_direct_loop_test._stop_event.clear()
    
    # Re-configure side effects for this direct test
    mock_schedule_lib['idle_seconds'].reset_mock()
    idle_call_count = 0
    def dynamic_idle_seconds():
        nonlocal idle_call_count
        idle_call_count +=1
        if idle_call_count <= 2: # Run pending twice
            return 0.001 # Indicate pending job, loop quickly
        task_for_direct_loop_test._stop_event.set() # After 2 runs, signal stop
        return 10 # Indicate no jobs soon, loop would sleep
    mock_schedule_lib['idle_seconds'].side_effect = dynamic_idle_seconds
    mock_schedule_lib['run_pending'].reset_mock()

    task_for_direct_loop_test._scheduler_loop() # Call the loop directly

    assert mock_schedule_lib['run_pending'].call_count == 2 # Called twice before stop
    assert task_for_direct_loop_test._stop_event.is_set()
    mock_cleanup_logger.debug.assert_any_call("Scheduler loop starting.")
    mock_cleanup_logger.debug.assert_any_call("Scheduler loop stopping.")
    
    # Ensure the original task started by test_start_task_retention_enabled_not_running is stopped
    # This is tricky if that test actually started a thread that's still running.
    # Pytest fixtures should isolate, but if task is module level or shared, it can be an issue.
    # The current setup creates a new task instance per test.


def test_get_status_method(
    mock_cleanup_logger, mock_appconfig_cleanup_defaults, monkeypatch, 
    mock_schedule_lib, mock_time_control
):
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 7)
    task = sc_module.ScheduledCleanupTask()

    # Case 1: Before start
    status_before_start = task.get_status()
    assert status_before_start["is_running"] is False
    assert status_before_start["retention_enabled"] is True
    assert status_before_start["retention_days"] == 7
    assert status_before_start["cleanup_interval_hours"] == 24
    assert status_before_start["last_run_time"] is None
    assert status_before_start["next_run_time"] is None
    assert status_before_start["last_deleted_count"] == 0
    assert status_before_start["total_deleted_count"] == 0

    # Case 2: After a successful job run (simulated)
    run_time_dt = datetime.datetime(2023, 1, 1, 10, 0, 0)
    run_time_ts = run_time_dt.timestamp()
    mock_time_control[2].now.return_value = run_time_dt # Mock datetime.now() used by _run_cleanup_job

    task.last_run_time = run_time_ts
    task.last_deleted_count = 5
    task.total_deleted_count = 10
    
    # Simulate next run time from schedule
    next_run_dt = datetime.datetime(2023, 1, 2, 10, 0, 0) # 24 hours later
    mock_schedule_lib['next_run'].return_value = next_run_dt
    # Simulate task is "running" (thread exists and is alive)
    task._thread = MagicMock()
    task._thread.is_alive.return_value = True
    task._stop_event.clear()


    status_after_run = task.get_status()
    assert status_after_run["is_running"] is True
    assert status_after_run["last_run_time"] == run_time_dt.isoformat()
    assert status_after_run["next_run_time"] == next_run_dt.isoformat()
    assert status_after_run["last_deleted_count"] == 5
    assert status_after_run["total_deleted_count"] == 10
    mock_schedule_lib['next_run'].assert_called_once()

    # Case 3: Retention disabled
    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 0)
    task_ret_disabled = sc_module.ScheduledCleanupTask() # Re-init to pick up new config
    status_ret_disabled = task_ret_disabled.get_status()
    assert status_ret_disabled["retention_enabled"] is False
    assert status_ret_disabled["next_run_time"] is None # No next run if disabled

    # Case 4: schedule.next_run() raises an exception (e.g., no jobs scheduled)
    mock_schedule_lib['next_run'].reset_mock()
    mock_schedule_lib['next_run'].side_effect = Exception("No jobs scheduled")
    status_no_next_run = task.get_status() # task is still the one from Case 2
    assert status_no_next_run["next_run_time"] == "Not scheduled or error."
    mock_cleanup_logger.warning.assert_any_call(
        "Could not determine next run time from schedule: No jobs scheduled"
    )
