import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call

# Assuming the module is viralStoryGenerator.src.worker_runner
from viralStoryGenerator.src import worker_runner as worker_runner_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_worker_runner_defaults(monkeypatch):
    """Set default app_config values for worker_runner tests."""
    monkeypatch.setattr(app_config.api, 'HOST', "0.0.0.0")
    monkeypatch.setattr(app_config.api, 'PORT', 8000)
    monkeypatch.setattr(app_config.api, 'LOG_LEVEL', "info")
    monkeypatch.setattr(app_config.api, 'RELOAD_DIRS', ["src", "utils"]) # Example
    # For worker command
    monkeypatch.setattr(app_config.project, 'ROOT_DIR', "/test/project/root")


@pytest.fixture
def mock_sys_exit():
    """Fixture to mock sys.exit."""
    with patch('sys.exit') as mock_exit:
        yield mock_exit

@pytest.fixture
def mock_worker_runner_logger():
    """Fixture to mock the _logger in worker_runner.py."""
    with patch('viralStoryGenerator.src.worker_runner._logger') as mock_logger:
        yield mock_logger

# --- Tests for Scenario 1: API server command (`api`) ---

@patch('argparse.ArgumentParser.parse_args')
@patch('uvicorn.run')
def test_main_api_command_default(
    mock_uvicorn_run, mock_parse_args, 
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger # Fixtures
):
    mock_args = MagicMock()
    mock_args.command = "api"
    mock_args.reload = False
    mock_args.uvicorn_extra_args = None
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_uvicorn_run.assert_called_once_with(
        "viralStoryGenerator.src.api:app", # app_string
        host=app_config.api.HOST,
        port=app_config.api.PORT,
        reload=False,
        log_level=app_config.api.LOG_LEVEL,
        reload_dirs=None # Default when reload is False
    )
    mock_worker_runner_logger.info.assert_any_call(
        f"Starting API server on {app_config.api.HOST}:{app_config.api.PORT} with reload: False"
    )


@patch('argparse.ArgumentParser.parse_args')
@patch('uvicorn.run')
def test_main_api_command_with_reload(
    mock_uvicorn_run, mock_parse_args, 
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger
):
    mock_args = MagicMock()
    mock_args.command = "api"
    mock_args.reload = True # --reload flag is True
    mock_args.uvicorn_extra_args = None
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_uvicorn_run.assert_called_once_with(
        "viralStoryGenerator.src.api:app",
        host=app_config.api.HOST,
        port=app_config.api.PORT,
        reload=True,
        log_level=app_config.api.LOG_LEVEL,
        reload_dirs=app_config.api.RELOAD_DIRS # Should use configured reload_dirs
    )
    mock_worker_runner_logger.info.assert_any_call(
        f"Starting API server on {app_config.api.HOST}:{app_config.api.PORT} with reload: True, watching: {app_config.api.RELOAD_DIRS}"
    )


@patch('argparse.ArgumentParser.parse_args')
@patch('os.execvp') # Mock os.execvp for the extra_args path
@patch('sys.executable', '/path/to/python') # Mock sys.executable
def test_main_api_command_with_uvicorn_extra_args(
    mock_os_execvp, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger
):
    extra_args_list = ["--workers", "2", "--ssl-keyfile", "key.pem"]
    mock_args = MagicMock()
    mock_args.command = "api"
    mock_args.reload = False # Assume no reload when extra_args are used, or test combination
    mock_args.uvicorn_extra_args = extra_args_list
    mock_parse_args.return_value = mock_args

    # os.execvp will replace the current process, so it should not return.
    # If it's called, the test effectively stops there for that path.
    # We can mock it to raise an exception to stop the test after asserting calls.
    class ExecCalled(Exception): pass
    mock_os_execvp.side_effect = ExecCalled

    with pytest.raises(ExecCalled):
        worker_runner_module.main()

    expected_base_cmd = [
        '/path/to/python', '-m', 'uvicorn', 
        "viralStoryGenerator.src.api:app",
        '--host', app_config.api.HOST,
        '--port', str(app_config.api.PORT), # Port should be string for execvp
        '--log-level', app_config.api.LOG_LEVEL
        # No --reload or --reload-dirs by default if mock_args.reload is False
    ]
    expected_full_cmd = expected_base_cmd + extra_args_list
    
    mock_os_execvp.assert_called_once_with(expected_full_cmd[0], expected_full_cmd)
    mock_worker_runner_logger.info.assert_any_call(
        f"Starting API server with extra uvicorn args: {extra_args_list}. Executing: {' '.join(expected_full_cmd)}"
    )

# --- Tests for Scenario 2: Queue worker command ---

@patch('argparse.ArgumentParser.parse_args')
@patch('viralStoryGenerator.src.queue_worker.main') # Mock queue_worker.main
def test_main_worker_command_queue_type(
    mock_queue_worker_main, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "queue"
    mock_args.reload = False # No reload for this test
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_queue_worker_main.assert_called_once()
    mock_worker_runner_logger.info.assert_any_call("Starting Queue Worker...")
    mock_sys_exit.assert_not_called() # Should not exit if worker main runs and exits normally


@patch('argparse.ArgumentParser.parse_args')
@patch('viralStoryGenerator.src.queue_worker.main', side_effect=Exception("Queue worker init failed")) # Simulate worker error
def test_main_worker_command_queue_type_worker_exception(
    mock_queue_worker_main_with_exc, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "queue"
    mock_args.reload = False
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_queue_worker_main_with_exc.assert_called_once()
    mock_worker_runner_logger.error.assert_any_call(
        "Error running Queue Worker: Queue worker init failed", exc_info=True
    )
    mock_sys_exit.assert_called_once_with(1) # Should exit with error code

# --- Tests for Scenario 3: Scrape worker command ---

@patch('argparse.ArgumentParser.parse_args')
@patch('viralStoryGenerator.src.scrape_worker.main') # Mock scrape_worker.main
def test_main_worker_command_scrape_type(
    mock_scrape_worker_main, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "scrape"
    mock_args.reload = False
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_scrape_worker_main.assert_called_once()
    mock_worker_runner_logger.info.assert_any_call("Starting Scraper Worker...")
    mock_sys_exit.assert_not_called()


@patch('argparse.ArgumentParser.parse_args')
@patch('viralStoryGenerator.src.scrape_worker.main', side_effect=Exception("Scraper worker init failed"))
def test_main_worker_command_scrape_type_worker_exception(
    mock_scrape_worker_main_with_exc, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "scrape"
    mock_args.reload = False
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_scrape_worker_main_with_exc.assert_called_once()
    mock_worker_runner_logger.error.assert_any_call(
        "Error running Scraper Worker: Scraper worker init failed", exc_info=True
    )
    mock_sys_exit.assert_called_once_with(1)

# --- Tests for Scenario 4: Worker command with --reload ---

@patch('argparse.ArgumentParser.parse_args')
@patch('watchfiles.run_process') # Mock watchfiles.run_process
@patch('viralStoryGenerator.src.queue_worker.main') # Target worker's main
def test_main_worker_command_queue_with_reload_watchfiles_installed(
    mock_queue_worker_main_target, mock_watchfiles_run_process, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "queue"
    mock_args.reload = True # --reload is True
    mock_args.uvicorn_extra_args = None # Not used for worker reload
    mock_parse_args.return_value = mock_args
    
    # Simulate watchfiles being available
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker_runner_module, 'run_process', mock_watchfiles_run_process)

    worker_runner_module.main()

    mock_watchfiles_run_process.assert_called_once()
    call_args_kwargs = mock_watchfiles_run_process.call_args[1] # Get kwargs
    
    assert call_args_kwargs['path'] == app_config.project.ROOT_DIR
    assert call_args_kwargs['target'] == mock_queue_worker_main_target # Check if target is queue_worker.main
    assert call_args_kwargs['watch_filter'] is not None # Check some WatchFilesFilter is passed
    
    mock_worker_runner_logger.info.assert_any_call(
        f"Starting Queue Worker with reload enabled, watching directory: {app_config.project.ROOT_DIR}"
    )
    mock_queue_worker_main_target.assert_not_called() # run_process calls the target, not main directly
    mock_sys_exit.assert_not_called() # Should run indefinitely via watchfiles


@patch('argparse.ArgumentParser.parse_args')
@patch('viralStoryGenerator.src.queue_worker.main') # Target worker's main, though not directly called
def test_main_worker_command_queue_with_reload_watchfiles_not_installed(
    mock_queue_worker_main_target_no_wf, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit, monkeypatch
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "queue"
    mock_args.reload = True
    mock_parse_args.return_value = mock_args
    
    # Simulate watchfiles not being available by setting run_process to None
    monkeypatch.setattr(worker_runner_module, 'run_process', None)

    worker_runner_module.main()

    mock_worker_runner_logger.error.assert_any_call(
        "Watchfiles is not installed. Please install it to use the --reload feature for workers. `pip install watchfiles`"
    )
    mock_sys_exit.assert_called_once_with(1)
    mock_queue_worker_main_target_no_wf.assert_not_called()

# --- Tests for Scenario 5: Invalid worker type ---

@patch('argparse.ArgumentParser.parse_args')
def test_main_worker_command_invalid_type(
    mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    mock_args = MagicMock()
    mock_args.command = "worker"
    mock_args.worker_type = "invalid_worker_type" # Invalid type
    mock_args.reload = False
    mock_parse_args.return_value = mock_args

    worker_runner_module.main()

    mock_worker_runner_logger.error.assert_any_call(
        f"Unknown worker type: {mock_args.worker_type}"
    )
    mock_sys_exit.assert_called_once_with(1)


# --- Tests for Scenario 6: No command or invalid command ---

@patch('argparse.ArgumentParser.parse_args')
@patch('argparse.ArgumentParser.print_help') # Mock print_help
def test_main_no_command_provided(
    mock_print_help, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    # Simulate parse_args returning args where command is None (or not set)
    # This typically happens if the ArgumentParser is set up to default command to None
    # or if subparsers are used and no subparser command is given.
    # For this test, let's assume the main() function's logic checks if args.command is None.
    
    mock_args = MagicMock()
    mock_args.command = None # No command provided
    mock_parse_args.return_value = mock_args
    
    # The ArgumentParser instance is created inside main(). To mock its print_help,
    # we need to get a reference to that instance.
    # A common way is to patch ArgumentParser itself to control the instance.
    
    # Re-patching ArgumentParser to get hold of the instance
    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        # Mock print_help on the instance that will be created
        mock_parser_instance.print_help = MagicMock() 
        MockArgumentParser.return_value = mock_parser_instance

        worker_runner_module.main()

        mock_parser_instance.print_help.assert_called_once()
        mock_sys_exit.assert_called_once_with(1)
        # Logger might not be called here, or might log "No command provided."
        # Based on typical argparse behavior, print_help then exit is common.


@patch('argparse.ArgumentParser.parse_args')
@patch('argparse.ArgumentParser.print_help')
def test_main_invalid_command_provided(
    mock_print_help, mock_parse_args,
    mock_appconfig_worker_runner_defaults, mock_worker_runner_logger, mock_sys_exit
):
    # This scenario is usually handled by argparse itself if the command is truly unknown
    # to the parser (e.g., it's not a defined subparser).
    # ArgumentParser.parse_args() would raise SystemExit or call error().
    # Let's assume main() has a default case after checking known commands.
    
    mock_args = MagicMock()
    mock_args.command = "unknown_command" # Command not 'api' or 'worker'
    mock_parse_args.return_value = mock_args

    with patch('argparse.ArgumentParser') as MockArgumentParser:
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser_instance.print_help = MagicMock()
        MockArgumentParser.return_value = mock_parser_instance

        worker_runner_module.main()

        mock_worker_runner_logger.error.assert_any_call(f"Unknown command: {mock_args.command}")
        mock_parser_instance.print_help.assert_called_once()
        mock_sys_exit.assert_called_once_with(1)
