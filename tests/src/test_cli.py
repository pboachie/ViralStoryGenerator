import pytest
import json
import os
import datetime
from unittest.mock import patch, MagicMock, mock_open

# Assuming the cli module is viralStoryGenerator.src.cli
from viralStoryGenerator.src import cli as cli_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Mocks for the entire module if needed, or per test ---

# --- Tests for cli_main (Argument Parsing) ---
@patch('argparse.ArgumentParser.parse_args')
@patch('viralStoryGenerator.src.cli._logger') # Patching the logger in cli.py
def test_cli_main_parses_args_and_logs_deprecation(mock_logger, mock_parse_args):
    # Simulate parse_args returning a simple namespace object
    mock_args = MagicMock()
    mock_args.topic = "Test Topic from CLI"
    mock_args.prompt = None
    mock_args.sources_folder = None
    mock_args.voice_id = None
    mock_args.include_storyboard = False 
    # Add other args if cli_main expects them after parse_args
    
    mock_parse_args.return_value = mock_args
    
    # cli_main is expected to call other functions which might need further mocking
    # For this test, we only care about parse_args and the deprecation log.
    # If cli_main tries to do significant work, we might need to stop it early
    # or mock out more dependencies.
    # For a minimal test, let's assume it doesn't crash after logging.
    # If it calls, e.g., create_story_task, that would need mocking.
    
    # Patching functions called by cli_main to prevent actual execution beyond what's tested.
    with patch('viralStoryGenerator.src.cli.create_story_task_from_cli_args', return_value=None) as mock_create_task_cli, \
         patch('viralStoryGenerator.src.cli.process_audio_queue', return_value=None) as mock_process_audio:
        
        cli_module.cli_main()

    mock_parse_args.assert_called_once()
    mock_logger.warning.assert_any_call("CLI execution is deprecated and will be removed in a future version. Please use the API.")
    # mock_create_task_cli might be called if topic is provided. This depends on cli_main logic.
    # If cli_main directly calls create_story_task or similar, that call should be asserted too if relevant.
    # Based on the prompt, the focus is on parse_args and the deprecation log.


# --- Tests for queue_failed_audio ---
@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
@patch('datetime.datetime') # To control datetime.now()
@patch('viralStoryGenerator.src.cli.sanitize_filename') # Assuming sanitize_filename is used
def test_queue_failed_audio_writes_correct_file(
    mock_sanitize_filename, mock_datetime, mock_json_dump, mock_open_file, mock_os_makedirs, monkeypatch
):
    # Setup app_config for AUDIO_QUEUE_DIR
    audio_queue_dir = "/mock/audio_queue"
    monkeypatch.setattr(app_config, 'AUDIO_QUEUE_DIR', audio_queue_dir)

    mock_now = datetime.datetime(2023, 10, 26, 12, 30, 0)
    mock_datetime.now.return_value = mock_now
    
    sanitized_topic = "test_topic_sanitized"
    mock_sanitize_filename.return_value = sanitized_topic
    
    metadata_to_queue = {
        "job_id": "job123",
        "topic": "Test Topic Raw", # Raw topic before sanitization for filename
        "story": "This is a story.",
        "mp3_file_path": "/path/to/audio.mp3",
        "voice_id": "voice_abc"
    }
    
    expected_timestamp = "20231026_123000"
    expected_filename = f"{sanitized_topic}_{expected_timestamp}.json"
    expected_full_path = os.path.join(audio_queue_dir, expected_filename)

    cli_module.queue_failed_audio(metadata_to_queue)

    mock_os_makedirs.assert_called_once_with(audio_queue_dir, exist_ok=True)
    mock_sanitize_filename.assert_called_once_with("Test Topic Raw")
    mock_open_file.assert_called_once_with(expected_full_path, 'w')
    mock_json_dump.assert_called_once_with(metadata_to_queue, mock_open_file(), indent=4)


# --- Tests for process_audio_queue (CLI version) ---
# This will be similar to the api_handlers version but will patch cli's appconfig
mock_cli_generate_audio = MagicMock()

@pytest.fixture(autouse=True)
def reset_cli_audio_mocks():
    mock_cli_generate_audio.reset_mock()

@patch('viralStoryGenerator.src.cli._logger') # Logger in cli.py
@patch('os.path.isdir')
@patch('os.listdir')
@patch('os.path.join')
@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
@patch('os.remove')
@patch('viralStoryGenerator.src.elevenlabs_tts.generate_elevenlabs_audio', mock_cli_generate_audio) # Patch where it's imported by cli
def test_cli_process_audio_queue_successful_job(
    mock_os_remove, mock_json_load, mock_builtin_open_cli, mock_os_path_join_cli,
    mock_os_listdir_cli, mock_os_path_isdir_cli, mock_logger_cli, monkeypatch
):
    queue_dir = "/cli_mock/audio_queue"
    job_file_name = "cli_job1.json"
    full_job_path = os.path.join(queue_dir, job_file_name) # Use os.path.join for consistency

    monkeypatch.setattr(cli_module.appconfig, 'AUDIO_QUEUE_DIR', queue_dir) # Patch appconfig in cli.py
    monkeypatch.setattr(cli_module.appconfig.elevenlabs, 'API_KEY', "cli_fake_key")

    mock_os_path_isdir_cli.return_value = True
    mock_os_listdir_cli.return_value = [job_file_name]
    mock_os_path_join_cli.return_value = full_job_path # Ensure join returns the expected path

    job_data = {
        "story": "CLI test story.", "mp3_file_path": "/cli_output/audio.mp3", "voice_id": "voice_cli"
    }
    mock_json_load.return_value = job_data
    mock_cli_generate_audio.return_value = True # Success

    cli_module.process_audio_queue() # Call the cli version

    mock_os_path_join_cli.assert_called_once_with(queue_dir, job_file_name)
    mock_builtin_open_cli.assert_called_once_with(full_job_path, 'r')
    mock_json_load.assert_called_once_with(mock_builtin_open_cli())
    mock_cli_generate_audio.assert_called_once_with(
        job_data["story"], job_data["mp3_file_path"], job_data["voice_id"]
    )
    mock_os_remove.assert_called_once_with(full_job_path)
    mock_logger_cli.info.assert_any_call(f"Audio generated successfully for job from file: {job_file_name}")


@patch('viralStoryGenerator.src.cli._logger')
@patch('os.path.isdir', return_value=True)
@patch('os.listdir', return_value=["cli_job_fail.json"])
@patch('os.path.join')
@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
@patch('os.remove')
@patch('viralStoryGenerator.src.elevenlabs_tts.generate_elevenlabs_audio', mock_cli_generate_audio)
def test_cli_process_audio_queue_generation_fails(
    mock_os_remove, mock_json_load, mock_builtin_open_cli, mock_os_path_join_cli,
    mock_os_listdir_cli, mock_os_path_isdir_cli, mock_logger_cli, monkeypatch
):
    queue_dir = "/cli_mock/audio_queue_fail"
    job_file_name = "cli_job_fail.json"
    full_job_path = os.path.join(queue_dir, job_file_name)

    monkeypatch.setattr(cli_module.appconfig, 'AUDIO_QUEUE_DIR', queue_dir)
    monkeypatch.setattr(cli_module.appconfig.elevenlabs, 'API_KEY', "cli_fake_key")

    mock_os_path_isdir_cli.return_value = True
    mock_os_listdir_cli.return_value = [job_file_name]
    mock_os_path_join_cli.return_value = full_job_path
    
    job_data = {"story": "CLI fail story", "mp3_file_path": "/cli_output/fail.mp3", "voice_id": "voice_cli_fail"}
    mock_json_load.return_value = job_data
    mock_cli_generate_audio.return_value = False # Generation fails

    cli_module.process_audio_queue()

    mock_cli_generate_audio.assert_called_once()
    mock_os_remove.assert_not_called()
    mock_logger_cli.warning.assert_any_call(f"Audio generation failed for job from file: {job_file_name}. Will retry later.")


# --- Tests for _read_sources_from_folder ---
@patch('os.path.isdir')
@patch('os.listdir')
@patch('os.path.isfile')
@patch('builtins.open', new_callable=mock_open)
def test_read_sources_from_folder_reads_files(
    mock_open_file_cli, mock_os_path_isfile, mock_os_listdir, mock_os_path_isdir
):
    folder_path = "/mock_sources"
    mock_os_path_isdir.return_value = True
    mock_os_listdir.return_value = ["file1.txt", "file2.md", "ignored.log"]
    
    # Simulate os.path.join and os.path.isfile behavior
    # os.path.join will be called like os.path.join(folder_path, filename)
    # os.path.isfile will be called with the result of os.path.join
    def isfile_side_effect(path):
        if path.endswith(".txt") or path.endswith(".md"):
            return True
        return False
    mock_os_path_isfile.side_effect = isfile_side_effect
    
    # Configure mock_open to return different content for different files
    # This requires a more complex side_effect for mock_open itself.
    def open_side_effect(filepath, mode='r'):
        if filepath == os.path.join(folder_path, "file1.txt"):
            return mock_open(read_data="Content of file1").return_value
        elif filepath == os.path.join(folder_path, "file2.md"):
            return mock_open(read_data="Content of file2").return_value
        else:
            # Should not happen if isfile_side_effect is correct
            raise FileNotFoundError(f"Unexpected file open: {filepath}")
            
    mock_open_file_cli.side_effect = open_side_effect

    sources = cli_module._read_sources_from_folder(folder_path)

    assert len(sources) == 2
    assert "Content of file1" in sources
    assert "Content of file2" in sources
    
    mock_os_path_isdir.assert_called_once_with(folder_path)
    mock_os_listdir.assert_called_once_with(folder_path)
    # isfile and open are called for each file in listdir result
    assert mock_os_path_isfile.call_count == 3
    # open is called only for .txt and .md files
    mock_open_file_cli.assert_any_call(os.path.join(folder_path, "file1.txt"), 'r')
    mock_open_file_cli.assert_any_call(os.path.join(folder_path, "file2.md"), 'r')
    assert mock_open_file_cli.call_count == 2


@patch('os.path.isdir')
def test_read_sources_from_folder_non_existent_folder(mock_os_path_isdir):
    folder_path = "/non_existent_sources"
    mock_os_path_isdir.return_value = False # Folder does not exist

    sources = cli_module._read_sources_from_folder(folder_path)

    assert sources == []
    mock_os_path_isdir.assert_called_once_with(folder_path)
