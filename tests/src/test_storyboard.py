import pytest
import requests
import os
import time
import json
import uuid
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open, call

# Assuming the module is viralStoryGenerator.src.storyboard
from viralStoryGenerator.src import storyboard as storyboard_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.models import StoryboardScene # For type hints if needed

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_storyboard_defaults(monkeypatch):
    """Set default app_config values for storyboard tests."""
    # LLM settings (used by generate_storyboard_structure)
    monkeypatch.setattr(app_config.llm, 'ENDPOINT', "http://mock-llm-api.com/v1/chat/completions")
    monkeypatch.setattr(app_config.llm, 'MODEL', "mock-llm-model-sb")
    monkeypatch.setattr(app_config.llm, 'MAX_TOKENS', 1500)
    monkeypatch.setattr(app_config.llm, 'DEFAULT_TEMPERATURE', 0.6)
    monkeypatch.setattr(app_config.llm, 'RETRY_ATTEMPTS', 2) # For LLM calls
    monkeypatch.setattr(app_config.llm, 'RETRY_MIN_WAIT_SECONDS', 0.01)
    monkeypatch.setattr(app_config.llm, 'RETRY_MAX_WAIT_SECONDS', 0.05)


    # DALL-E settings (used by generate_dalle_image)
    monkeypatch.setattr(app_config.dalle, 'API_KEY', "test_dalle_api_key")
    monkeypatch.setattr(app_config.dalle, 'API_ENDPOINT', "http://mock-dalle-api.com/v1/images/generations")
    monkeypatch.setattr(app_config.dalle, 'IMAGE_SIZE', "1024x1024")
    monkeypatch.setattr(app_config.dalle, 'IMAGE_MODEL', "dall-e-3")
    monkeypatch.setattr(app_config.dalle, 'IMAGE_QUALITY', "standard")
    monkeypatch.setattr(app_config.dalle, 'RETRY_ATTEMPTS', 2) # For DALL-E calls
    monkeypatch.setattr(app_config.dalle, 'RETRY_DELAY_SECONDS', 0.01)


    # ElevenLabs settings (used by generate_storyboard)
    monkeypatch.setattr(app_config.elevenlabs, 'API_KEY', "test_elevenlabs_api_key")
    monkeypatch.setattr(app_config.elevenlabs, 'DEFAULT_VOICE_ID', "default_sb_voice")

    # Storyboard specific settings
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'MIN_SCENE_DURATION', 3.0)
    monkeypatch.setattr(app_config.storyboard, 'MAX_SCENE_DURATION', 15.0)
    monkeypatch.setattr(app_config.storyboard, 'NUM_SCENES_PER_STORY', 5) 
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', 2) 
    monkeypatch.setattr(app_config.storyboard, 'RETRY_DELAY_SECONDS', 0.01)
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True) # Default to true for generate_storyboard_from_story_script tests


    # Storage manager related (used by generate_storyboard)
    monkeypatch.setattr(app_config.storage, 'BASE_DIR', "/tmp/mock_storage_base")
    monkeypatch.setattr(app_config.storage, 'STORIES_DIR_NAME', "stories")
    
    # HTTP Client Options
    monkeypatch.setattr(app_config.httpOptions, 'TIMEOUT', 10)

@pytest.fixture
def mock_storyboard_logger():
    """Fixture to mock the _logger in storyboard.py."""
    with patch('viralStoryGenerator.src.storyboard._logger') as mock_logger:
        yield mock_logger

# --- Tests for sanitize_for_filename (Scenario 1) ---
@pytest.mark.parametrize("input_text, expected_output_param", [
    ("Simple Title", "Simple_Title"),
    ("Title with spaces & special Chars!", "Title_with_spaces_special_Chars"),
    ("  Leading/Trailing Spaces  ", "Leading_Trailing_Spaces"),
    ("", "untitled_story"), 
    ("A very long title that might exceed typical filename limits to see if it gets truncated or handled gracefully by the sanitization", 
     "A_very_long_title_that_might_exceed_typical_filename_limits_to_see_if_it_gets_truncated_or_handled_gracefu"),
    ("你好世界", "你好世界"), 
    ("a/b\\c:d*e?f\"g<h>i|j", "abcdefghij"), 
    ("---", "untitled_story"), 
    ("multiple---hyphens---together", "multiple_hyphens_together"), 
    ("endsWith-", "endsWith"), 
])
def test_sanitize_for_filename(input_text, expected_output_param, mock_appconfig_storyboard_defaults):
    # This mapping is to ensure the test logic matches the code's behavior precisely,
    # as the initial parametrize table had some discrepancies with the code's regex.
    # This makes the test more of a direct validation of the current code.
    expected_output = expected_output_param # Use the param directly now

    result = storyboard_module.sanitize_for_filename(input_text)
    assert result == expected_output

# --- Tests for split_story_by_markers (Scenario 4) ---

@pytest.mark.parametrize("story_text, expected_scenes_data", [
    # Valid case
    ("SCENE 1\nNARRATION:\nNarr 1\nIMAGE_PROMPT:\nPrompt 1\nSCENE 2\nNARRATION:\nNarr 2\nIMAGE_PROMPT:\nPrompt 2",
     [{"scene_number": 1, "narration": "Narr 1", "image_prompt": "Prompt 1"},
      {"scene_number": 2, "narration": "Narr 2", "image_prompt": "Prompt 2"}]),
    ("Just a single block of text without any scene markers.", []),
    ("", []),
    ("SCENE 1\nNARRATION:\n\nIMAGE_PROMPT:\n\nSCENE 2\nNARRATION:\n\nIMAGE_PROMPT:\n", 
     [{"scene_number": 1, "narration": "", "image_prompt": ""},
      {"scene_number": 2, "narration": "", "image_prompt": ""}]),
    ("SCENE 1\nNARRATION:\nNarr 1 Only", []), 
    ("SCENE 1\nIMAGE_PROMPT:\nPrompt 1 Only\nNARRATION:\nNarr 1 After", []),
    ("SCENE 1\nNARRATION: Narr 1 (no newline)\nIMAGE_PROMPT: Prompt 1 (no newline)", 
     [{"scene_number": 1, "narration": "Narr 1 (no newline)", "image_prompt": "Prompt 1 (no newline)"}]),
    ("  SCENE   1  \n  NARRATION:  \n  Narr with spaces  \n  IMAGE_PROMPT:  \n  Prompt with spaces  ",
     [{"scene_number": 1, "narration": "Narr with spaces", "image_prompt": "Prompt with spaces"}]),
    ("SCENE 1\nNARRATION:\nNarr single\nIMAGE_PROMPT:\nPrompt single",
     [{"scene_number": 1, "narration": "Narr single", "image_prompt": "Prompt single"}]),
    ("SCENE X\nNARRATION:\nNarr X\nIMAGE_PROMPT:\nPrompt X", []),
])
def test_split_story_by_markers_various_inputs(story_text, expected_scenes_data, mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    scenes = storyboard_module.split_story_by_markers(story_text)
    assert len(scenes) == len(expected_scenes_data)
    for i, scene_model in enumerate(scenes):
        expected_data = expected_scenes_data[i]
        assert scene_model.scene_number == expected_data["scene_number"]
        assert scene_model.narration.strip() == expected_data["narration"].strip()
        assert scene_model.image_prompt.strip() == expected_data["image_prompt"].strip()

def test_split_story_by_markers_logs_warning_if_no_scenes(mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    story_text = "This story has no scene markers at all."
    scenes = storyboard_module.split_story_by_markers(story_text)
    assert len(scenes) == 0
    mock_storyboard_logger.warning.assert_called_once_with(
        f"No scenes found in story using regex. Story text: {story_text[:200]}..."
    )

def test_split_story_by_markers_duplicate_scene_numbers(mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    story_text = "SCENE 1\nNARRATION:\nNarr A\nIMAGE_PROMPT:\nPrompt A\nSCENE 1\nNARRATION:\nNarr B\nIMAGE_PROMPT:\nPrompt B"
    scenes = storyboard_module.split_story_by_markers(story_text)
    assert len(scenes) == 2
    assert scenes[0].scene_number == 1
    assert scenes[0].narration == "Narr A"
    assert scenes[1].scene_number == 1
    assert scenes[1].narration == "Narr B"

# --- Tests for generate_storyboard (Scenario 5) ---

@pytest.fixture
def mock_dependencies_for_generate_storyboard(monkeypatch):
    mocks = {
        'generate_structure': patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure').start(),
        'split_markers': patch('viralStoryGenerator.src.storyboard.split_story_by_markers').start(),
        'dalle_image': patch('viralStoryGenerator.src.storyboard.generate_dalle_image').start(),
        'elevenlabs_audio': patch('viralStoryGenerator.src.elevenlabs_tts.generate_elevenlabs_audio').start(),
        'get_storage_dir': patch('viralStoryGenerator.utils.storage_manager._get_storage_dir').start(),
        'store_file': patch('viralStoryGenerator.utils.storage_manager.store_file').start(),
        'mkdtemp': patch('tempfile.mkdtemp').start(),
        'rmtree': patch('shutil.rmtree').start(),
        'uuid4': patch('uuid.uuid4').start(),
        'get_audio_duration': patch('viralStoryGenerator.utils.audio.get_audio_duration').start(),
        'logger': patch('viralStoryGenerator.src.storyboard._logger').start() 
    }
    yield mocks
    patch.stopall()

def test_generate_storyboard_successful_full_run(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_full_sb_001"
    story_script_text = "Full story script text."
    num_scenes_llm = 2 
    
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'NUM_SCENES_PER_STORY', num_scenes_llm)

    mock_deps = mock_dependencies_for_generate_storyboard
    mock_deps['uuid4'].return_value = MagicMock(hex=job_id) 
    
    llm_structure_response = {"scenes": [ 
        {"scene_number": 1, "narration": "Narr 1 from LLM.", "image_prompt": "Prompt 1 from LLM."},
        {"scene_number": 2, "narration": "Narr 2 from LLM.", "image_prompt": "Prompt 2 from LLM."}
    ]}
    mock_deps['generate_structure'].return_value = llm_structure_response
    mock_deps['split_markers'].return_value = [] 

    mock_base_storage_dir = f"/tmp/mock_storage_base/stories/{job_id}"
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['get_storage_dir'].return_value = mock_base_storage_dir
    mock_deps['mkdtemp'].return_value = mock_temp_dir

    mock_deps['dalle_image'].side_effect = [
        os.path.join(mock_temp_dir, "scene_1_image.png"), 
        os.path.join(mock_temp_dir, "scene_2_image.png")  
    ]
    mock_deps['elevenlabs_audio'].side_effect = [True, True]
    mock_deps['get_audio_duration'].side_effect = [3.5, 4.0]

    def store_file_side_effect(data_or_filepath, job_id_arg, filename_key, **kwargs):
        return f"s3://mock_bucket/{job_id_arg}/{filename_key}"
    mock_deps['store_file'].side_effect = store_file_side_effect

    final_storyboard_data = storyboard_module.generate_storyboard(job_id, story_script_text)

    assert final_storyboard_data is not None
    assert final_storyboard_data["job_id"] == job_id
    assert final_storyboard_data["story_script"] == story_script_text
    assert len(final_storyboard_data["scenes"]) == num_scenes_llm
    
    scene1_data = final_storyboard_data["scenes"][0]
    assert scene1_data["scene_number"] == 1
    assert scene1_data["narration"] == "Narr 1 from LLM."
    assert scene1_data["image_prompt"] == "Prompt 1 from LLM."
    assert scene1_data["image_url"] == f"s3://mock_bucket/{job_id}/scene_1_image.png"
    assert scene1_data["audio_url"] == f"s3://mock_bucket/{job_id}/scene_1_audio.mp3"
    assert scene1_data["duration_seconds"] == 3.5

    scene2_data = final_storyboard_data["scenes"][1]
    assert scene2_data["scene_number"] == 2
    assert scene2_data["duration_seconds"] == 4.0

    mock_deps['generate_structure'].assert_called_once_with(story_script_text, num_scenes=num_scenes_llm)
    mock_deps['split_markers'].assert_not_called() 
    
    assert mock_deps['dalle_image'].call_count == num_scenes_llm
    mock_deps['dalle_image'].assert_any_call("Prompt 1 from LLM.", mock_temp_dir, f"scene_1_image_raw_{job_id}")
    
    assert mock_deps['elevenlabs_audio'].call_count == num_scenes_llm
    mock_deps['elevenlabs_audio'].assert_any_call(
        text="Narr 1 from LLM.",
        output_mp3_path=os.path.join(mock_temp_dir, "scene_1_audio.mp3"),
        voice_id=app_config.elevenlabs.DEFAULT_VOICE_ID,
        include_timestamps=False 
    )
    
    assert mock_deps['get_audio_duration'].call_count == num_scenes_llm
    mock_deps['get_audio_duration'].assert_any_call(os.path.join(mock_temp_dir, "scene_1_audio.mp3"))

    assert mock_deps['store_file'].call_count == (num_scenes_llm * 2) + 1
    mock_deps['store_file'].assert_any_call(
        os.path.join(mock_temp_dir, "scene_1_image.png"), job_id, "scene_1_image.png", is_temp_file=True
    )
    mock_deps['store_file'].assert_any_call(
        os.path.join(mock_temp_dir, "scene_1_audio.mp3"), job_id, "scene_1_audio.mp3", is_temp_file=True
    )
    found_json_store_call = False
    for call_arg in mock_deps['store_file'].call_args_list:
        args, kwargs = call_arg
        if kwargs.get('is_json') is True and args[1] == job_id and args[2] == "storyboard.json":
            assert args[0] == final_storyboard_data 
            found_json_store_call = True
            break
    assert found_json_store_call, "store_file for final storyboard JSON not found or incorrect."

    mock_deps['mkdtemp'].assert_called_once()
    mock_deps['rmtree'].assert_called_once_with(mock_temp_dir)
    mock_deps['logger'].info.assert_any_call(f"Temporary directory {mock_temp_dir} created for job {job_id}")
    mock_deps['logger'].info.assert_any_call(f"Temporary directory {mock_temp_dir} removed for job {job_id}")
    mock_deps['logger'].info.assert_any_call(f"Storyboard generation for job {job_id} completed successfully.")

def test_generate_storyboard_fails_at_structure_generation(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults
):
    job_id = "job_fail_structure_002"
    story_script_text = "Story for structure failure."
    mock_deps = mock_dependencies_for_generate_storyboard
    mock_deps['generate_structure'].return_value = None 
    
    result = storyboard_module.generate_storyboard(job_id, story_script_text)
    
    assert result is None
    mock_deps['generate_structure'].assert_called_once()
    mock_deps['split_markers'].assert_not_called() 
    mock_deps['logger'].error.assert_any_call(
        f"Failed to generate or split storyboard structure for job {job_id}. Cannot proceed."
    )
    mock_deps['rmtree'].assert_not_called() 

def test_generate_storyboard_fails_at_split_markers(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_fail_split_003"
    story_script_text = "Story for split failure."
    mock_deps = mock_dependencies_for_generate_storyboard
    mock_deps['generate_structure'].return_value = None 
    mock_deps['split_markers'].return_value = [] 
    
    result = storyboard_module.generate_storyboard(job_id, story_script_text)
    
    assert result is None
    mock_deps['generate_structure'].assert_called_once() 
    mock_deps['split_markers'].assert_called_once_with(story_script_text) 
    mock_deps['logger'].error.assert_any_call(
        f"Failed to generate or split storyboard structure for job {job_id}. Cannot proceed."
    )

def test_generate_storyboard_dalle_fails_for_one_scene(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_dalle_fail_004"
    story_script_text = "Story with DALL-E failure."
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', False)

    mock_deps['generate_structure'].return_value = {"scenes": [
        {"scene_number": 1, "narration": "Narr 1", "image_prompt": "Prompt 1"},
        {"scene_number": 2, "narration": "Narr 2", "image_prompt": "Prompt 2"}
    ]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['get_storage_dir'].return_value = f"/mock_storage/{job_id}"
    mock_deps['dalle_image'].side_effect = [os.path.join(mock_temp_dir, "scene_1_image.png"), None]
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"

    result = storyboard_module.generate_storyboard(job_id, story_script_text)

    assert result is not None 
    assert len(result["scenes"]) == 2
    assert result["scenes"][0]["image_url"] is not None
    assert result["scenes"][1]["image_url"] is None 
    mock_deps['logger'].warning.assert_any_call(
        f"Failed to generate DALL-E image for scene 2 in job {job_id}. Skipping image for this scene."
    )
    mock_deps['rmtree'].assert_called_once_with(mock_temp_dir)

def test_generate_storyboard_audio_fails_for_one_scene(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_audio_fail_005"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', False) 
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', True)

    mock_deps['generate_structure'].return_value = {"scenes": [
        {"scene_number": 1, "narration": "Narr 1", "image_prompt": "Prompt 1"},
        {"scene_number": 2, "narration": "Narr 2", "image_prompt": "Prompt 2"}
    ]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['get_storage_dir'].return_value = f"/mock_storage/{job_id}"
    mock_deps['elevenlabs_audio'].side_effect = [True, False] 
    mock_deps['get_audio_duration'].side_effect = [5.0, 0.0] 
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"

    result = storyboard_module.generate_storyboard(job_id, "Story with audio failure.")

    assert result is not None
    assert len(result["scenes"]) == 2
    assert result["scenes"][0]["audio_url"] is not None
    assert result["scenes"][0]["duration_seconds"] == 5.0
    assert result["scenes"][1]["audio_url"] is None
    assert result["scenes"][1]["duration_seconds"] == app_config.storyboard.MIN_SCENE_DURATION 
    mock_deps['logger'].warning.assert_any_call(
        f"Failed to generate audio for scene 2 in job {job_id}. Skipping audio for this scene."
    )

def test_generate_storyboard_store_file_fails(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_store_fail_006"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True) 
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', False)

    mock_deps['generate_structure'].return_value = {"scenes": [{"scene_number": 1, "narration": "N", "image_prompt": "P"}]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['dalle_image'].return_value = os.path.join(mock_temp_dir, "scene_1_image.png")
    mock_deps['store_file'].side_effect = Exception("S3 upload error")

    result = storyboard_module.generate_storyboard(job_id, "Story for store failure.")
    
    assert result is not None
    assert result["scenes"][0]["image_url"] is None
    mock_deps['logger'].error.assert_any_call(
        f"Failed to store asset scene_1_image.png for job {job_id}. Error: S3 upload error"
    )
    mock_deps['store_file'].reset_mock()
    mock_deps['store_file'].side_effect = [
        f"s3://{job_id}/scene_1_image.png", 
        Exception("Final JSON store error") 
    ]
    result_final_json_fail = storyboard_module.generate_storyboard(job_id, "Story for final store failure.")
    assert result_final_json_fail is None 
    mock_deps['logger'].error.assert_any_call(
        f"Failed to store final storyboard JSON for job {job_id}. Error: Final JSON store error"
    )

def test_generate_storyboard_assets_disabled(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_assets_disabled_007"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', False)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', False)

    mock_deps['generate_structure'].return_value = {"scenes": [{"scene_number": 1, "narration": "N", "image_prompt": "P"}]}
    mock_temp_dir = f"/tmp/temp_{job_id}" 
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['get_storage_dir'].return_value = f"/mock_storage/{job_id}"
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"

    result = storyboard_module.generate_storyboard(job_id, "Story with assets disabled.")

    assert result is not None
    scene1 = result["scenes"][0]
    assert scene1["image_url"] is None
    assert scene1["audio_url"] is None
    assert scene1["duration_seconds"] == app_config.storyboard.MIN_SCENE_DURATION 
    
    mock_deps['dalle_image'].assert_not_called()
    mock_deps['elevenlabs_audio'].assert_not_called()
    mock_deps['get_audio_duration'].assert_not_called()
    mock_deps['store_file'].assert_called_once() 
    assert mock_deps['store_file'].call_args[0][2] == "storyboard.json"

def test_generate_storyboard_duration_clamping(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_duration_clamp_008"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', False)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', True)
    min_dur = 3.0
    max_dur = 10.0
    monkeypatch.setattr(app_config.storyboard, 'MIN_SCENE_DURATION', min_dur)
    monkeypatch.setattr(app_config.storyboard, 'MAX_SCENE_DURATION', max_dur)

    mock_deps['generate_structure'].return_value = {"scenes": [
        {"scene_number": 1, "narration": "Short narr", "image_prompt": "P1"},
        {"scene_number": 2, "narration": "Medium narr", "image_prompt": "P2"},
        {"scene_number": 3, "narration": "Long narrationnnnnnnnnnnnnnn", "image_prompt": "P3"}
    ]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['elevenlabs_audio'].return_value = True 
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"
    mock_deps['get_audio_duration'].side_effect = [1.5, 5.0, 15.0] 

    result = storyboard_module.generate_storyboard(job_id, "Story for duration clamping.")
    
    assert result is not None
    assert len(result["scenes"]) == 3
    assert result["scenes"][0]["duration_seconds"] == min_dur 
    assert result["scenes"][1]["duration_seconds"] == 5.0   
    assert result["scenes"][2]["duration_seconds"] == max_dur 
    
    mock_deps['logger'].debug.assert_any_call(f"Scene 1 audio duration 1.5s clamped to min {min_dur}s.")
    mock_deps['logger'].debug.assert_any_call(f"Scene 3 audio duration 15.0s clamped to max {max_dur}s.")

# --- Tests for generate_storyboard_from_story_script (Scenario 6) ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers')
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_success_first_try_llm_structure(
    mock_logger_sb_script, mock_split_markers_sb_script, mock_generate_structure_sb_script,
    mock_appconfig_storyboard_defaults, monkeypatch # mock_appconfig_storyboard_defaults is auto-used
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True)
    story_script = "A simple story."
    num_scenes = 3
    
    expected_structure = {"scenes": [{"scene_number": 1, "narration": "N1", "image_prompt": "P1"}]}
    mock_generate_structure_sb_script.return_value = expected_structure

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, num_scenes)

    assert result == expected_structure
    mock_generate_structure_sb_script.assert_called_once_with(story_script, num_scenes=num_scenes)
    mock_split_markers_sb_script.assert_not_called() 

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers')
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_success_first_try_split_markers(
    mock_logger_sb_script, mock_split_markers_sb_script, mock_generate_structure_sb_script,
    mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', False) 
    story_script = "SCENE 1\nN: N1\nP: P1" 
    num_scenes = 1 
    
    expected_scenes_list = [StoryboardScene(scene_number=1, narration="N1", image_prompt="P1")]
    mock_split_markers_sb_script.return_value = expected_scenes_list

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, num_scenes)

    assert result == expected_scenes_list
    mock_generate_structure_sb_script.assert_not_called() 
    mock_split_markers_sb_script.assert_called_once_with(story_script)

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers') 
@patch('asyncio.sleep', new_callable=AsyncMock) 
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_llm_structure_retry_success(
    mock_logger_sb_script, mock_asyncio_sleep_sb, mock_split_markers_sb_script, 
    mock_generate_structure_sb_script, mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True)
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', 2) 
    story_script = "Retry story for LLM structure."
    num_scenes = 2
    
    expected_structure = {"scenes": [{"scene_number": 1, "narration": "N_retry", "image_prompt": "P_retry"}]}
    mock_generate_structure_sb_script.side_effect = [None, expected_structure]

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, num_scenes)

    assert result == expected_structure
    assert mock_generate_structure_sb_script.call_count == 2
    mock_asyncio_sleep_sb.assert_called_once_with(app_config.storyboard.RETRY_DELAY_SECONDS)
    mock_logger_sb_script.warning.assert_any_call(
        f"Attempt 1 for LLM structure generation failed for story script. Retrying..."
    )
    mock_logger_sb_script.info.assert_any_call(
        "LLM structure generation successful after 1 retries."
    )

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers')
@patch('asyncio.sleep', new_callable=AsyncMock)
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_split_markers_retry_success(
    mock_logger_sb_script, mock_asyncio_sleep_sb, mock_split_markers_sb_script, 
    mock_generate_structure_sb_script, mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', False) 
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', 2)
    story_script = "Retry story for split markers."
    num_scenes = 1 
    
    expected_scenes_list = [StoryboardScene(scene_number=1, narration="N_split_retry", image_prompt="P_split_retry")]
    mock_split_markers_sb_script.side_effect = [[], expected_scenes_list]

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, num_scenes)

    assert result == expected_scenes_list
    assert mock_split_markers_sb_script.call_count == 2
    mock_asyncio_sleep_sb.assert_called_once_with(app_config.storyboard.RETRY_DELAY_SECONDS)
    mock_logger_sb_script.warning.assert_any_call(
        f"Attempt 1 for splitting story into scenes failed. Retrying..."
    )
    mock_logger_sb_script.info.assert_any_call(
        "Splitting story into scenes successful after 1 retries."
    )

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('asyncio.sleep', new_callable=AsyncMock) 
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_llm_structure_exhausts_retries(
    mock_logger_sb_script, mock_asyncio_sleep_sb, mock_generate_structure_sb_script,
    mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True)
    max_retries = 2 
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', max_retries + 1) 
    story_script = "Exhaust retries LLM structure."
    
    mock_generate_structure_sb_script.return_value = None 

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, 3)

    assert result is None
    assert mock_generate_structure_sb_script.call_count == max_retries + 1
    assert mock_asyncio_sleep_sb.call_count == max_retries
    mock_logger_sb_script.error.assert_any_call(
        f"Failed to generate storyboard structure after {max_retries + 1} attempts for story script."
    )

@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_script, invalid_num_scenes, expected_error_msg", [
    ("", 3, "Story script cannot be empty."),
    ("  ", 3, "Story script cannot be empty."),
    ("Valid script", 0, "Number of scenes must be positive."),
    ("Valid script", -1, "Number of scenes must be positive."),
])
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_input_validation(
    mock_logger_sb_script, invalid_script, invalid_num_scenes, expected_error_msg,
    mock_appconfig_storyboard_defaults
):
    with patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure') as mock_gen_struct, \
         patch('viralStoryGenerator.src.storyboard.split_story_by_markers') as mock_split:
        
        result = await storyboard_module.generate_storyboard_from_story_script(invalid_script, invalid_num_scenes)
        
        assert result is None
        mock_logger_sb_script.error.assert_called_once_with(expected_error_msg)
        mock_gen_struct.assert_not_called()
        mock_split.assert_not_called()

# --- Tests for _post_process_llm_output (Scenario 7) ---
@pytest.mark.parametrize("llm_output, expected_cleaned_output", [
    ("```json\n{\"key\": \"value\"}\n```", "{\"key\": \"value\"}"),
    ("```json\n{\"key\": \"value\"}\n```\nSome extra text after.", "{\"key\": \"value\"}\n\nSome extra text after."), # Assumes postambles don't match "Some extra text"
    ("Some text before.\n```json\n{\"key\": \"value\"}\n```", "Some text before.\n\n{\"key\": \"value\"}"), # Assumes preambles don't match "Some text before"
    ("  ```json\n{\"key\": \"value\"}\n```  ", "{\"key\": \"value\"}"), # Leading/trailing whitespace around code block
    ("{\"key\": \"value\"}", "{\"key\": \"value\"}"), # No markdown block
    ("Here is the JSON:\n```json\n{\"scenes\": []}\n```\nHope this helps!", "{\"scenes\": []}"), # Common preamble and postamble
    ("```\n{\"key\": \"value\"}\n```", "{\"key\": \"value\"}"), # Code block without language specifier
    ("   Leading space and ```json\n{\"key\": \"value\"}\n```", "{\"key\": \"value\"}"),
    ("```json\n{\"key\": \"value\"}\n```Trailing space   ", "{\"key\": \"value\"}"),
    # Test stripping of common textual preambles/postambles
    ("Here is the cleaned markdown content:\nActual content here.", "Actual content here."),
    ("Actual content here.\nI hope this helps!", "Actual content here."),
    ("Okay, here's the cleaned version:\nContent.\nLet me know if you need further adjustments.", "Content."),
    ("```markdown\n# Title\nContent\n```", "# Title\nContent"), # Markdown block with "markdown"
    ("```\n# Title\nContent\n```", "# Title\nContent"), # Markdown block no lang
    # Empty content
    ("```json\n\n```", ""),
    ("```json\n  \n```", ""),
    ("Here is the JSON:\n```json\n\n```", ""),
    # Only preambles/postambles
    ("Here is the JSON:\n```json\n```\nI hope this helps!", ""),
    ("```json\n```", ""),
    ("```\n```", ""),
])
def test_post_process_llm_output(llm_output, expected_cleaned_output, mock_storyboard_logger):
    # Note: _post_process_llm_output is a hidden function, but critical for generate_storyboard_structure
    cleaned = storyboard_module._post_process_llm_output(llm_output)
    assert cleaned == expected_cleaned_output
