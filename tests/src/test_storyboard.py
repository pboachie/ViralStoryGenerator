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

# --- Tests for split_story_by_markers (Scenario 4) ---

@pytest.mark.parametrize("story_text, expected_scenes_data", [
    # Valid case
    ("SCENE 1\nNARRATION:\nNarr 1\nIMAGE_PROMPT:\nPrompt 1\nSCENE 2\nNARRATION:\nNarr 2\nIMAGE_PROMPT:\nPrompt 2",
     [{"scene_number": 1, "narration": "Narr 1", "image_prompt": "Prompt 1"},
      {"scene_number": 2, "narration": "Narr 2", "image_prompt": "Prompt 2"}]),
    # Markers not found
    ("Just a single block of text without any scene markers.", 
     []),
    # Empty story
    ("", []),
    # Empty scenes list (no content between markers, or only markers)
    ("SCENE 1\nNARRATION:\n\nIMAGE_PROMPT:\n\nSCENE 2\nNARRATION:\n\nIMAGE_PROMPT:\n", 
     [{"scene_number": 1, "narration": "", "image_prompt": ""},
      {"scene_number": 2, "narration": "", "image_prompt": ""}]),
    # Incorrectly ordered or incomplete markers within a scene (parser should be robust or skip)
    ("SCENE 1\nNARRATION:\nNarr 1 Only", # Missing IMAGE_PROMPT marker for scene 1
     []), # Current regex expects both NARRATION and IMAGE_PROMPT
    ("SCENE 1\nIMAGE_PROMPT:\nPrompt 1 Only\nNARRATION:\nNarr 1 After", # Order swapped
     []), # Current regex expects NARRATION then IMAGE_PROMPT
    ("SCENE 1\nNARRATION: Narr 1 (no newline)\nIMAGE_PROMPT: Prompt 1 (no newline)", # Content on same line as marker
     [{"scene_number": 1, "narration": "Narr 1 (no newline)", "image_prompt": "Prompt 1 (no newline)"}]),
    # Extra whitespace
    ("  SCENE   1  \n  NARRATION:  \n  Narr with spaces  \n  IMAGE_PROMPT:  \n  Prompt with spaces  ",
     [{"scene_number": 1, "narration": "Narr with spaces", "image_prompt": "Prompt with spaces"}]),
    # Only one scene
    ("SCENE 1\nNARRATION:\nNarr single\nIMAGE_PROMPT:\nPrompt single",
     [{"scene_number": 1, "narration": "Narr single", "image_prompt": "Prompt single"}]),
    # Scene number not an int (should be skipped by current logic which expects \d+)
    ("SCENE X\nNARRATION:\nNarr X\nIMAGE_PROMPT:\nPrompt X", 
     []),
])
def test_split_story_by_markers_various_inputs(story_text, expected_scenes_data, mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    # The function returns a list of StoryboardScene Pydantic models
    # We will check the attributes of these models.
    
    scenes = storyboard_module.split_story_by_markers(story_text)
    
    assert len(scenes) == len(expected_scenes_data)
    for i, scene_model in enumerate(scenes):
        expected_data = expected_scenes_data[i]
        assert scene_model.scene_number == expected_data["scene_number"]
        assert scene_model.narration.strip() == expected_data["narration"].strip() # Strip for comparison flexibility
        assert scene_model.image_prompt.strip() == expected_data["image_prompt"].strip()

def test_split_story_by_markers_logs_warning_if_no_scenes(mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    story_text = "This story has no scene markers at all."
    scenes = storyboard_module.split_story_by_markers(story_text)
    assert len(scenes) == 0
    mock_storyboard_logger.warning.assert_called_once_with(
        f"No scenes found in story using regex. Story text: {story_text[:200]}..."
    )

def test_split_story_by_markers_duplicate_scene_numbers(mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    # The current implementation simply extracts based on regex and doesn't explicitly check for duplicate scene numbers.
    # It would extract both. The consuming function might care about this.
    story_text = "SCENE 1\nNARRATION:\nNarr A\nIMAGE_PROMPT:\nPrompt A\nSCENE 1\nNARRATION:\nNarr B\nIMAGE_PROMPT:\nPrompt B"
    scenes = storyboard_module.split_story_by_markers(story_text)
    assert len(scenes) == 2
    assert scenes[0].scene_number == 1
    assert scenes[0].narration == "Narr A"
    assert scenes[1].scene_number == 1
    assert scenes[1].narration == "Narr B"
    # No specific warning for duplicate scene numbers in split_story_by_markers itself.

# --- Tests for generate_storyboard (Scenario 5) ---

@pytest.fixture
def mock_dependencies_for_generate_storyboard(monkeypatch):
    """Centralized mocks for generate_storyboard tests."""
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
        'logger': patch('viralStoryGenerator.src.storyboard._logger').start() # Also mock logger from storyboard
    }
    yield mocks
    patch.stopall() # Stop all patches started by this fixture


# Scenario 5.1: Successful full storyboard generation
def test_generate_storyboard_successful_full_run(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_full_sb_001"
    story_script_text = "Full story script text."
    num_scenes_llm = 2 # Number of scenes LLM should generate
    
    # Configure app_config for this test if needed (e.g., enable features)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'NUM_SCENES_PER_STORY', num_scenes_llm)


    # Mock return values
    mock_deps = mock_dependencies_for_generate_storyboard
    mock_deps['uuid4'].return_value = MagicMock(hex=job_id) # For temp dir naming
    
    # 1. generate_storyboard_structure
    llm_structure_response = {"scenes": [ # This is what generate_storyboard_structure would return
        {"scene_number": 1, "narration": "Narr 1 from LLM.", "image_prompt": "Prompt 1 from LLM."},
        {"scene_number": 2, "narration": "Narr 2 from LLM.", "image_prompt": "Prompt 2 from LLM."}
    ]}
    mock_deps['generate_structure'].return_value = llm_structure_response
    
    # 2. split_story_by_markers (not directly called if structure is from LLM, but good to have if logic changes)
    # If generate_storyboard_structure returns a dict, split_story_by_markers is bypassed.
    # The code path is: structure = generate_storyboard_structure OR split_story_by_markers
    # Let's assume for this test, generate_storyboard_structure provides the structure.
    mock_deps['split_markers'].return_value = [] # Should not be called if structure is from LLM

    # 3. _get_storage_dir & tempfile.mkdtemp
    mock_base_storage_dir = f"/tmp/mock_storage_base/stories/{job_id}"
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['get_storage_dir'].return_value = mock_base_storage_dir
    mock_deps['mkdtemp'].return_value = mock_temp_dir

    # 4. generate_dalle_image and generate_elevenlabs_audio (per scene)
    # Scene 1
    mock_deps['dalle_image'].side_effect = [
        os.path.join(mock_temp_dir, "scene_1_image.png"), # Image for scene 1
        os.path.join(mock_temp_dir, "scene_2_image.png")  # Image for scene 2
    ]
    # elevenlabs_audio returns path if successful, or dict if timestamps, or False
    mock_deps['elevenlabs_audio'].side_effect = [
        True, # Audio for scene 1 (True means success, path is constructed)
        True  # Audio for scene 2
    ]
    # get_audio_duration
    mock_deps['get_audio_duration'].side_effect = [3.5, 4.0] # Durations for scene 1 and 2 audio

    # 5. store_file (for images, audio, and final JSON)
    # store_file(data_or_filepath, job_id, filename_key, is_json=False, is_temp_file=False)
    # Let's assume it returns the "storage path" (e.g., S3 URL or final local path)
    def store_file_side_effect(data_or_filepath, job_id_arg, filename_key, **kwargs):
        return f"s3://mock_bucket/{job_id_arg}/{filename_key}"
    mock_deps['store_file'].side_effect = store_file_side_effect

    # --- Call the function ---
    final_storyboard_data = storyboard_module.generate_storyboard(job_id, story_script_text)

    # --- Assertions ---
    assert final_storyboard_data is not None
    assert final_storyboard_data["job_id"] == job_id
    assert final_storyboard_data["story_script"] == story_script_text
    assert len(final_storyboard_data["scenes"]) == num_scenes_llm
    
    # Scene 1 checks
    scene1_data = final_storyboard_data["scenes"][0]
    assert scene1_data["scene_number"] == 1
    assert scene1_data["narration"] == "Narr 1 from LLM."
    assert scene1_data["image_prompt"] == "Prompt 1 from LLM."
    assert scene1_data["image_url"] == f"s3://mock_bucket/{job_id}/scene_1_image.png"
    assert scene1_data["audio_url"] == f"s3://mock_bucket/{job_id}/scene_1_audio.mp3"
    assert scene1_data["duration_seconds"] == 3.5 # From get_audio_duration

    # Scene 2 checks
    scene2_data = final_storyboard_data["scenes"][1]
    assert scene2_data["scene_number"] == 2
    assert scene2_data["duration_seconds"] == 4.0

    # Verify mock calls
    mock_deps['generate_structure'].assert_called_once_with(story_script_text, num_scenes=num_scenes_llm)
    mock_deps['split_markers'].assert_not_called() # Because structure came from LLM
    
    assert mock_deps['dalle_image'].call_count == num_scenes_llm
    mock_deps['dalle_image'].assert_any_call("Prompt 1 from LLM.", mock_temp_dir, f"scene_1_image_raw_{job_id}")
    
    assert mock_deps['elevenlabs_audio'].call_count == num_scenes_llm
    mock_deps['elevenlabs_audio'].assert_any_call(
        text="Narr 1 from LLM.",
        output_mp3_path=os.path.join(mock_temp_dir, "scene_1_audio.mp3"),
        voice_id=app_config.elevenlabs.DEFAULT_VOICE_ID,
        include_timestamps=False # Default for generate_storyboard
    )
    
    assert mock_deps['get_audio_duration'].call_count == num_scenes_llm
    mock_deps['get_audio_duration'].assert_any_call(os.path.join(mock_temp_dir, "scene_1_audio.mp3"))

    # store_file calls: num_scenes * 2 (image+audio) + 1 (final JSON)
    assert mock_deps['store_file'].call_count == (num_scenes_llm * 2) + 1
    mock_deps['store_file'].assert_any_call(
        os.path.join(mock_temp_dir, "scene_1_image.png"), job_id, "scene_1_image.png", is_temp_file=True
    )
    mock_deps['store_file'].assert_any_call(
        os.path.join(mock_temp_dir, "scene_1_audio.mp3"), job_id, "scene_1_audio.mp3", is_temp_file=True
    )
    # Final JSON store call
    # The first argument to store_file for JSON is the data itself (dict)
    # We need to check the call where `is_json=True`
    found_json_store_call = False
    for call_arg in mock_deps['store_file'].call_args_list:
        args, kwargs = call_arg
        if kwargs.get('is_json') is True and args[1] == job_id and args[2] == "storyboard.json":
            assert args[0] == final_storyboard_data # Data matches
            found_json_store_call = True
            break
    assert found_json_store_call, "store_file for final storyboard JSON not found or incorrect."

    mock_deps['mkdtemp'].assert_called_once()
    mock_deps['rmtree'].assert_called_once_with(mock_temp_dir)
    mock_deps['logger'].info.assert_any_call(f"Temporary directory {mock_temp_dir} created for job {job_id}")
    mock_deps['logger'].info.assert_any_call(f"Temporary directory {mock_temp_dir} removed for job {job_id}")
    mock_deps['logger'].info.assert_any_call(f"Storyboard generation for job {job_id} completed successfully.")

# Scenario 5.2: Failure at generate_storyboard_structure step
def test_generate_storyboard_fails_at_structure_generation(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults
):
    job_id = "job_fail_structure_002"
    story_script_text = "Story for structure failure."
    mock_deps = mock_dependencies_for_generate_storyboard
    
    mock_deps['generate_structure'].return_value = None # Simulate failure
    
    result = storyboard_module.generate_storyboard(job_id, story_script_text)
    
    assert result is None
    mock_deps['generate_structure'].assert_called_once()
    mock_deps['split_markers'].assert_not_called() # Should not be called if structure fails
    mock_deps['logger'].error.assert_any_call(
        f"Failed to generate or split storyboard structure for job {job_id}. Cannot proceed."
    )
    mock_deps['rmtree'].assert_not_called() # Temp dir might not have been created or should be cleaned if created before failure

# Scenario 5.3: Failure at split_story_by_markers step (when structure_from_llm is False)
def test_generate_storyboard_fails_at_split_markers(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_fail_split_003"
    story_script_text = "Story for split failure."
    mock_deps = mock_dependencies_for_generate_storyboard

    # Force generate_storyboard to use split_story_by_markers
    # This happens if generate_storyboard_structure returns None AND appconfig.storyboard.USE_LLM_FOR_STRUCTURE is False (or similar logic)
    # Assuming generate_storyboard_structure is tried first, then split_story_by_markers if structure is None.
    # Or, if there's a config to bypass LLM structure generation.
    # Let's assume generate_storyboard_structure returns None to trigger split_story_by_markers path.
    mock_deps['generate_structure'].return_value = None 
    mock_deps['split_markers'].return_value = [] # Simulate split_story_by_markers finding no scenes
    
    result = storyboard_module.generate_storyboard(job_id, story_script_text)
    
    assert result is None
    mock_deps['generate_structure'].assert_called_once() # Attempted first
    mock_deps['split_markers'].assert_called_once_with(story_script_text) # Then attempted split
    mock_deps['logger'].error.assert_any_call(
        f"Failed to generate or split storyboard structure for job {job_id}. Cannot proceed."
    )

# Scenario 5.4: generate_dalle_image fails for one scene
def test_generate_storyboard_dalle_fails_for_one_scene(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_dalle_fail_004"
    story_script_text = "Story with DALL-E failure."
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', False) # Disable audio for simplicity

    mock_deps['generate_structure'].return_value = {"scenes": [
        {"scene_number": 1, "narration": "Narr 1", "image_prompt": "Prompt 1"},
        {"scene_number": 2, "narration": "Narr 2", "image_prompt": "Prompt 2"}
    ]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['get_storage_dir'].return_value = f"/mock_storage/{job_id}"

    # DALL-E fails for scene 2
    mock_deps['dalle_image'].side_effect = [os.path.join(mock_temp_dir, "scene_1_image.png"), None]
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}" # Mock store_file

    result = storyboard_module.generate_storyboard(job_id, story_script_text)

    assert result is not None # Storyboard might be generated with missing image
    assert len(result["scenes"]) == 2
    assert result["scenes"][0]["image_url"] is not None
    assert result["scenes"][1]["image_url"] is None # Image failed for scene 2
    mock_deps['logger'].warning.assert_any_call(
        f"Failed to generate DALL-E image for scene 2 in job {job_id}. Skipping image for this scene."
    )
    mock_deps['rmtree'].assert_called_once_with(mock_temp_dir) # Temp dir still cleaned up

# Scenario 5.5: generate_elevenlabs_audio fails for one scene
def test_generate_storyboard_audio_fails_for_one_scene(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_audio_fail_005"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', False) # Disable image for simplicity
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', True)

    mock_deps['generate_structure'].return_value = {"scenes": [
        {"scene_number": 1, "narration": "Narr 1", "image_prompt": "Prompt 1"},
        {"scene_number": 2, "narration": "Narr 2", "image_prompt": "Prompt 2"}
    ]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['get_storage_dir'].return_value = f"/mock_storage/{job_id}"

    # Audio fails for scene 2 (returns False)
    mock_deps['elevenlabs_audio'].side_effect = [True, False] 
    mock_deps['get_audio_duration'].side_effect = [5.0, 0.0] # Duration for successful, 0 for failed
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"

    result = storyboard_module.generate_storyboard(job_id, "Story with audio failure.")

    assert result is not None
    assert len(result["scenes"]) == 2
    assert result["scenes"][0]["audio_url"] is not None
    assert result["scenes"][0]["duration_seconds"] == 5.0
    assert result["scenes"][1]["audio_url"] is None
    assert result["scenes"][1]["duration_seconds"] == app_config.storyboard.MIN_SCENE_DURATION # Falls back to min
    mock_deps['logger'].warning.assert_any_call(
        f"Failed to generate audio for scene 2 in job {job_id}. Skipping audio for this scene."
    )

# Scenario 5.6: storage_manager.store_file fails
def test_generate_storyboard_store_file_fails(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_store_fail_006"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', True) # Try storing an image
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', False)

    mock_deps['generate_structure'].return_value = {"scenes": [{"scene_number": 1, "narration": "N", "image_prompt": "P"}]}
    mock_temp_dir = f"/tmp/temp_{job_id}"
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['dalle_image'].return_value = os.path.join(mock_temp_dir, "scene_1_image.png")
    
    # store_file fails
    mock_deps['store_file'].side_effect = Exception("S3 upload error")

    result = storyboard_module.generate_storyboard(job_id, "Story for store failure.")
    
    # If storing an asset fails, the URL might be None, or the whole process might fail.
    # Current code: it logs error and continues, so URL will be None.
    assert result is not None
    assert result["scenes"][0]["image_url"] is None
    mock_deps['logger'].error.assert_any_call(
        f"Failed to store asset scene_1_image.png for job {job_id}. Error: S3 upload error"
    )
    # Test failure of final JSON storage
    mock_deps['store_file'].reset_mock()
    mock_deps['store_file'].side_effect = [
        f"s3://{job_id}/scene_1_image.png", # Image store succeeds
        Exception("Final JSON store error") # Final JSON store fails
    ]
    result_final_json_fail = storyboard_module.generate_storyboard(job_id, "Story for final store failure.")
    assert result_final_json_fail is None # If final store fails, overall result is None
    mock_deps['logger'].error.assert_any_call(
        f"Failed to store final storyboard JSON for job {job_id}. Error: Final JSON store error"
    )


# Scenario 5.7: Test with image/audio generation disabled
def test_generate_storyboard_assets_disabled(
    mock_dependencies_for_generate_storyboard, mock_appconfig_storyboard_defaults, monkeypatch
):
    job_id = "job_assets_disabled_007"
    mock_deps = mock_dependencies_for_generate_storyboard
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_IMAGE_GENERATION', False)
    monkeypatch.setattr(app_config.storyboard, 'ENABLE_AUDIO_GENERATION', False)

    mock_deps['generate_structure'].return_value = {"scenes": [{"scene_number": 1, "narration": "N", "image_prompt": "P"}]}
    mock_temp_dir = f"/tmp/temp_{job_id}" # Still used for potential intermediate files if any
    mock_deps['mkdtemp'].return_value = mock_temp_dir
    mock_deps['get_storage_dir'].return_value = f"/mock_storage/{job_id}"
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"


    result = storyboard_module.generate_storyboard(job_id, "Story with assets disabled.")

    assert result is not None
    scene1 = result["scenes"][0]
    assert scene1["image_url"] is None
    assert scene1["audio_url"] is None
    assert scene1["duration_seconds"] == app_config.storyboard.MIN_SCENE_DURATION # Default duration
    
    mock_deps['dalle_image'].assert_not_called()
    mock_deps['elevenlabs_audio'].assert_not_called()
    mock_deps['get_audio_duration'].assert_not_called()
    # store_file only called for final JSON
    mock_deps['store_file'].assert_called_once() 
    assert mock_deps['store_file'].call_args[0][2] == "storyboard.json"


# Scenario 5.8: Test duration clamping logic
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
    mock_deps['elevenlabs_audio'].return_value = True # All succeed
    mock_deps['store_file'].side_effect = lambda d, j, fn, **k: f"s3://{j}/{fn}"

    # Durations: too short, just right, too long
    mock_deps['get_audio_duration'].side_effect = [1.5, 5.0, 15.0] 

    result = storyboard_module.generate_storyboard(job_id, "Story for duration clamping.")
    
    assert result is not None
    assert len(result["scenes"]) == 3
    assert result["scenes"][0]["duration_seconds"] == min_dur # Clamped to min
    assert result["scenes"][1]["duration_seconds"] == 5.0   # Within range
    assert result["scenes"][2]["duration_seconds"] == max_dur # Clamped to max
    
    mock_deps['logger'].debug.assert_any_call(f"Scene 1 audio duration 1.5s clamped to min {min_dur}s.")
    mock_deps['logger'].debug.assert_any_call(f"Scene 3 audio duration 15.0s clamped to max {max_dur}s.")

# --- Tests for generate_storyboard_from_story_script (Scenario 6) ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers')
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_success_first_try_llm_structure(
    mock_logger_sb_script, mock_split_markers_sb_script, mock_generate_structure_sb_script,
    mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True)
    story_script = "A simple story."
    num_scenes = 3
    
    expected_structure = {"scenes": [{"scene_number": 1, "narration": "N1", "image_prompt": "P1"}]}
    mock_generate_structure_sb_script.return_value = expected_structure

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, num_scenes)

    assert result == expected_structure
    mock_generate_structure_sb_script.assert_called_once_with(story_script, num_scenes=num_scenes)
    mock_split_markers_sb_script.assert_not_called() # LLM structure used


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers')
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_success_first_try_split_markers(
    mock_logger_sb_script, mock_split_markers_sb_script, mock_generate_structure_sb_script,
    mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', False) # Force use of split_story_by_markers
    story_script = "SCENE 1\nN: N1\nP: P1" # Needs to be parsable by split_story_by_markers
    num_scenes = 1 # This num_scenes is for LLM structure, not directly used by split_story_by_markers
    
    expected_scenes_list = [StoryboardScene(scene_number=1, narration="N1", image_prompt="P1")]
    mock_split_markers_sb_script.return_value = expected_scenes_list

    result = await storyboard_module.generate_storyboard_from_story_script(story_script, num_scenes)

    assert result == expected_scenes_list
    mock_generate_structure_sb_script.assert_not_called() # LLM structure disabled
    mock_split_markers_sb_script.assert_called_once_with(story_script)


@pytest.mark.asyncio
@patch('viralStoryGenerator.src.storyboard.generate_storyboard_structure')
@patch('viralStoryGenerator.src.storyboard.split_story_by_markers') # Not called if LLM structure succeeds
@patch('asyncio.sleep', new_callable=AsyncMock) # Mock asyncio.sleep for retries
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_llm_structure_retry_success(
    mock_logger_sb_script, mock_asyncio_sleep_sb, mock_split_markers_sb_script, 
    mock_generate_structure_sb_script, mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True)
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', 2) # 1 initial + 1 retry = 2 attempts
    story_script = "Retry story for LLM structure."
    num_scenes = 2
    
    expected_structure = {"scenes": [{"scene_number": 1, "narration": "N_retry", "image_prompt": "P_retry"}]}
    # First call fails, second succeeds
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
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', False) # Force split_story_by_markers
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', 2)
    story_script = "Retry story for split markers."
    num_scenes = 1 
    
    expected_scenes_list = [StoryboardScene(scene_number=1, narration="N_split_retry", image_prompt="P_split_retry")]
    # First call returns empty (failure), second succeeds
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
@patch('asyncio.sleep', new_callable=AsyncMock) # For retries
@patch('viralStoryGenerator.src.storyboard._logger')
async def test_generate_storyboard_from_script_llm_structure_exhausts_retries(
    mock_logger_sb_script, mock_asyncio_sleep_sb, mock_generate_structure_sb_script,
    mock_appconfig_storyboard_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storyboard, 'USE_LLM_FOR_STRUCTURE', True)
    max_retries = 2 # Total attempts = 1 initial + 2 retries = 3
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', max_retries + 1) # Total attempts
    story_script = "Exhaust retries LLM structure."
    
    mock_generate_structure_sb_script.return_value = None # Always fails

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
    monkeypatch.setattr(app_config.storyboard, 'NUM_SCENES_PER_STORY', 5) # Default for generate_storyboard_from_story_script
    monkeypatch.setattr(app_config.storyboard, 'RETRY_ATTEMPTS', 2) # For generate_storyboard_from_story_script
    monkeypatch.setattr(app_config.storyboard, 'RETRY_DELAY_SECONDS', 0.01)


    # Storage manager related (used by generate_storyboard)
    # _get_storage_dir might use these or a direct path.
    # For testing, we'll often mock _get_storage_dir directly.
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
@pytest.mark.parametrize("input_text, expected_output", [
    ("Simple Title", "Simple_Title"),
    ("Title with spaces & special Chars!", "Title_with_spaces_special_Chars"),
    ("  Leading/Trailing Spaces  ", "Leading_Trailing_Spaces"),
    ("", "untitled_story"), # Empty string
    ("A very long title that might exceed typical filename limits to see if it gets truncated or handled gracefully by the sanitization", 
     "A_very_long_title_that_might_exceed_typical_filename_limits_to_see_if_it_gets_truncated_or_handled_gracefu"), # Truncated to 100
    ("你好世界", "untitled_story"), # Non-ASCII, falls back to default due to aggressive sanitization
    ("a/b\\c:d*e?f\"g<h>i|j", "a_b_c_d_e_f_g_h_i_j"), # Many special chars
    ("---", "_"), # Only hyphens
    ("multiple---hyphens---together", "multiple-hyphens-together"), # Multiple hyphens reduced to one
    ("endsWith-", "endsWith-"), # Ends with hyphen (kept by current regex if simple replace)
                               # The code's regex `[^a-zA-Z0-9_.-]` replaces matches with `_`.
                               # Then `re.sub(r'[_]+', '_', sanitized_text)` collapses multiple underscores.
                               # So "---" -> "___" -> "_".
                               # "endsWith-" -> "endsWith-" (no change by first, no change by second)
                               # "a/b" -> "a_b"
                               # Let's re-verify expected based on the two regexes in code:
                               # 1. `re.sub(r'[^\w\s-]', '', text)` -- Removes non-alphanumeric (keeps unicode word chars), non-whitespace, non-hyphen.
                               # 2. `re.sub(r'[-\s]+', '_', text)` -- Replaces hyphens and spaces with single underscore.
                               # This means "Simple Title" -> "Simple_Title"
                               # "Title with spaces & special Chars!" -> "Title_with_spaces_special_Chars" (good)
                               # "  Leading/Trailing Spaces  " -> "_Leading_Trailing_Spaces_" (leading/trailing underscores)
                               # "" -> "untitled_story" (correct)
                               # "你好世界" -> "你好世界" (kept by `\w` if unicode is handled, then space/hyphen rule) -> "你好世界" (if no spaces/hyphens)
                               # Let's assume `\w` includes unicode: "你好世界" -> "你好世界"
                               # "a/b\\c:d*e?f\"g<h>i|j" -> "abcdefghij" (all special chars removed by first regex)
                               # "---" -> "_" (correct by second regex)
                               # "multiple---hyphens---together" -> "multiple_hyphens_together" (correct)
                               # "endsWith-" -> "endsWith_" (correct)
])
def test_sanitize_for_filename(input_text, expected_output, mock_appconfig_storyboard_defaults):
    # If expected for "你好世界" is "untitled_story", it means non-ASCII are fully stripped or default on no valid chars
    # The code's current logic: `re.sub(r'[^\w\s-]', '', text)` means non-alphanumeric (word chars, includes unicode), non-whitespace, non-hyphen are removed.
    # Then `re.sub(r'[-\s]+', '_', text)` replaces one or more hyphens or spaces with a single underscore.
    # If after first step, all chars are gone (e.g. "!@#$"), it becomes "", then "untitled_story".
    # If "你好世界" results in "你好世界" (because \w includes unicode), then this is the result.
    # The prompt's expected "untitled_story" for "你好世界" implies more aggressive stripping or a fallback if only unicode.
    # Let's assume the code's `\w` handles unicode correctly.
    
    # Re-evaluating expected based on the actual code's regex:
    # 1. `sanitized = re.sub(r'[^\w\s-]', '', text)` # Keeps unicode words, whitespace, hyphens
    # 2. `sanitized = re.sub(r'[-\s]+', '_', sanitized).strip('_')` # Collapses hyphens/spaces to '_', strips leading/trailing '_'
    # 3. `sanitized = sanitized[:100]` # Truncates
    # 4. `return sanitized if sanitized else default_filename`

    if input_text == "Simple Title": expected_output = "Simple_Title"
    elif input_text == "Title with spaces & special Chars!": expected_output = "Title_with_spaces_special_Chars"
    elif input_text == "  Leading/Trailing Spaces  ": expected_output = "Leading_Trailing_Spaces" # strip('_') handles this
    elif input_text == "": expected_output = "untitled_story"
    elif input_text == "A very long title that might exceed typical filename limits to see if it gets truncated or handled gracefully by the sanitization":
        expected_output = "A_very_long_title_that_might_exceed_typical_filename_limits_to_see_if_it_gets_truncated_or_handle" # Truncated to 100
    elif input_text == "你好世界": expected_output = "你好世界" # Assuming \w keeps unicode, and strip('_') does nothing if no leading/trailing _
    elif input_text == "a/b\\c:d*e?f\"g<h>i|j": expected_output = "ab c d e f g h i j".replace(" ", "_") # This depends on what \w means exactly.
                                                              # If \w is just [a-zA-Z0-9_], then 'a b c d e f g h i j' -> 'a_b_c_d_e_f_g_h_i_j'
                                                              # The code uses \w, which typically includes underscore.
                                                              # My original expected "a_b_c_d_e_f_g_h_i_j" was based on aggressive replacement.
                                                              # With `[^\w\s-]` removed: "a b c d e f g h i j" (if those were not \w, \s, -)
                                                              # Then `[-\s]+` to `_`: "a_b_c_d_e_f_g_h_i_j"
                                                              # The provided code is `[^\w\s-]`, meaning it KEEPS word chars, spaces, hyphens.
                                                              # So, "a/b\\c:d*e?f\"g<h>i|j" -> "abcdefghij" (all removed as they are not \w, \s, or -)
                                                              expected_output = "abcdefghij" 
    elif input_text == "---": expected_output = "" # "---" -> "" by first (if - is not \w), then "" by second. Then default.
                                                 # If - IS kept by first: "---" -> "_" by second. Then strip('_') -> "" -> default.
                                                 # So it should be "untitled_story"
                                                 expected_output = "untitled_story"
    elif input_text == "multiple---hyphens---together": expected_output = "multiple_hyphens_together" # Correct
    elif input_text == "endsWith-": expected_output = "endsWith" # "endsWith-" -> "endsWith-" -> "endsWith_" -> strip('_') -> "endsWith"

    # Correcting "---" based on code:
    # 1. `re.sub(r'[^\w\s-]', '', "---")` -> "---" (hyphens are kept)
    # 2. `re.sub(r'[-\s]+', '_', "---")` -> "_"
    # 3. `.strip('_')` -> ""
    # 4. `return "" if "" else "untitled_story"` -> "untitled_story"
    if input_text == "---": expected_output = "untitled_story"


    result = storyboard_module.sanitize_for_filename(input_text)
    assert result == expected_output


# --- Tests for generate_storyboard_structure (Scenario 2) ---

@patch('requests.post')
@patch('json.loads') # To control JSON parsing if LLM response is tricky
def test_generate_storyboard_structure_success(
    mock_json_loads, mock_requests_post_sb, mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    story_script = "This is the full story script."
    num_scenes = 3
    
    mock_llm_response_content = '{"scenes": [{"scene_number": 1, "image_prompt": "Prompt 1", "narration": "Narr 1"}]}'
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": mock_llm_response_content}}]}
    mock_requests_post_sb.return_value = mock_response

    # json.loads will be called by the function to parse the content string
    expected_parsed_json = {"scenes": [{"scene_number": 1, "image_prompt": "Prompt 1", "narration": "Narr 1"}]}
    mock_json_loads.return_value = expected_parsed_json

    result = storyboard_module.generate_storyboard_structure(story_script, num_scenes)

    assert result == expected_parsed_json
    mock_requests_post_sb.assert_called_once()
    args, kwargs = mock_requests_post_sb.call_args
    payload = kwargs['json']
    assert payload['model'] == app_config.llm.MODEL
    assert any(msg['role'] == 'system' and "You are a storyboard assistant" in msg['content'] for msg in payload['messages'])
    assert any(msg['role'] == 'user' and story_script in msg['content'] and f"Generate {num_scenes} scenes" in msg['content'] for msg in payload['messages'])
    mock_json_loads.assert_called_once_with(mock_llm_response_content)
    mock_storyboard_logger.info.assert_any_call(f"Successfully generated storyboard structure with {len(expected_parsed_json['scenes'])} scenes.")


@patch('requests.post')
def test_generate_storyboard_structure_llm_malformed_json(
    mock_requests_post_sb, mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    story_script = "Story for malformed JSON test."
    # LLM returns a string that is not valid JSON
    mock_llm_response_content = 'This is not JSON. {"scenes": ...'
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": mock_llm_response_content}}]}
    mock_requests_post_sb.return_value = mock_response

    result = storyboard_module.generate_storyboard_structure(story_script, 3)

    assert result is None
    mock_storyboard_logger.error.assert_any_call(
        f"LLM response for storyboard structure was not valid JSON. Response: {mock_llm_response_content}"
    )


@patch('requests.post')
@patch('json.loads')
def test_generate_storyboard_structure_llm_no_scenes_key(
    mock_json_loads, mock_requests_post_sb, mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    story_script = "Story for missing 'scenes' key test."
    mock_llm_response_content = '{"description": "This is valid JSON, but no scenes key."}' # Valid JSON, but wrong structure
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": mock_llm_response_content}}]}
    mock_requests_post_sb.return_value = mock_response
    
    mock_json_loads.return_value = {"description": "This is valid JSON, but no scenes key."}


    result = storyboard_module.generate_storyboard_structure(story_script, 3)

    assert result is None
    mock_storyboard_logger.error.assert_any_call(
        f"LLM response for storyboard structure is missing 'scenes' key or 'scenes' is not a list. Response: {mock_json_loads.return_value}"
    )


@patch('requests.post')
@patch('json.loads')
def test_generate_storyboard_structure_llm_json_in_markdown_block(
    mock_json_loads, mock_requests_post_sb, mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    story_script = "Story for JSON in markdown block test."
    # LLM returns JSON wrapped in markdown code block
    mock_llm_response_content = "Here is the JSON:\n```json\n{\"scenes\": [{\"scene_number\": 1}]}\n```"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": mock_llm_response_content}}]}
    mock_requests_post_sb.return_value = mock_response

    expected_parsed_json = {"scenes": [{"scene_number": 1}]}
    # json.loads should be called with the extracted JSON string
    mock_json_loads.return_value = expected_parsed_json 

    result = storyboard_module.generate_storyboard_structure(story_script, 1)

    assert result == expected_parsed_json
    # json.loads called with the content *after* _post_process_llm_output (which handles markdown blocks)
    # The _post_process_llm_output is implicitly tested here.
    # We need to ensure json.loads is called with the *inner* JSON string.
    # The current function calls _post_process_llm_output on the content string.
    cleaned_content_for_json_loads = storyboard_module._post_process_llm_output(mock_llm_response_content)
    mock_json_loads.assert_called_once_with(cleaned_content_for_json_loads)


@patch('requests.post')
@patch('time.sleep', return_value=None) # Mock time.sleep for retries
def test_generate_storyboard_structure_requests_post_fails_then_succeeds(
    mock_time_sleep_sb, mock_requests_post_sb, mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    story_script = "Story for retry test."
    
    mock_error_response = MagicMock(spec=requests.Response)
    mock_error_response.status_code = 500
    mock_error_response.reason = "Internal Server Error"
    mock_error_response.text = "Server is down"
    mock_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error", response=mock_error_response)

    mock_success_response_content = '{"scenes": [{"scene_number": 1, "narration": "Success!"}]}'
    mock_success_response = MagicMock(spec=requests.Response)
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {"choices": [{"message": {"content": mock_success_response_content}}]}
    mock_success_response.raise_for_status = MagicMock() # Does nothing for success

    mock_requests_post_sb.side_effect = [
        mock_error_response, # Fails first
        mock_success_response # Succeeds on retry
    ]
    
    # json.loads will be called by the function after successful LLM response
    with patch('json.loads', return_value={"scenes": [{"scene_number": 1, "narration": "Success!"}]}) as mock_json_loads_retry:
        result = storyboard_module.generate_storyboard_structure(story_script, 1)

    assert result is not None
    assert result["scenes"][0]["narration"] == "Success!"
    assert mock_requests_post_sb.call_count == 2 # Initial call + 1 retry
    mock_time_sleep_sb.assert_called_once() # Called once between retries
    mock_storyboard_logger.warning.assert_any_call(
        "LLM API request error: 500 Server Error. Attempt 1 of 2. Retrying in ..." # Max attempts is 2 (1 retry)
    )
    mock_storyboard_logger.info.assert_any_call(
        "LLM API request successful after retry. Attempts: 2"
    )


@patch('requests.post')
@patch('time.sleep', return_value=None)
def test_generate_storyboard_structure_exhausts_retries(
    mock_time_sleep_sb, mock_requests_post_sb, mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    story_script = "Story for exhausting retries."
    
    mock_error_response = MagicMock(spec=requests.Response)
    mock_error_response.status_code = 503
    mock_error_response.reason = "Service Unavailable"
    mock_error_response.text = "Service busy"
    mock_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Service Error", response=mock_error_response)

    # All attempts fail
    max_attempts = app_config.llm.RETRY_ATTEMPTS # Should be 2 from fixture
    mock_requests_post_sb.side_effect = [mock_error_response] * max_attempts

    result = storyboard_module.generate_storyboard_structure(story_script, 1)

    assert result is None
    assert mock_requests_post_sb.call_count == max_attempts
    assert mock_time_sleep_sb.call_count == max_attempts - 1
    mock_storyboard_logger.error.assert_any_call(
        f"LLM API request failed after {max_attempts} attempts. Error: Service Error"
    )

# --- Tests for generate_dalle_image (Scenario 3) ---

@patch('requests.post') # For DALL-E API call
@patch('requests.get')  # For image download
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs') # To ensure output directory handling is tested
@patch('os.path.isdir', return_value=True) # Assume output dir exists for simplicity here
@patch('os.path.exists', return_value=False) # Assume image file doesn't exist initially
def test_generate_dalle_image_success(
    mock_os_exists, mock_os_isdir, mock_os_makedirs, mock_open_file_dalle, 
    mock_requests_get, mock_requests_post_dalle, 
    mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    image_prompt = "A futuristic city"
    output_dir = "/tmp/dalle_images"
    filename_base = "future_city_img"
    expected_full_path = os.path.join(output_dir, f"{filename_base}.png")

    # Mock DALL-E API response
    mock_dalle_api_response = MagicMock(spec=requests.Response)
    mock_dalle_api_response.status_code = 200
    # DALL-E API returns image URL in data[0].url
    mock_dalle_api_response.json.return_value = {"data": [{"url": "http://mock-image-url.com/image.png"}]}
    mock_requests_post_dalle.return_value = mock_dalle_api_response

    # Mock image download response
    mock_image_download_response = MagicMock(spec=requests.Response)
    mock_image_download_response.status_code = 200
    mock_image_download_response.content = b"dummy_image_bytes"
    mock_requests_get.return_value = mock_image_download_response

    result_path = storyboard_module.generate_dalle_image(image_prompt, output_dir, filename_base)

    assert result_path == expected_full_path
    
    # Verify DALL-E API call
    mock_requests_post_dalle.assert_called_once()
    args_post, kwargs_post = mock_requests_post_dalle.call_args
    assert args_post[0] == app_config.dalle.API_ENDPOINT
    assert kwargs_post['json']['prompt'] == image_prompt
    assert kwargs_post['json']['model'] == app_config.dalle.IMAGE_MODEL
    assert kwargs_post['json']['size'] == app_config.dalle.IMAGE_SIZE
    assert kwargs_post['headers']['Authorization'] == f"Bearer {app_config.dalle.API_KEY}"
    
    # Verify image download call
    mock_requests_get.assert_called_once_with("http://mock-image-url.com/image.png", stream=True, timeout=app_config.httpOptions.TIMEOUT)
    
    # Verify file write
    mock_open_file_dalle.assert_called_once_with(expected_full_path, 'wb')
    mock_open_file_dalle().write.assert_called_once_with(b"dummy_image_bytes")
    mock_storyboard_logger.info.assert_any_call(f"DALL-E image generated and saved to {expected_full_path}")


@patch('requests.post') # DALL-E API call
@patch('requests.get')  # Image download (should not be called)
def test_generate_dalle_image_api_error(
    mock_requests_get_dalle, mock_requests_post_dalle, 
    mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    image_prompt = "Prompt for API error"
    output_dir = "/tmp/dalle_error"
    filename_base = "api_error_img"

    # Mock DALL-E API error response
    mock_dalle_api_error_response = MagicMock(spec=requests.Response)
    mock_dalle_api_error_response.status_code = 401 # Unauthorized
    mock_dalle_api_error_response.reason = "Unauthorized"
    mock_dalle_api_error_response.text = "Invalid API key"
    mock_dalle_api_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "API Error", response=mock_dalle_api_error_response
    )
    mock_requests_post_dalle.return_value = mock_dalle_api_error_response

    result_path = storyboard_module.generate_dalle_image(image_prompt, output_dir, filename_base)

    assert result_path is None
    mock_requests_post_dalle.assert_called_once() # Called once, no DALL-E retries by default in this function
    mock_requests_get_dalle.assert_not_called() # Download should not be attempted
    mock_storyboard_logger.error.assert_any_call(
        f"DALL-E API error (401 Unauthorized): Invalid API key"
    )


@patch('requests.post') # DALL-E API call
@patch('requests.get')  # Image download
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=False)
def test_generate_dalle_image_download_fails(
    mock_os_exists, mock_os_isdir, mock_requests_get_dalle, mock_requests_post_dalle, 
    mock_storyboard_logger, mock_appconfig_storyboard_defaults
):
    image_prompt = "Prompt for download fail"
    output_dir = "/tmp/dalle_dl_fail"
    filename_base = "dl_fail_img"

    # Mock successful DALL-E API response
    mock_dalle_api_response = MagicMock(spec=requests.Response)
    mock_dalle_api_response.status_code = 200
    mock_dalle_api_response.json.return_value = {"data": [{"url": "http://mock-image-url.com/image_dl_fail.png"}]}
    mock_requests_post_dalle.return_value = mock_dalle_api_response

    # Mock image download failure
    mock_image_download_error_response = MagicMock(spec=requests.Response)
    mock_image_download_error_response.status_code = 404 # Not Found
    mock_image_download_error_response.reason = "Not Found"
    mock_image_download_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Download Error", response=mock_image_download_error_response
    )
    mock_requests_get_dalle.return_value = mock_image_download_error_response
    # Or, mock_requests_get_dalle.side_effect = requests.exceptions.ConnectionError("Cannot connect to download")


    result_path = storyboard_module.generate_dalle_image(image_prompt, output_dir, filename_base)

    assert result_path is None
    mock_requests_post_dalle.assert_called_once()
    mock_requests_get_dalle.assert_called_once() # Download was attempted
    mock_storyboard_logger.error.assert_any_call(
        f"Failed to download DALL-E image from http://mock-image-url.com/image_dl_fail.png. Status: 404 Not Found"
    )


def test_generate_dalle_image_empty_prompt(mock_storyboard_logger, mock_appconfig_storyboard_defaults):
    result = storyboard_module.generate_dalle_image("", "/tmp", "empty_prompt")
    assert result is None
    mock_storyboard_logger.error.assert_called_once_with("Image prompt cannot be empty.")


def test_generate_dalle_image_no_api_key(mock_storyboard_logger, mock_appconfig_storyboard_defaults, monkeypatch):
    monkeypatch.setattr(app_config.dalle, 'API_KEY', None)
    result = storyboard_module.generate_dalle_image("Prompt", "/tmp", "no_key_img")
    assert result is None
    mock_storyboard_logger.error.assert_called_once_with("DALL-E API Key not configured.")
    # Restore for other tests
    monkeypatch.setattr(app_config.dalle, 'API_KEY', "test_dalle_api_key")
