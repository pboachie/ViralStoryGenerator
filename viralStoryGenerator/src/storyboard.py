# viralStoryGenerator/src/storyboard.py

import os
import json
import datetime
import requests
import re
import time
import tempfile
import shutil
from typing import Dict, Any, Optional, List

from viralStoryGenerator.src.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.prompts.prompts import get_storyboard_prompt
import logging
from viralStoryGenerator.utils.config import config as appconfig
from viralStoryGenerator.utils.security import is_safe_filename
from viralStoryGenerator.models.models import STORYBOARD_RESPONSE_FORMAT
from viralStoryGenerator.src.llm import _extract_chain_of_thought
from viralStoryGenerator.utils.storage_manager import storage_manager

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

APP_USER_AGENT = f"{appconfig.APP_TITLE}/{appconfig.VERSION}"

# Helper function to sanitize topic for use in filenames
def sanitize_for_filename(text: str, max_length: int = 100) -> str:
    """Removes unsafe characters and shortens text for use in filenames."""
    if not text:
        return "untitled"
    sanitized = re.sub(r'[\\/*?:"<>|\0]', '_', text)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('._ ')
    sanitized = sanitized[:max_length]
    return sanitized if sanitized else "sanitized_topic"

def generate_storyboard_structure(story: str, llm_endpoint: str, temperature: float) -> Optional[Dict[str, Any]]:
    """
    Uses the LLM (specifically the MODEL_MULTI) to produce a storyboard breakdown
    containing scene markers and image prompts in JSON format.
    Includes basic error handling and parsing.

    Returns:
        Parsed storyboard data (with scene_start_marker) as dict, or None on failure.
    """
    model = appconfig.llm.MODEL_MULTI
    if not story or not llm_endpoint or not model:
        _logger.error("Missing required arguments for storyboard structure generation (story, endpoint, or configured MODEL_MULTI).")
        return None

    prompt = get_storyboard_prompt(story).strip()
    _logger.info(f"Requesting storyboard structure (with markers) from LLM using model: {model}...")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that strictly follows instructions. Generate ONLY the requested structured JSON output. Do NOT include any other text, explanations, or reasoning before or after the JSON object."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": appconfig.llm.MAX_TOKENS,
        "stream": False,
        "response_format": STORYBOARD_RESPONSE_FORMAT
    }

    try:
        response = requests.post(
            llm_endpoint,
            headers=headers,
            json=data,
            timeout=appconfig.httpOptions.TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 400 and "response_format" in data:
            _logger.warning(f"LLM ({model}) returned 400 Bad Request with 'response_format', retrying without it.")
            data.pop("response_format", None)
            try:
                response = requests.post(llm_endpoint, headers=headers, json=data, timeout=appconfig.httpOptions.TIMEOUT)
                response.raise_for_status()
            except requests.exceptions.RequestException as retry_e:
                 _logger.error(f"LLM ({model}) retry request for storyboard failed: {retry_e}")
                 return None
        else:
            _logger.error(f"LLM ({model}) request for storyboard failed: {e}. Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
         _logger.error(f"LLM ({model}) request for storyboard timed out after {appconfig.httpOptions.TIMEOUT} seconds.")
         return None
    except requests.exceptions.RequestException as e:
        payload_str = json.dumps(data)
        _logger.error(f"Failed to generate storyboard structure from LLM ({model}): {e}. Request Payload: {payload_str[:500]}...", exc_info=True)
        return None

    try:
        response_data = response.json()
        if not response_data or "choices" not in response_data or not response_data["choices"]:
            _logger.error(f"LLM ({model}) response missing 'choices' field or empty. Response: {response.text[:200]}")
            return None

        raw_content = response_data["choices"][0]["message"]["content"]

        cleaned_content, thinking = _extract_chain_of_thought(raw_content)
        _logger.debug(f"Extracted thinking block: {thinking[:100]}...")

        json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if json_match:
            cleaned_content = json_match.group(0)
        else:
             _logger.warning(f"Could not find explicit JSON object in cleaned LLM message content. Raw (cleaned): {cleaned_content[:200]}...")

    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as extract_err:
         _logger.error(f"Error parsing LLM response structure or extracting content: {extract_err}. Response text: {response.text[:500]}...", exc_info=True)
         return None
    except Exception as text_extract_err:
         _logger.error(f"Unexpected error extracting/cleaning text from LLM response: {text_extract_err}. Response status: {response.status_code}", exc_info=True)
         return None

    try:
         storyboard_data = json.loads(cleaned_content)
         if "scenes" not in storyboard_data or not isinstance(storyboard_data["scenes"], list):
             raise ValueError("Missing or invalid 'scenes' list in LLM response.")
         if not storyboard_data["scenes"]:
              _logger.warning(f"LLM ({model}) returned an empty 'scenes' list.")
              return None # todo: or allow empty list to proceed
         for i, scene in enumerate(storyboard_data["scenes"]):
             if "scene_start_marker" not in scene or not isinstance(scene["scene_start_marker"], str) or not scene["scene_start_marker"].strip():
                 raise ValueError(f"Scene {i+1} is missing or has an invalid 'scene_start_marker'.")
             if "image_prompt" not in scene or not isinstance(scene["image_prompt"], str):
                 if "image_prompt" not in scene:
                     _logger.warning(f"Scene {i+1} is missing 'image_prompt'.")
                 elif not isinstance(scene["image_prompt"], str):
                      _logger.warning(f"Scene {i+1} has non-string 'image_prompt': {type(scene['image_prompt'])}. Setting to empty.")
                      scene["image_prompt"] = "" # Attempt recovery
             scene.setdefault("duration", 0)
             scene.setdefault("start_time", 0)

         _logger.info(f"Successfully parsed storyboard structure with {len(storyboard_data['scenes'])} scenes (using markers).")
         return storyboard_data
    except (json.JSONDecodeError, ValueError) as e:
         _logger.error(f"JSON decode/validation error after cleaning (model: {model}): {e}. Cleaned content: {cleaned_content[:500]}...")
         return None

def generate_dalle_image(image_prompt: str, output_image_path: str, openai_api_key: str) -> Optional[str]:
    """
    Uses the DALL·E 3 API to generate an image based on the given image_prompt.
    Downloads and saves the image to output_image_path.

    Args:
        image_prompt: Text description for DALL-E.
        output_image_path: Full path to save the generated PNG image.
        openai_api_key: OpenAI API key.

    Returns:
        The output_image_path if successful, None otherwise.
    """
    if not image_prompt:
        _logger.warning("Cannot generate DALL-E image: image_prompt is empty.")
        return None
    if not openai_api_key:
        _logger.error("Cannot generate DALL-E image: Missing OpenAI API key.")
        return None

    _logger.info(f"Requesting DALL-E image generation. Output: {output_image_path}")
    _logger.debug(f"DALL-E Prompt (potential injection risk): {image_prompt[:100]}...")

    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
    }
    payload = {
        # "prompt": f"Generate a safe image depicting the following scene for a video storyboard: {image_prompt}",
        "prompt": image_prompt,
        "model": "dall-e-3", # Explicitly use DALL-E 3 if available
        "n": 1,
        "size": "1024x1024",
        "quality": "standard", # Use "hd" for higher quality
        "style": "vivid" # or "natural"
    }
    dalle_timeout = 60

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=dalle_timeout)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data["data"][0]["url"]

        image_download_timeout = 60
        image_response = requests.get(image_url, timeout=image_download_timeout, stream=True)
        image_response.raise_for_status()

        # Save the image securely
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with open(output_image_path, "wb") as f:
             for chunk in image_response.iter_content(chunk_size=8192):
                 f.write(chunk)

        _logger.info(f"DALL-E image saved successfully to {output_image_path}")
        return output_image_path

    except requests.exceptions.Timeout as e:
         _logger.error(f"DALL-E API call timed out after {dalle_timeout}s (or download timeout): {e}")
         return None
    except requests.exceptions.RequestException as e:
        error_details = ""
        if hasattr(e, 'response') and e.response is not None:
            error_details = e.response.text[:200]
        _logger.error(f"DALL-E API call failed: {e}. Details: {error_details}")
        return None
    except (KeyError, IndexError, TypeError) as e:
         _logger.error(f"Failed processing DALL-E response or downloading image: {e}. Response data: {response_data if 'response_data' in locals() else 'N/A'}")
         return None
    except IOError as e:
         _logger.error(f"Failed to save DALL-E image to {output_image_path}: {e}")
         return None


def generate_storyboard(story: str, topic: str, task_id: str, llm_endpoint: str, temperature: float, voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Generates a full storyboard from a story script using scene markers.

    Steps:
    1. Generate scene structure (markers, prompts) using LLM (MODEL_MULTI).
    2. Split the original story into scene narrations using the markers.
    3. Determine base directory and filename using task_id.
    4. For each scene:
        - Assign the split narration text.
        - Calculate duration based on the narration text.
        - Generate DALL-E image using the prompt (if enabled).
    5. Combine narration texts.
    6. Generate a single audio file for the combined narration using ElevenLabs (if enabled).
    7. Save the complete storyboard JSON data using storage_manager.

    Args:
        story: The full story script text.
        topic: The original topic (used for logging/context, not path).
        task_id: The unique ID for this generation task.
        llm_endpoint: LLM API endpoint.
        temperature: LLM temperature setting.
        voice_id: Optional ElevenLabs voice ID.

    Returns:
        The complete storyboard data dictionary if successful, None otherwise.
    """

    if not appconfig.storyboard.ENABLE_STORYBOARD_GENERATION:
        _logger.info(f"Storyboard generation is disabled for task: '{task_id}'. Skipping storyboard generation.")
        return None

    _logger.info(f"Starting storyboard generation for task: '{task_id}' (Topic: '{topic}') using marker-based splitting.")

    storyboard_data = generate_storyboard_structure(story, llm_endpoint, temperature)
    if not storyboard_data or "scenes" not in storyboard_data or not storyboard_data["scenes"]:
        _logger.error(f"Storyboard structure generation (with markers) failed or returned empty scenes for task {task_id}.")
        return None

    scenes = storyboard_data["scenes"]
    scene_narrations = split_story_by_markers(story, scenes)
    if not scene_narrations or len(scene_narrations) != len(scenes):
        _logger.error(f"Failed to split story accurately based on markers for task {task_id}. Aborting storyboard generation.")
        # Optionally save the faulty structure for debugging
        # storage_manager.store_file(...)
        return None

    if not task_id:
        _logger.error("Task ID is missing, cannot generate storyboard filenames.")
        return None
    base_filename = task_id
    storyboard_base_dir = storage_manager._get_storage_dir("storyboard")
    try:
        os.makedirs(storyboard_base_dir, exist_ok=True)
        _logger.info(f"Ensured storyboard base directory exists: {storyboard_base_dir}")
    except OSError as e:
        _logger.error(f"Failed to ensure storyboard base directory '{storyboard_base_dir}': {e}")
        return None

    openai_api_key = appconfig.openAI.API_KEY
    elevenlabs_api_key = appconfig.elevenLabs.API_KEY
    image_generation_enabled = appconfig.openAI.ENABLED
    audio_generation_enabled = appconfig.elevenLabs.ENABLED

    if not appconfig.ENABLE_IMAGE_GENERATION:
        _logger.info("Image generation is disabled. Skipping image generation for storyboard.")
        image_generation_enabled = False

    if not appconfig.ENABLE_AUDIO_GENERATION:
        _logger.info("Audio generation is disabled. Skipping audio generation for storyboard.")
        audio_generation_enabled = False

    cumulative_time = 0.0
    combined_narration_texts = []
    word_per_minute_rate = 150
    image_results = []

    # --- Start Change ---
    # Define desired duration range
    MIN_SCENE_DURATION = 3.0
    MAX_SCENE_DURATION = 5.0
    # --- End Change ---

    # Process each scene (assign text, calculate duration, generate image)
    for i, scene in enumerate(scenes):
        scene_number = scene.get("scene_number", i + 1)
        scene["scene_number"] = scene_number

        narration_text = scene_narrations[i]
        scene["narration_text"] = narration_text
        combined_narration_texts.append(narration_text)

        image_prompt = scene.get("image_prompt", "").strip()
        duration = scene.get("duration")

        # --- Start Change ---
        # Calculate or use provided duration, then clamp it
        calculated_duration = 0.0
        if not isinstance(duration, (int, float)) or duration <= 0:
             words = len(narration_text.split())
             estimated_duration = round((words / word_per_minute_rate) * 60.0, 1) if words > 0 else MIN_SCENE_DURATION
             calculated_duration = max(MIN_SCENE_DURATION, estimated_duration)
             _logger.debug(f"Task {task_id}, Scene {scene_number}: Estimated duration {estimated_duration}s for {words} words.")
        else:
             calculated_duration = max(MIN_SCENE_DURATION, float(duration))

        final_duration = min(max(calculated_duration, MIN_SCENE_DURATION), MAX_SCENE_DURATION)
        scene["duration"] = round(final_duration, 1)
        _logger.info(f"Task {task_id}, Scene {scene_number}: Final duration set to {scene['duration']}s (clamped from {calculated_duration:.1f}s). Narration: '{narration_text[:50]}...'")
        # --- End Change ---

        scene["start_time"] = round(cumulative_time, 2)
        cumulative_time += scene["duration"]

        scene["image_file"] = None
        scene["image_url"] = None
        if image_generation_enabled and openai_api_key and image_prompt:
            image_filename = f"{base_filename}_scene_{scene_number}.png"
            temp_image_dir = tempfile.mkdtemp(prefix=f"storyboard_img_{task_id}_")
            temp_image_path = os.path.join(temp_image_dir, image_filename)

            try:
                generated_temp_path = generate_dalle_image(image_prompt, temp_image_path, openai_api_key)
                if generated_temp_path:
                    with open(generated_temp_path, "rb") as img_file:
                        store_result = storage_manager.store_file(
                            file_data=img_file,
                            file_type="storyboard",
                            filename=image_filename,
                            content_type="image/png"
                        )
                    image_results.append(store_result)

                    if "error" not in store_result:
                        scene["image_file"] = store_result.get("file_path")
                        scene["image_url"] = store_result.get("url")
                        _logger.info(f"Task {task_id}, Scene {scene_number}: Image stored via {store_result.get('provider')}: {scene['image_file']}")
                    else:
                        _logger.error(f"Task {task_id}, Scene {scene_number}: Failed to store image via storage_manager: {store_result.get('error')}")
                else:
                     _logger.warning(f"Task {task_id}, Scene {scene_number}: DALL-E image generation did not return a path for prompt: '{image_prompt[:50]}...'")

            except Exception as img_e:
                 _logger.error(f"Task {task_id}, Scene {scene_number}: Error generating or storing DALL-E image: {img_e}", exc_info=True)
            finally:
                 try:
                     if os.path.exists(temp_image_dir):
                         shutil.rmtree(temp_image_dir)
                 except OSError as cleanup_err:
                      _logger.warning(f"Task {task_id}: Failed to clean up temp image directory {temp_image_dir}: {cleanup_err}")

        elif not image_generation_enabled:
             _logger.debug(f"Task {task_id}, Scene {scene_number}: Skipping image generation (globally disabled).")
        elif not openai_api_key:
             _logger.debug(f"Task {task_id}, Scene {scene_number}: Skipping image generation (API key missing).")
        elif not image_prompt:
             _logger.debug(f"Task {task_id}, Scene {scene_number}: Skipping image generation (image_prompt is empty).")

    # Generate Combined Audio
    combined_narration = "\n\n".join(combined_narration_texts).strip()
    storyboard_data["audio_file"] = None
    storyboard_data["audio_url"] = None
    storyboard_data["scene_timestamps"] = {}
    audio_result_info = None

    if not combined_narration:
         _logger.warning(f"Task {task_id}: Combined narration text (after splitting) is empty. Skipping audio generation.")
    elif not audio_generation_enabled:
         _logger.info(f"Task {task_id}: Audio generation is globally disabled. Skipping audio generation.")
    elif not elevenlabs_api_key:
         _logger.warning(f"Task {task_id}: ElevenLabs API key missing. Skipping audio generation.")
    else:
        _logger.info(f"Task {task_id}: Generating combined audio with timestamps for {len(combined_narration_texts)} scenes...")
        audio_filename = f"{base_filename}_narration.mp3"
        temp_audio_dir = tempfile.mkdtemp(prefix=f"storyboard_audio_{task_id}_")
        temp_audio_path = os.path.join(temp_audio_dir, audio_filename)

        try:
            audio_gen_result = generate_elevenlabs_audio(
                text=combined_narration,
                api_key=elevenlabs_api_key,
                output_mp3_path=temp_audio_path,
                voice_id=voice_id,
                model_id=appconfig.elevenLabs.DEFAULT_MODEL_ID,
                return_timestamps=True
            )

            timestamps = None
            audio_generated = False
            if isinstance(audio_gen_result, dict) and "timestamps" in audio_gen_result:
                timestamps = audio_gen_result["timestamps"]
                audio_generated = True
                _logger.info(f"Task {task_id}: Audio generated with timestamps to temp path: {temp_audio_path}")
            elif audio_gen_result is True:
                 audio_generated = True
                 _logger.warning(f"Task {task_id}: Audio generated to temp path, but timestamp data was not returned by generate_elevenlabs_audio.")
            else:
                 _logger.error(f"Task {task_id}: Combined audio generation failed (generate_elevenlabs_audio returned False or None).")

            if audio_generated and os.path.exists(temp_audio_path):
                with open(temp_audio_path, "rb") as audio_file:
                    store_result = storage_manager.store_file(
                        file_data=audio_file,
                        file_type="storyboard",
                        filename=audio_filename,
                        content_type="audio/mpeg"
                    )
                audio_result_info = store_result

                if "error" not in store_result:
                    storyboard_data["audio_file"] = store_result.get("file_path")
                    storyboard_data["audio_url"] = store_result.get("url")
                    if timestamps:
                        # todo: Potentially map ElevenLabs timestamps back to scenes if needed,
                        storyboard_data["scene_timestamps"] = timestamps
                    _logger.info(f"Task {task_id}: Combined audio stored via {store_result.get('provider')}: {storyboard_data['audio_file']}")
                else:
                    _logger.error(f"Task {task_id}: Failed to store combined audio via storage_manager: {store_result.get('error')}")
            elif audio_generated:
                 _logger.error(f"Task {task_id}: Audio generation reported success, but temp file '{temp_audio_path}' not found for storage.")


        except Exception as audio_e:
             _logger.error(f"Task {task_id}: Error during combined audio generation or storage: {audio_e}", exc_info=True)
        finally:
             try:
                 if os.path.exists(temp_audio_dir):
                     shutil.rmtree(temp_audio_dir)
             except OSError as cleanup_err:
                  _logger.warning(f"Task {task_id}: Failed to clean up temp audio directory {temp_audio_dir}: {cleanup_err}")

    storyboard_json_filename = f"{base_filename}_storyboard.json"
    storyboard_json_result = None
    try:
        storyboard_json_str = json.dumps(storyboard_data, indent=2)
        store_result = storage_manager.store_file(
            file_data=storyboard_json_str,
            file_type="storyboard",
            filename=storyboard_json_filename,
            content_type="application/json"
        )
        storyboard_json_result = store_result

        if "error" not in store_result:
            final_path = store_result.get('file_path')
            _logger.info(f"Task {task_id}: Complete storyboard JSON stored via {store_result.get('provider')}: {final_path}")
            storyboard_data["storyboard_file"] = final_path
            storyboard_data["storyboard_url"] = store_result.get("url")
        else:
            _logger.error(f"Task {task_id}: Failed to store final storyboard JSON via storage_manager: {store_result.get('error')}")
            return None

    except (IOError, TypeError) as e:
        _logger.error(f"Task {task_id}: Failed to serialize or initiate storage for final storyboard JSON: {e}")
        return None

    _logger.info(f"Storyboard generation process completed successfully for task: '{task_id}'")

    return storyboard_data

def split_story_by_markers(story: str, scenes: List[Dict[str, Any]]) -> List[str]:
    """
    Splits the original story text into segments based on scene_start_markers.

    Args:
        story: The full original story text.
        scenes: A list of scene dictionaries, each containing 'scene_start_marker'.

    Returns:
        A list of strings, where each string is the narration text for a scene.
        Returns an empty list if splitting fails.
    """
    if not story or not scenes:
        _logger.error("Cannot split story: Story text or scene list is empty.")
        return []

    scene_narrations = []
    current_pos = 0
    story_len = len(story)
    last_marker_end = 0

    for i, scene in enumerate(scenes):
        raw_marker = scene.get("scene_start_marker", "").strip()

        marker = re.sub(r'\s*//.*$', '', raw_marker).strip(' \'"')

        if not marker:
            _logger.error(f"Scene {i+1} has an empty start marker after cleaning ('{raw_marker}'). Cannot split accurately.")
            return []

        try:
            marker_pattern = re.escape(marker)
            search_region = story[last_marker_end:]
            match = re.search(marker_pattern, search_region, re.IGNORECASE)

            if not match:
                 _logger.warning(f"Marker for scene {i+1} ('{marker[:30]}...') not found starting after pos {last_marker_end}. Searching from beginning.")
                 match = re.search(marker_pattern, story, re.IGNORECASE)
                 if not match:
                     _logger.error(f"Scene {i+1} start marker ('{marker[:30]}...') not found anywhere in the story. Aborting split.")
                     return []
                 else:
                     found_at = match.start()
                     if found_at < current_pos:
                          _logger.warning(f"Marker for scene {i+1} ('{marker[:30]}...') found at {found_at}, which is before current position {current_pos}. This might indicate overlapping markers or incorrect order.")
                          # todo: Decide how to handle: skip scene, error out, or try to proceed?
                          start_index = found_at
                     else:
                          start_index = found_at
            else:
                 start_index = match.start() + last_marker_end

            if i > 0:
                prev_scene_end_index = start_index
                prev_scene_text = story[current_pos:prev_scene_end_index].strip()
                if not prev_scene_text:
                     _logger.warning(f"Empty text segment generated for scene {i} between markers.")
                scene_narrations.append(prev_scene_text)

            current_pos = start_index
            last_marker_end = match.end() + (last_marker_end if match.start() > 0 or last_marker_end == 0 else 0) # Adjust based on where match was found


        except re.error as e:
            _logger.error(f"Regex error while searching for marker '{marker[:30]}...': {e}")
            return []
        except Exception as e:
             _logger.error(f"Unexpected error finding marker for scene {i+1} ('{marker[:30]}...'): {e}", exc_info=True)
             return []

    last_scene_text_raw = story[current_pos:].strip()
    desc_marker = "[Description]"
    desc_index = last_scene_text_raw.find(desc_marker)
    if desc_index != -1:
        last_scene_text = last_scene_text_raw[:desc_index].strip()
        _logger.debug(f"Removed description marker and content from the last scene segment.")
    else:
        last_scene_text = last_scene_text_raw
        _logger.warning(f"Description marker '[Description]' not found in the last scene segment. It might have been missing or formatted differently.")

    if not last_scene_text:
         _logger.warning(f"Empty text segment generated for the last scene ({len(scenes)}) after potential description removal.")
    scene_narrations.append(last_scene_text)

    if len(scene_narrations) != len(scenes):
        _logger.error(f"Splitting mismatch: Expected {len(scenes)} scenes, but generated {len(scene_narrations)} text segments. Markers might be overlapping or incorrect.")
        markers = [s.get("scene_start_marker", "N/A") for s in scenes]
        _logger.debug(f"Markers used: {markers}")
        _logger.debug(f"Generated segments: {[seg[:50] + '...' for seg in scene_narrations]}")
        return []

    _logger.info(f"Successfully split story into {len(scene_narrations)} segments based on markers.")
    return scene_narrations