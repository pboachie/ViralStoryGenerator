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
import asyncio

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
    Includes basic error handling, parsing, and retry logic.

    Returns:
        Parsed storyboard data (with scene_start_marker) as dict, or None on failure.
    """
    model = appconfig.llm.MODEL_MULTI
    if not story or not llm_endpoint or not model:
        _logger.error("Missing required arguments for storyboard structure generation (story, endpoint, or configured MODEL_MULTI).")
        return None

    prompt = get_storyboard_prompt(story).strip()
    _logger.debug(f"Requesting storyboard structure using model: {model}")

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

    max_retries = appconfig.storyboard.STRUCTURE_MAX_RETRIES
    for attempt in range(max_retries + 1): # +1 to include the initial attempt
        try:
            response = requests.post(llm_endpoint, headers=headers, json=data, timeout=appconfig.httpOptions.TIMEOUT)
            response.raise_for_status()
            raw_response_content = response.text
            parsed_json = None

            try:
                outer_json = json.loads(raw_response_content)

                # Extract the string content that should contain the actual storyboard JSON
                if (outer_json.get("choices") and
                    isinstance(outer_json["choices"], list) and
                    len(outer_json["choices"]) > 0 and
                    outer_json["choices"][0].get("message") and
                    isinstance(outer_json["choices"][0]["message"], dict) and
                    outer_json["choices"][0]["message"].get("content")):

                    content_str = outer_json["choices"][0]["message"]["content"]

                    try:
                        parsed_json = json.loads(content_str)
                        _logger.debug(f"Successfully parsed nested JSON from 'message.content' for storyboard structure. Attempt {attempt + 1}/{max_retries + 1}")
                        if "scenes" in parsed_json and parsed_json["scenes"]:
                            return parsed_json
                        else:
                            _logger.warning(f"Parsed JSON lacks scenes (attempt {attempt+1}/{max_retries+1})")
                    except json.JSONDecodeError as e_nested:
                        _logger.warning(f"Failed to parse nested JSON string from 'message.content'. Attempt {attempt + 1}/{max_retries + 1}. Error: {e_nested}. Content: {content_str[:500]}. Falling back to other methods.")
                else:
                    _logger.warning(f"Outer JSON structure does not match expected format for nested content. Attempt {attempt + 1}/{max_retries + 1}. Outer JSON: {raw_response_content[:500]}. Falling back to other methods.")

                # Fallback 1: Try parsing the raw_response_content directly if not already successfully parsed from nested
                if not (parsed_json and "scenes" in parsed_json and parsed_json["scenes"]):
                    try:
                        parsed_json = json.loads(raw_response_content)
                        _logger.debug(f"Successfully parsed entire raw response as JSON for storyboard structure. Attempt {attempt + 1}/{max_retries + 1}")
                        if "scenes" in parsed_json and parsed_json["scenes"]:
                            return parsed_json
                        # else: # Warning already logged if it was the nested attempt that failed here
                            # _logger.warning(f"LLM response parsed as JSON but lacks 'scenes' or scenes are empty. Attempt {attempt + 1}/{max_retries + 1}. Response: {raw_response_content[:500]}")
                    except json.JSONDecodeError:
                        _logger.debug(f"Failed to parse entire raw response as JSON, will attempt markdown extraction. Attempt {attempt + 1}/{max_retries + 1}")

                # Fallback 2: Try markdown extraction if other methods failed
                if not (parsed_json and "scenes" in parsed_json and parsed_json["scenes"]):
                    match = re.search(r"```json\\s*([\\s\\S]*?)\\s*```", raw_response_content, re.IGNORECASE)
                    if match:
                        json_str = match.group(1).strip()
                        try:
                            parsed_json = json.loads(json_str)
                            _logger.info(f"Successfully extracted and parsed JSON from markdown for storyboard structure. Attempt {attempt + 1}/{max_retries + 1}")
                            if "scenes" in parsed_json and parsed_json["scenes"]:
                                return parsed_json
                            else:
                                _logger.warning(f"LLM response extracted from markdown but lacks 'scenes' or scenes are empty. Attempt {attempt + 1}/{max_retries + 1}. Extracted: {json_str[:500]}")
                        except json.JSONDecodeError as e_markdown:
                            _logger.error(f"Failed to parse JSON extracted from markdown for storyboard structure. Attempt {attempt + 1}/{max_retries + 1}. Error: {e_markdown}. Extracted: {json_str[:500]}")
                    else:
                        _logger.error(f"No JSON found in LLM response for storyboard structure (neither direct, nested, nor in markdown). Attempt {attempt + 1}/{max_retries + 1}. Response: {raw_response_content[:500]}")

            except json.JSONDecodeError as e_outer:
                 _logger.error(f"Failed to parse the initial LLM response as JSON. Attempt {attempt + 1}/{max_retries + 1}. Error: {e_outer}. Response: {raw_response_content[:500]}")


            # If we reach here, it means parsing failed or scenes were not valid
            if attempt < max_retries:
                _logger.debug(f"Retrying storyboard structure generation attempt {attempt+1}/{max_retries+1}")
                time.sleep(1 * (attempt + 1))
                continue # Go to next attempt
            else:
                _logger.error(f"All {max_retries + 1} attempts failed to generate valid storyboard structure.")
                return None

        except requests.exceptions.HTTPError as e:
            _logger.error(f"HTTP error during storyboard structure request (Attempt {attempt + 1}/{max_retries + 1}): {e}. Response: {e.response.text if e.response else 'No response'}")
        except requests.exceptions.Timeout:
            _logger.error(f"Timeout during storyboard structure request (Attempt {attempt + 1}/{max_retries + 1}).")
        except requests.exceptions.RequestException as e:
            _logger.error(f"Request exception during storyboard structure request (Attempt {attempt + 1}/{max_retries + 1}): {e}")
        except Exception as e:
            _logger.error(f"Unexpected error during storyboard structure generation attempt {attempt + 1}/{max_retries + 1}: {e}")

        if attempt < max_retries:
            _logger.info(f"Retrying storyboard structure generation due to error ({attempt + 1}/{max_retries + 1})...")
            time.sleep(1 * (attempt + 1))
        else:
            _logger.error(f"All {max_retries + 1} attempts failed for storyboard structure generation after encountering errors.")
            return None
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

    _logger.debug(f"DALL-E image request for: {output_image_path}")
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
    response_data = None

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

        _logger.debug(f"DALL-E image saved to {output_image_path}")
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
         _logger.error(f"Failed processing DALL-E response or downloading image: {e}. Response data: {response_data if response_data is not None else 'N/A'}")
         return None
    except IOError as e:
         _logger.error(f"Failed to save DALL-E image to {output_image_path}: {e}")
         return None


def generate_storyboard(story: str, topic: str, task_id: str, llm_endpoint: str, temperature: float, voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Generates a full storyboard from a story script using scene markers.
    Retries fetching markers and splitting the story if initial attempts fail.

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
        _logger.debug(f"Storyboard generation disabled for task: '{task_id}'")
        return None

    _logger.debug(f"Starting storyboard generation for task: '{task_id}' (Topic: '{topic}')")

    max_overall_attempts = getattr(appconfig.storyboard, "RL_MAX_DELAY", 3)
    retry_delay_seconds = getattr(appconfig.storyboard, "RL_MAX_RETRIES", 2)

    if not isinstance(max_overall_attempts, int) or max_overall_attempts <= 0:
        _logger.warning(f"Task {task_id}: Invalid OVERALL_STORYBOARD_ATTEMPTS ({max_overall_attempts}), defaulting to 3.")
        max_overall_attempts = 3
    if not isinstance(retry_delay_seconds, (int, float)) or retry_delay_seconds < 0:
        _logger.warning(f"Task {task_id}: Invalid RETRY_DELAY_SECONDS ({retry_delay_seconds}), defaulting to 2.")
        retry_delay_seconds = 2

    successful_storyboard_data: Optional[Dict[str, Any]] = None
    successful_scenes_list: Optional[List[Dict[str, Any]]] = None
    successful_scene_narrations: Optional[List[str]] = None

    for attempt in range(max_overall_attempts):
        _logger.debug(f"Storyboard attempt {attempt+1}/{max_overall_attempts} for task {task_id}")

        # Step 1: Generate storyboard structure (markers)
        current_storyboard_structure = generate_storyboard_structure(story, llm_endpoint, temperature)

        if not current_storyboard_structure or "scenes" not in current_storyboard_structure or not current_storyboard_structure["scenes"]:
            _logger.warning(f"Task {task_id}, Attempt {attempt + 1}: Failed to generate storyboard structure or no scenes found.")
            if attempt < max_overall_attempts - 1:
                _logger.info(f"Task {task_id}: Retrying structure generation. Waiting {retry_delay_seconds * (attempt + 1)}s...")
                time.sleep(retry_delay_seconds * (attempt + 1))
                continue  # Go to next overall attempt
            else:
                _logger.error(f"Task {task_id}: All {max_overall_attempts} attempts failed to generate a valid storyboard structure.")
                break # Exit loop, will return None later

        current_scenes_from_structure = current_storyboard_structure["scenes"]

        # Step 2: Split story using the markers from the current structure
        _logger.debug(f"Splitting story using {len(current_scenes_from_structure)} markers")
        narrations = split_story_by_markers(story, current_scenes_from_structure)

        if narrations and len(narrations) == len(current_scenes_from_structure):
            _logger.debug(f"Generated and split into {len(narrations)} scenes")
            successful_storyboard_data = current_storyboard_structure
            successful_scenes_list = current_scenes_from_structure
            successful_scene_narrations = narrations
            break  # Success, exit the overall attempt loop
        else:
            _logger.warning(f"Task {task_id}, Attempt {attempt + 1}: Failed to split story accurately. Expected {len(current_scenes_from_structure)} narrations, got {len(narrations) if narrations is not None else 'None'}.")
            markers_for_debug = [s.get("scene_start_marker", "N/A") for s in current_scenes_from_structure]
            _logger.debug(f"Task {task_id}, Attempt {attempt + 1}: Markers that failed splitting: {markers_for_debug}")
            if attempt < max_overall_attempts - 1:
                _logger.info(f"Task {task_id}: Retrying with new set of markers. Waiting {retry_delay_seconds * (attempt + 1)}s...")
                time.sleep(retry_delay_seconds * (attempt + 1))
            # else: last attempt, loop will end

    if not successful_storyboard_data or not successful_scenes_list or not successful_scene_narrations:
        _logger.error(f"Task {task_id}: Failed to generate and split storyboard after {max_overall_attempts} attempts. Proceeding without storyboard.")
        return None

    storyboard_data = successful_storyboard_data
    scenes = successful_scenes_list
    scene_narrations = successful_scene_narrations

    if not task_id:
        _logger.error("Task ID is missing, cannot generate storyboard filenames.")
        return None
    base_filename = task_id
    storyboard_base_dir = storage_manager._get_storage_dir("storyboard")
    try:
        os.makedirs(storyboard_base_dir, exist_ok=True)
        _logger.debug(f"Storyboard base directory ready: {storyboard_base_dir}")
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
    word_per_minute_rate = appconfig.storyboard.WPM
    image_results = []

    # --- Start Change ---
    # Define desired duration range
    MIN_SCENE_DURATION = 3.0
    MAX_SCENE_DURATION = 15.0
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
        _logger.debug(f"Scene {scene_number}: Duration {scene['duration']}s; narration start '{narration_text[:50]}...'")
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
         _logger.debug(f"Audio generation disabled for task {task_id}")
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

    _logger.debug(f"Storyboard generation completed for task: '{task_id}'")

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

    _logger.debug(f"Split into {len(scene_narrations)} segments")
    return scene_narrations

async def generate_storyboard_from_story_script(
    job_id: str,
    story_script: str,
    num_scenes: int,
    image_style: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    llm_provider_config: Optional[Dict[str, Any]] = None,
    retry_attempts: int = 3
) -> Dict[str, Any]:
    """
    Generates a storyboard from a story script, including scene descriptions,
    visual suggestions, and camera shots. Retries marker generation and splitting on failure.
    """
    _logger.info(f"Job {job_id}: Starting storyboard generation from script. Target scenes: {num_scenes}, Style: {image_style}")
    if not story_script:
        _logger.error(f"Job {job_id}: Story script is empty.")
        raise ValueError("Story script cannot be empty.")
    if num_scenes <= 0:
        _logger.error(f"Job {job_id}: Number of scenes must be positive, got {num_scenes}.")
        raise ValueError("Number of scenes must be positive.")

    if not llm_provider_config:
        _logger.error(f"Job {job_id}: llm_provider_config is not provided.")
        raise ValueError("llm_provider_config is required.")

    scenes_content: List[Dict[str, Any]] = []
    last_split_errors: List[str] = []
    last_scene_markers: List[Dict[str, Any]] = []
    success = False
    storyboard_structure: Optional[Dict[str, Any]] = None

    for attempt in range(retry_attempts):
        _logger.info(f"Job {job_id}: Storyboard generation attempt {attempt + 1} of {retry_attempts}.")

        try:
            llm_endpoint = llm_provider_config.get("endpoint")
            temperature = llm_provider_config.get("temperature")

            if not llm_endpoint:
                raise ValueError("LLM endpoint not found in llm_provider_config.")
            if temperature is None:
                raise ValueError("Temperature not found in llm_provider_config.")

            storyboard_structure = await asyncio.to_thread(
                generate_storyboard_structure,
                story_script,
                llm_endpoint,
                float(temperature)
            )
        except Exception as e_struct:
            _logger.error(f"Job {job_id}: Error calling generate_storyboard_structure in attempt {attempt + 1}: {e_struct}")
            last_split_errors = [f"Error generating storyboard structure: {e_struct}"]
            if attempt < retry_attempts - 1:
                _logger.info(f"Job {job_id}: Retrying structure generation. Waiting a moment...")
                await asyncio.sleep(min(5, 2 ** (attempt + 1)))
                continue
            else:
                break

        if not storyboard_structure:
            _logger.error(f"Job {job_id}: generate_storyboard_structure returned None in attempt {attempt + 1}.")
            last_split_errors = ["generate_storyboard_structure returned None."]
            if attempt < retry_attempts - 1:
                _logger.info(f"Job {job_id}: Retrying to get storyboard structure. Waiting a moment...")
                await asyncio.sleep(min(5, 2 ** (attempt + 1)))
                continue
            else:
                break

        current_scene_markers = storyboard_structure.get("scenes", [])
        last_scene_markers = current_scene_markers

        if not current_scene_markers:
            _logger.error(f"Job {job_id}: LLM did not return scene markers in attempt {attempt + 1}. Structure: {storyboard_structure}")
            last_split_errors = ["LLM did not return scene markers."]
            if attempt < retry_attempts - 1:
                _logger.info(f"Job {job_id}: Retrying to get scene markers. Waiting a moment...")
                await asyncio.sleep(min(5, 2 ** (attempt + 1))) # Exponential backoff
                continue
            else:
                break

        # 2. Split story script using the generated markers
        _logger.debug(f"Job {job_id}: Attempt {attempt + 1}: Splitting story with {len(current_scene_markers)} markers.")
        narrations = split_story_by_markers(story_script, current_scene_markers)

        if narrations and len(narrations) == len(current_scene_markers):
            _logger.info(f"Job {job_id}: Successfully split story into {len(narrations)} narrations in attempt {attempt + 1}.")

            processed_scenes = []
            for i, scene_marker_data in enumerate(current_scene_markers):
                scene_data = dict(scene_marker_data)
                scene_data['narration_text'] = narrations[i]
                processed_scenes.append(scene_data)

            scenes_content = processed_scenes
            last_split_errors = []
            success = True
            break
        else:
            error_msg = f"Failed to split story accurately. Expected {len(current_scene_markers)} narrations, got {len(narrations) if narrations is not None else 'None'}."
            last_split_errors = [error_msg]
            _logger.warning(f"Job {job_id}: Attempt {attempt + 1} failed to split story: {error_msg}")
            _logger.debug(f"Job {job_id}: Attempt {attempt + 1}: Story script for failed split (first 500 chars):\n{story_script[:500]}")
            _logger.debug(f"Job {job_id}: Attempt {attempt + 1}: Scene markers for failed split: {current_scene_markers}")

            if attempt < retry_attempts - 1:
                _logger.info(f"Job {job_id}: Retrying storyboard generation. Waiting a moment...")
                await asyncio.sleep(min(5, 2 ** (attempt + 1))) # Exponential backoff
            else:
                break

    if not success:
        error_detail = f"Last errors: {'; '.join(last_split_errors)}."
        # Ensure storyboard_structure is not None before accessing it in the error message
        markers_tried_message = f"Last scene markers tried: {last_scene_markers if last_scene_markers else (storyboard_structure.get('scenes') if storyboard_structure else 'None returned or structure generation failed')}."
        if not last_scene_markers and any("LLM did not return scene markers" in e for e in last_split_errors) or any("generate_storyboard_structure returned None" in e for e in last_split_errors):
             error_message = f"Job {job_id}: Failed to generate scene markers or structure after {retry_attempts} attempts. {error_detail}"
        elif any("Error generating storyboard structure" in e for e in last_split_errors):
            error_message = f"Job {job_id}: Failed to generate storyboard structure after {retry_attempts} attempts. {error_detail}"
        else:
            error_message = (
                f"Job {job_id}: Failed to split story script into scenes after {retry_attempts} attempts. "
                f"{error_detail} "
                f"{markers_tried_message}"
            )
        _logger.error(error_message)
        raise ValueError(error_message)

    _logger.info(f"Job {job_id}: Successfully processed storyboard structure and narrations for {len(scenes_content)} scenes.")

    # Ensure storyboard_structure is not None before returning it.
    if storyboard_structure is None: # Should not happen if success is True
        _logger.error(f"Job {job_id}: Storyboard structure is None even after successful processing. This indicates a logic error.")
        raise ValueError("Storyboard structure is unexpectedly None after successful processing.")

    return {"scenes": scenes_content, "original_structure": storyboard_structure}