# viralStoryGenerator/src/storyboard.py

import os
import json
import datetime
import requests
import re
import time
import tempfile
import shutil
from typing import Dict, Any, Optional

from viralStoryGenerator.src.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.prompts.prompts import get_storyboard_prompt
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig
from viralStoryGenerator.utils.security import is_safe_filename
from viralStoryGenerator.models.models import STORYBOARD_RESPONSE_FORMAT
from viralStoryGenerator.src.llm import THINK_PATTERN
from viralStoryGenerator.utils.storage_manager import storage_manager

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
    Uses the LLM (specifically the MODEL_MULTI) to produce a storyboard breakdown in JSON format.
    Includes basic error handling and parsing.

    Returns:
        Parsed storyboard data as dict, or None on failure.
    """
    model = appconfig.llm.MODEL_MULTI
    if not story or not llm_endpoint or not model:
        _logger.error("Missing required arguments for storyboard structure generation (story, endpoint, or configured MODEL_MULTI).")
        return None

    prompt = get_storyboard_prompt(story).strip()
    _logger.info(f"Requesting storyboard structure from LLM using model: {model}...")

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
        if response.status_code == 400:
            _logger.warning(f"LLM ({model}) returned 400 Bad Request, retrying without 'response_format' parameter.")
            data.pop("response_format", None)
            response = requests.post(llm_endpoint, headers=headers, json=data, timeout=appconfig.httpOptions.TIMEOUT)
            response.raise_for_status()
        else:
            _logger.error(f"LLM ({model}) request for storyboard failed: {e}")
            return None
    except requests.exceptions.Timeout:
         _logger.error(f"LLM ({model}) request for storyboard timed out after {appconfig.httpOptions.TIMEOUT} seconds.")
         return None
    except requests.exceptions.RequestException as e:
        payload_str = json.dumps(data)
        _logger.error(f"Failed to generate storyboard structure from LLM ({model}): {e}. Request Payload: {payload_str}", exc_info=True)
        return None

    try:
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        _logger.error(f"Failed to parse or access expected keys in LLM response for storyboard (model: {model}): {e}. Response text: {response.text[:500]}")
        return None


    cleaned_content = content.strip()

    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    cleaned_content = re.sub(r'^```(?:json)?\s*', '', cleaned_content, flags=re.IGNORECASE)
    cleaned_content = re.sub(r'\s*```$', '', cleaned_content)
    cleaned_content = cleaned_content.strip()

    if cleaned_content.lower().startswith("json"):
        cleaned_content = cleaned_content[4:].strip()

    try:
         storyboard_data = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
         _logger.error(f"JSON decode error after cleaning (model: {model}): {e}. Cleaned content: {cleaned_content[:500]}...")
         return None


    return storyboard_data


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
    Generates a full storyboard from a story script.

    Steps:
    1. Generate scene structure (text, prompts, durations) using LLM (MODEL_MULTI).
    2. Determine base directory and filename using task_id.
    3. For each scene:
        - Calculate duration if missing.
        - Generate DALL-E image using the prompt (if enabled).
    4. Combine narration texts.
    5. Generate a single audio file for the combined narration using ElevenLabs (if enabled).
    6. Save the complete storyboard JSON data using storage_manager.

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
    _logger.info(f"Starting storyboard generation for task: '{task_id}' (Topic: '{topic}')")

    storyboard_data = generate_storyboard_structure(story, llm_endpoint, temperature)
    if not storyboard_data or "scenes" not in storyboard_data or not storyboard_data["scenes"]:
        _logger.error(f"Storyboard structure generation failed or returned empty scenes for task {task_id}.")
        return None

    if not task_id:
        _logger.error("Task ID is missing, cannot generate storyboard filenames.")
        return None

    base_filename = task_id
    storyboard_base_dir = storage_manager._get_storage_dir("storyboard") # Get the root storyboard dir

    try:
        os.makedirs(storyboard_base_dir, exist_ok=True)
        _logger.info(f"Ensured storyboard base directory exists: {storyboard_base_dir}")
    except OSError as e:
        _logger.error(f"Failed to ensure storyboard base directory '{storyboard_base_dir}': {e}")
        return None

    # Get API keys from config
    openai_api_key = appconfig.openAI.API_KEY
    elevenlabs_api_key = appconfig.elevenLabs.API_KEY
    image_generation_enabled = appconfig.openAI.ENABLED
    audio_generation_enabled = appconfig.elevenLabs.ENABLED

    if not image_generation_enabled:
        _logger.warning(f"Task {task_id}: Image generation is globally disabled via config. Skipping DALL-E calls.")
    elif not openai_api_key:
        _logger.warning(f"Task {task_id}: OpenAI API key (for DALL-E) is not configured. Skipping image generation.")

    if not audio_generation_enabled:
         _logger.warning(f"Task {task_id}: Audio generation is globally disabled via config. Skipping ElevenLabs calls.")
    elif not elevenlabs_api_key:
        _logger.warning(f"Task {task_id}: ElevenLabs API key is not configured. Skipping audio generation.")

    cumulative_time = 0.0
    scene_texts = []
    word_per_minute_rate = 150
    image_results = []

    for i, scene in enumerate(storyboard_data["scenes"]):
        scene_number = scene.get("scene_number", i + 1)
        scene["scene_number"] = scene_number

        narration_text = scene.get("narration_text", "").strip()
        image_prompt = scene.get("image_prompt", "").strip()
        duration = scene.get("duration")

        if not isinstance(duration, (int, float)) or duration <= 0:
            words = len(narration_text.split())
            estimated_duration = round((words / word_per_minute_rate) * 60.0, 1)
            scene["duration"] = estimated_duration
            _logger.debug(f"Task {task_id}, Scene {scene_number}: Calculated duration {estimated_duration}s for {words} words.")
        else:
             scene["duration"] = float(duration)


        scene["start_time"] = round(cumulative_time, 2)
        cumulative_time += scene["duration"]

        scene_texts.append(narration_text)

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
                     _logger.warning(f"Task {task_id}, Scene {scene_number}: DALL-E image generation did not return a path.")

            except Exception as img_e:
                 _logger.error(f"Task {task_id}, Scene {scene_number}: Error generating or storing DALL-E image: {img_e}")
            finally:
                 try:
                     shutil.rmtree(temp_image_dir)
                 except OSError as cleanup_err:
                      _logger.warning(f"Task {task_id}: Failed to clean up temp image directory {temp_image_dir}: {cleanup_err}")

        elif not image_generation_enabled:
             _logger.debug(f"Task {task_id}, Scene {scene_number}: Skipping image generation (globally disabled).")
        elif not openai_api_key:
             _logger.debug(f"Task {task_id}, Scene {scene_number}: Skipping image generation (API key missing).")
        # < --- End Image Handling ---

    combined_narration = "\n\n".join(scene_texts).strip()
    storyboard_data["audio_file"] = None
    storyboard_data["audio_url"] = None
    storyboard_data["scene_timestamps"] = {}
    audio_result_info = None

    if not combined_narration:
         _logger.warning(f"Task {task_id}: Combined narration text is empty. Skipping audio generation.")
    elif not audio_generation_enabled:
         _logger.info(f"Task {task_id}: Audio generation is globally disabled. Skipping audio generation.")
    elif not elevenlabs_api_key:
         _logger.warning(f"Task {task_id}: ElevenLabs API key missing. Skipping audio generation.")
    else:
        _logger.info(f"Task {task_id}: Generating combined audio with timestamps...")
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
                 _logger.warning(f"Task {task_id}: Audio generated to temp path, but timestamp data was not returned.")
            else:
                 _logger.error(f"Task {task_id}: Combined audio generation failed.")

            # If audio generated, store it using storage_manager
            if audio_generated:
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
                        storyboard_data["scene_timestamps"] = timestamps
                    _logger.info(f"Task {task_id}: Combined audio stored via {store_result.get('provider')}: {storyboard_data['audio_file']}")
                else:
                    _logger.error(f"Task {task_id}: Failed to store combined audio via storage_manager: {store_result.get('error')}")

        except Exception as audio_e:
             _logger.error(f"Task {task_id}: Error during combined audio generation or storage: {audio_e}")
        finally:
             try:
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
            _logger.info(f"Task {task_id}: Complete storyboard JSON stored via {store_result.get('provider')}: {store_result.get('file_path')}")
            storyboard_data["storyboard_file"] = store_result.get("file_path")
            storyboard_data["storyboard_url"] = store_result.get("url")
        else:
            _logger.error(f"Task {task_id}: Failed to store storyboard JSON via storage_manager: {store_result.get('error')}")
            return None

    except (IOError, TypeError) as e:
        _logger.error(f"Task {task_id}: Failed to serialize or initiate storage for storyboard JSON: {e}")
        return None

    _logger.info(f"Storyboard generation process completed for task: '{task_id}'")

    return storyboard_data