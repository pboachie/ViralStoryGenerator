# viralStoryGenerator/src/storyboard.py

import os
import json
import datetime
import requests
import re
import time
from typing import Dict, Any, Optional

from viralStoryGenerator.src.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.prompts.prompts import get_storyboard_prompt
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils import config

appconfig = config.config

def generate_storyboard_structure(story, llm_endpoint, model, temperature):
    """
    Uses the LLM to produce a storyboard breakdown in JSON format.
    Includes basic error handling and parsing.

    Returns:
        Parsed storyboard data as dict, or None on failure.
    """
    if not story or not llm_endpoint or not model:
        _logger.error("Missing required arguments for storyboard structure generation (story, endpoint, model).")
        return None

    prompt = get_storyboard_prompt(story).strip()
    _logger.info("Requesting storyboard structure from LLM...")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that strictly follows instructions to generate structured JSON output."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": appconfig.llm.MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(
            llm_endpoint,
            headers=headers,
            data=json.dumps(data),
            timeout=appconfig.httpOptions.TIMEOUT
        )
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]

        if isinstance(content, str):
             content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content).strip()
             storyboard_data = json.loads(content)
        elif isinstance(content, dict):
             storyboard_data = content
        else:
             _logger.error(f"Unexpected content type in LLM response for storyboard: {type(content)}")
             return None

        # Basic validation of structure
        if "scenes" not in storyboard_data or not isinstance(storyboard_data["scenes"], list):
             _logger.error("Generated storyboard JSON is missing 'scenes' list.")
             return None

        _logger.info(f"Successfully generated storyboard structure with {len(storyboard_data['scenes'])} scenes.")

        thinking = response_json["choices"][0]["message"].get("reasoning_content")
        if thinking: storyboard_data["thinking"] = thinking

        return storyboard_data

    except requests.exceptions.Timeout:
         _logger.error(f"LLM request for storyboard timed out after {appconfig.httpOptions.TIMEOUT} seconds.")
         return None
    except requests.exceptions.RequestException as e:
        _logger.error(f"Failed to generate storyboard structure from LLM: {e}")
        return None
    except json.JSONDecodeError as e:
         _logger.error(f"Failed to parse storyboard JSON response from LLM: {e}. Response text: {response.text[:500]}")
         return None
    except (KeyError, IndexError) as e:
         _logger.error(f"Unexpected LLM response structure for storyboard: {e}. Response: {response_json}")
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


def generate_storyboard(story: str, topic: str, llm_endpoint: str, model: str, temperature: float, voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Generates a full storyboard from a story script.

    Steps:
    1. Generate scene structure (text, prompts, durations) using LLM.
    2. Create a safe base directory using sanitized topic and date info.
    3. For each scene:
        - Calculate duration if missing.
        - Generate DALL-E image using the prompt.
    4. Combine narration texts.
    5. Generate a single audio file for the combined narration using ElevenLabs (with timestamps).
    6. Save the complete storyboard JSON data.

    Args:
        story: The full story script text.
        topic: The original topic (used for naming files).
        llm_endpoint: LLM API endpoint.
        model: LLM model name.
        temperature: LLM temperature setting.
        voice_id: Optional ElevenLabs voice ID.

    Returns:
        The complete storyboard data dictionary if successful, None otherwise.
    """
    _logger.info(f"Starting storyboard generation for topic: '{topic}'")

    storyboard_data = generate_storyboard_structure(story, llm_endpoint, model, temperature)
    if not storyboard_data or "scenes" not in storyboard_data or not storyboard_data["scenes"]:
        _logger.error("Storyboard structure generation failed or returned empty scenes.")
        return None

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar().week
    safe_topic_base = sanitize_for_filename(topic)
    base_filename = f"{safe_topic_base}_{date_str}"

    base_folder_path = os.path.join("GeneratedStoryboards", f"Week{week_num}", safe_topic_base)
    try:
        os.makedirs(base_folder_path, exist_ok=True)
        _logger.info(f"Ensured storyboard output directory exists: {base_folder_path}")
    except OSError as e:
        _logger.error(f"Failed to create storyboard output directory '{base_folder_path}': {e}")
        return None # Cannot proceed without output directory

    # Get API keys from config
    openai_api_key = appconfig.openAI.API_KEY
    elevenlabs_api_key = appconfig.elevenLabs.API_KEY

    if not openai_api_key:
        _logger.warning("OpenAI API key (for DALL-E) is not configured. Skipping image generation.")
    if not elevenlabs_api_key:
        _logger.warning("ElevenLabs API key is not configured. Skipping audio generation.")

    cumulative_time = 0.0
    scene_texts = []
    word_per_minute_rate = 150

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
            _logger.debug(f"Scene {scene_number}: Calculated duration {estimated_duration}s for {words} words.")
        else:
             scene["duration"] = float(duration)


        scene["start_time"] = round(cumulative_time, 2)
        cumulative_time += scene["duration"]

        scene_texts.append(narration_text)

        # Generate scene image
        scene["image_file"] = ""
        if openai_api_key and image_prompt:
            image_filename = f"{base_filename}_scene_{scene_number}.png"
            image_path = os.path.join(base_folder_path, image_filename)
            try:
                generated_path = generate_dalle_image(image_prompt, image_path, openai_api_key)
                if generated_path:
                    scene["image_file"] = os.path.relpath(generated_path, os.path.dirname(base_folder_path))
                    # Or filename: scene["image_file"] = image_filename
            except Exception as img_e:
                 _logger.error(f"Error generating DALL-E image for scene {scene_number}: {img_e}")
                 # Continue without image for this scene

    # Combine narration
    combined_narration = "\n\n".join(scene_texts).strip()
    if not combined_narration:
         _logger.warning("Combined narration text is empty. Skipping audio generation.")
         storyboard_data["audio_file"] = ""
         storyboard_data["scene_timestamps"] = {}
    elif not elevenlabs_api_key:
         _logger.warning("ElevenLabs API key missing. Skipping audio generation.")
         storyboard_data["audio_file"] = ""
         storyboard_data["scene_timestamps"] = {}
    else:
        # Generate combined audio with timestamps
        _logger.info("Generating combined audio with timestamps...")
        audio_filename = f"{base_filename}_narration.mp3"
        audio_path = os.path.join(base_folder_path, audio_filename)
        storyboard_data["audio_file"] = ""
        storyboard_data["scene_timestamps"] = {}

        try:
            audio_result = generate_elevenlabs_audio(
                text=combined_narration,
                api_key=elevenlabs_api_key,
                output_mp3_path=audio_path,
                voice_id=voice_id,
                model_id=appconfig.elevenLabs.DEFAULT_MODEL_ID,
                return_timestamps=True
            )

            if isinstance(audio_result, dict) and "timestamps" in audio_result:
                storyboard_data["audio_file"] = os.path.relpath(audio_path, os.path.dirname(base_folder_path))
                # Or filename: storyboard_data["audio_file"] = audio_filename
                storyboard_data["scene_timestamps"] = audio_result["timestamps"]
                _logger.info(f"Combined audio with timestamps generated: {audio_filename}")
            elif audio_result is True:
                 storyboard_data["audio_file"] = os.path.relpath(audio_path, os.path.dirname(base_folder_path))
                 _logger.warning("Audio generated but timestamp data was not returned.")
            else:
                 _logger.error("Combined audio generation failed.")
        except Exception as audio_e:
             _logger.error(f"Error during combined audio generation: {audio_e}")


    # Save the complete storyboard JSON
    storyboard_json_filename = f"{base_filename}_storyboard.json"
    storyboard_json_path = os.path.join(base_folder_path, storyboard_json_filename)
    try:
        with open(storyboard_json_path, "w", encoding="utf-8") as f:
            json.dump(storyboard_data, f, indent=2)
        _logger.info(f"Complete storyboard JSON saved to {storyboard_json_path}")
    except IOError as e:
        _logger.error(f"Failed to save storyboard JSON file to {storyboard_json_path}: {e}")
        # return None
    except TypeError as e:
         _logger.error(f"Failed to serialize storyboard data to JSON: {e}")
         # return None


    _logger.info(f"Storyboard generation process completed for topic: '{topic}'")
    return storyboard_data