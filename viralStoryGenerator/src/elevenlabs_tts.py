# viralStoryGenerator/src/elevenlabs_tts.py

import requests
import time
import json
import base64
import os
import uuid
import tempfile
from typing import Dict, Any, Optional, Union

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig

DEFAULT_VOICE_ID = appconfig.elevenLabs.VOICE_ID or "JZ3e95uoTACVf6tXaaEi"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"

# Define a user agent for HTTP requests
APP_USER_AGENT = f"{appconfig.APP_TITLE}/{appconfig.VERSION}"

def generate_audio(text: str) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED? (Check usage) - Generates audio using ElevenLabs and saves to a temporary file.
    Modern flow likely uses generate_elevenlabs_audio directly with paths managed by storage_manager/handlers.

    Args:
        text: Text to convert to speech

    Returns:
        Dict with temp file information or None on failure.
    """
    _logger.warning("generate_audio function called - this might be deprecated. Check workflow.")
    try:
        # Check for API key
        api_key = appconfig.elevenLabs.API_KEY
        voice_id = appconfig.elevenLabs.VOICE_ID or DEFAULT_VOICE_ID

        if not api_key:
            _logger.error("No ElevenLabs API key configured. Cannot generate audio.")
            return None

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp3"
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        _logger.info(f"Generating temporary audio file at: {temp_path}")

        # Generate the audio file using the core function
        success = generate_elevenlabs_audio(
            text=text,
            api_key=api_key,
            output_mp3_path=temp_path,
            voice_id=voice_id,
            model_id=DEFAULT_MODEL_ID,
            stability=0.5,
            similarity_boost=0.75
        )

        if not success:
            _logger.error(f"Failed to generate temporary audio file for text starting with: {text[:50]}...")
            # Clean up potentially empty file if creation failed mid-write?
            if os.path.exists(temp_path):
                 try: os.remove(temp_path)
                 except OSError: pass
            return None

        return {
            "name": filename,
            "path": temp_path,
            "file_path": temp_path
        }

    except Exception as e:
        _logger.exception(f"Unexpected error in generate_audio: {e}")
        return None


def generate_elevenlabs_audio(
    text: str,
    api_key: str,
    output_mp3_path: str,
    voice_id: Optional[str] = None,
    model_id: str = DEFAULT_MODEL_ID,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    timeout: int = 90,
    return_timestamps: bool = False
) -> Union[bool, Dict[str, Any]]:
    """
    Calls the ElevenLabs TTS API to convert text into speech audio.

    Handles standard MP3 generation and generation with timestamps (returns dict).
    Includes retries and timeout.

    Args:
        text: The text content to synthesize.
        api_key: ElevenLabs API key.
        output_mp3_path: The full path where the generated MP3 audio file will be saved.
                         SECURITY: The caller MUST validate this path is safe before passing it.
        voice_id: The specific ElevenLabs voice ID to use. Defaults to configured or global default.
        model_id: The ElevenLabs model ID (e.g., 'eleven_multilingual_v2').
        stability: Voice stability setting (0.0 to 1.0).
        similarity_boost: Voice similarity boost setting (0.0 to 1.0).
        timeout: Request timeout in seconds.
        return_timestamps: If True, requests and returns timestamp data along with saving audio.

    Returns:
        - bool: True if audio was saved successfully (when return_timestamps=False).
        - Dict[str, Any]: Dictionary containing timestamp data if successful (when return_timestamps=True).
        - bool: False if any error occurred during the process.
    """
    # --- Input Validation ---
    if not text or text.isspace():
        _logger.error("Cannot generate audio: Input text is empty.")
        return False
    if not api_key:
        _logger.error("Cannot generate audio: Missing ElevenLabs API key.")
        return False
    # Basic check on output path existence (directory part)
    output_dir = os.path.dirname(output_mp3_path)
    if not os.path.isdir(output_dir):
         _logger.error(f"Cannot generate audio: Output directory does not exist: {output_dir}")
         os.makedirs(output_dir, exist_ok=True)

    # Check if the output directory is writable
    if not os.access(output_dir, os.W_OK):
         _logger.error(f"Cannot generate audio: Output directory is not writable: {output_dir}")
         return False


    resolved_voice_id = voice_id or DEFAULT_VOICE_ID
    _logger.info(f"Requesting TTS from ElevenLabs for voice '{resolved_voice_id}'. Output: {output_mp3_path}")

    # Determine endpoint and headers based on timestamp request
    if return_timestamps:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{resolved_voice_id}/with-timestamps"
        headers = {
            "xi-api-key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": APP_USER_AGENT
        }
        _logger.debug("Requesting TTS with timestamps.")
    else:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{resolved_voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "User-Agent": APP_USER_AGENT
        }
        _logger.debug("Requesting standard TTS audio.")

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    # --- API Request with Retries ---
    max_retries = 3
    response = None
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            _logger.debug(f"TTS Request Attempt {attempt}/{max_retries} to {url}")
            # _logger.debug(f"Payload (partial): {json.dumps(payload, indent=2)[:500]}...") # Avoid logging full text
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            _logger.debug(f"TTS response status code: {response.status_code}")
            last_exception = None
            break

        except requests.exceptions.Timeout as e:
            last_exception = e
            _logger.warning(f"TTS Attempt {attempt} timed out after {timeout}s: {e}")
        except requests.exceptions.HTTPError as e:
            last_exception = e
            error_content = response.text[:200] if response is not None else "No response content"
            _logger.warning(f"TTS Attempt {attempt} HTTP error: {e.status_code}. Response: {error_content}...")
            if response is not None:
                 if response.status_code == 401:
                      _logger.error("ElevenLabs API key is invalid or unauthorized.")
                      return False
                 elif response.status_code == 422:
                      _logger.error(f"ElevenLabs returned 422 Unprocessable Entity. Check input text/voice settings. Response: {error_content}")
                      return False
        except requests.exceptions.RequestException as e:
            last_exception = e
            _logger.warning(f"TTS Attempt {attempt} request exception: {e}")
        except Exception as e:
            last_exception = e
            _logger.exception(f"TTS Attempt {attempt} unexpected error: {e}")

        if attempt < max_retries:
            wait_time = 1 * (2**(attempt-1))
            _logger.info(f"Retrying TTS request in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            _logger.error(f"All {max_retries} TTS attempts failed. Last error: {last_exception}")
            return False


    # --- Process Response ---
    try:
        if return_timestamps:
            response_data = response.json()
            audio_b64 = response_data.get("audio")
            timestamps = response_data.get("timestamps")

            if not audio_b64:
                _logger.error("No 'audio' field found in ElevenLabs timestamp response.")
                return False

            # Decode base64 audio and save
            audio_bytes = base64.b64decode(audio_b64)
            with open(output_mp3_path, "wb") as f:
                f.write(audio_bytes)

            _logger.info(f"Audio with timestamps successfully saved to {output_mp3_path}")
            return {"timestamps": timestamps} if timestamps else {}

        else:
            with open(output_mp3_path, "wb") as f:
                # Stream content to handle potentially large files
                for chunk in response.iter_content(chunk_size=8192):
                     f.write(chunk)
            _logger.info(f"Audio successfully saved to {output_mp3_path}")
            return True

    except (json.JSONDecodeError, base64.binascii.Error) as e:
         _logger.error(f"Error processing ElevenLabs response data: {e}")
         return False
    except IOError as e:
        _logger.error(f"Failed to write audio file to {output_mp3_path}: {e}")
        return False
    except Exception as e:
        _logger.exception(f"Unexpected error processing TTS response or saving file: {e}")
        return False