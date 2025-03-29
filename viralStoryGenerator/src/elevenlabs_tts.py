import requests
import logging
import time
import json
import base64

DEFAULT_VOICE_ID = "JZ3e95uoTACVf6tXaaEi"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"

def generate_elevenlabs_audio(
    text,
    api_key,
    output_mp3_path,
    voice_id=None,
    model_id=DEFAULT_MODEL_ID,
    stability=0.5,
    similarity_boost=0.75,
    timeout=90,  # Timeout parameter in seconds
    return_timestamps=False
):
    """
    Calls ElevenLabs TTS API to convert 'text' into speech.

    If return_timestamps is False, the audio is saved as MP3 from the basic endpoint.
    If return_timestamps is True, the /with-timestamps endpoint is used and the response is
    expected to include precise character-level timing information. In that case, the function
    decodes the base64 audio from the JSON response, saves the file, and returns a dictionary
    containing the timestamps.
    """
    if not voice_id:
        voice_id = DEFAULT_VOICE_ID

    # Use the /with-timestamps endpoint if you want timing data.
    if return_timestamps:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        # Request JSON response (which contains both audio and timestamp data)
        headers = {
            "xi-api-key": api_key,
            "Accept": "application/json",    # expecting JSON with timing info
            "Content-Type": "application/json"
        }
    else:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Accept": "audio/mpeg",          # raw MP3 audio
            "Content-Type": "application/json"
        }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    max_retries = 3
    response = None

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_retries}: Sending TTS request to ElevenLabs for voice_id='{voice_id}' with timeout={timeout}s...")
            logging.debug(f"URL: {url}")
            logging.debug(f"Headers: {headers}")
            logging.debug(f"Payload: {json.dumps(payload, indent=2)}")
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            logging.debug(f"TTS response status code: {response.status_code}")
            break  # success; exit retry loop
        except requests.exceptions.Timeout as e:
            logging.error(f"Attempt {attempt} timed out after {timeout}s: {e}")
        except requests.exceptions.HTTPError as e:
            error_content = response.text if response is not None else "No response"
            logging.error(f"HTTP error during attempt {attempt}: {e}. Response content: {error_content}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception during attempt {attempt}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during attempt {attempt}: {e}")

        if attempt == max_retries:
            logging.error("All attempts failed, giving up on audio generation.")
            return False

        logging.info("Retrying TTS request...")
        time.sleep(1)

    # Process the response based on whether we asked for timestamps.
    if return_timestamps:
        try:
            # Expecting a JSON response with both audio and timestamps info.
            response_data = response.json()
            audio_b64 = response_data.get("audio")
            timestamps = response_data.get("timestamps")
            if not audio_b64:
                logging.error("No audio data returned in the response.")
                return False

            # Decode the base64-encoded audio and save it.
            audio_bytes = base64.b64decode(audio_b64)
            with open(output_mp3_path, "wb") as f:
                f.write(audio_bytes)
            logging.info(f"Audio with timestamps successfully saved to {output_mp3_path}")
            return {"timestamps": timestamps}
        except Exception as e:
            logging.error(f"Error processing TTS response with timestamps: {e}")
            return False
    else:
        try:
            with open(output_mp3_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Audio successfully saved to {output_mp3_path}")
        except Exception as e:
            logging.error(f"Failed to save audio file at {output_mp3_path}: {e}")
            return False

        return True
