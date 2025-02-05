import requests
import logging
import time
import json

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
    timeout=90  # Timeout parameter in seconds
):
    """
    Calls ElevenLabs TTS API to convert 'text' into speech, saving the result as MP3 at 'output_mp3_path'.
    Additional request and response objects are dumped for better debugging.
    """
    if not voice_id:
        voice_id = DEFAULT_VOICE_ID

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
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
            break
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

    try:
        with open(output_mp3_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Audio successfully saved to {output_mp3_path}")
    except Exception as e:
        logging.error(f"Failed to save audio file at {output_mp3_path}: {e}")
        return False

    return True
