# viralStoryGenerator/elevenlabs_tts.py

import requests
import logging

DEFAULT_VOICE_ID = "JZ3e95uoTACVf6tXaaEi"  # Johnny - Upbeat Professional American...
DEFAULT_MODEL_ID = "eleven_multilingual_v2"

def generate_elevenlabs_audio(
    text,
    api_key,
    output_mp3_path,
    voice_id=None,
    model_id=DEFAULT_MODEL_ID,
    stability=0.5,
    similarity_boost=0.75
):
    """
    Calls ElevenLabs TTS API to convert 'text' into speech, saving the result as MP3 at 'output_mp3_path'.
    - If 'voice_id' is not provided, uses a default.
    - Adjust stability/similarity_boost as you like.
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

    logging.info(f"Sending TTS request to ElevenLabs for voice_id='{voice_id}'...")
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"ElevenLabs TTS request failed: {e}")
        return False

    # Write the received MP3 bytes to the specified file
    with open(output_mp3_path, "wb") as f:
        f.write(response.content)

    logging.info(f"Audio successfully saved to {output_mp3_path}")
    return True
