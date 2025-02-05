import os
import json
import logging
import datetime
import requests
import re  # Imported for chain-of-thought extraction

from viralStoryGenerator.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.prompts.prompts import get_storyboard_prompt

# Configure logging (if not already configured globally)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_storyboard_structure(story, llm_endpoint, model, temperature):
    """
    Uses the LLM to produce a storyboard breakdown in JSON format.
    The JSON must have a "scenes" key which is a list of scenes.
    Each scene should include:
      - scene_number: sequential integer starting at 1.
      - narration_text: portion of the story for narration.
      - image_prompt: an ultra-detailed description for DALL·E 3 image generation.
      - duration: estimated scene duration (in seconds), calculated from narration text.
    """
    prompt = get_storyboard_prompt(story).strip()

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that strictly follows instructions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(llm_endpoint, headers=headers, data=json.dumps(data))
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to generate storyboard structure: {e}")
        return None

    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"]

    # Remove chain-of-thought if present.
    match = re.search(r'(<think>.*?</think>)', content, re.DOTALL)
    if match:
        thinking = match.group(1)
        # logging.info(f"Chain-of-thought found: {thinking}")
        content = content.replace(thinking, "").strip()

    # Remove markdown code block markers if present.
    if content.startswith("```"):
        code_block_match = re.search(r"^```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if code_block_match:
            content = code_block_match.group(1).strip()

    if not content:
        logging.error("LLM response content is empty after removing chain-of-thought and markdown formatting.")
        return None

    try:
        storyboard_data = json.loads(content)

    except Exception as e:
        logging.error(f"Failed to parse storyboard JSON: {e}")
        return None

    return storyboard_data

def generate_dalle_image(image_prompt, output_image_path, openai_api_key):
    """
    Uses the DALL·E 3 API to generate an image based on the given image_prompt.
    Downloads and saves the image to output_image_path.
    """
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": image_prompt,
        "n": 1,
        "size": "1024x1024"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"DALL·E 3 API call failed: {e}")
        return None

    try:
        response_data = response.json()
        image_url = response_data["data"][0]["url"]
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        with open(output_image_path, "wb") as f:
            f.write(image_response.content)
        logging.info(f"Image saved to {output_image_path}")
        return output_image_path
    except Exception as e:
        logging.error(f"Failed to download DALL·E image: {e}")
        return None

def generate_storyboard(story, topic, llm_endpoint, model, temperature, voice_id=None):
    """
    Generates a storyboard from the given story by:
      - Splitting the story into scenes with exact narration text, image prompts, and durations.
      - Generating an image for each scene.
      - Combining all scene narrations into one text and generating a single audio file using ElevenLabs TTS with timing.
      - Storing scene start times and timing details from the TTS API.
    """
    storyboard_data = generate_storyboard_structure(story, llm_endpoint, model, temperature)
    if not storyboard_data or "scenes" not in storyboard_data:
        logging.error("Storyboard structure generation failed.")
        return None

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar().week
    folder_path = os.path.join("GeneratedStoryboards", f"Week{week_num}")
    os.makedirs(folder_path, exist_ok=True)

    # Get API keys from ENV
    openai_api_key = os.environ.get("OPENAI_API_KEY", "sk-svcacct-D-bcMpovTDwBpAGHe-EmgUwwdkz7q6JChfWgOx6dS97A4v3IhhzF21Uqyc96Z755LL1fObPT3BlbkFJuanvq8Dbe_FCs9YmheMfHsR1lRWzHoPsr_Aa-1pnd8rYzDImCrsCFLDu4JoGAKghkaS5aAA")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "sk_15cb1ec5322909d636dd3afb9223dd65578013807895d481")
    if not openai_api_key:
        logging.error("No OpenAI API key found for DALL·E 3 image generation.")
    if not elevenlabs_api_key:
        logging.error("No ElevenLabs API key found for audio generation.")

    cumulative_time = 0
    scene_texts = []
    for scene in storyboard_data["scenes"]:
        scene_number = scene.get("scene_number")
        narration_text = scene.get("narration_text", "").strip()
        image_prompt = scene.get("image_prompt", "").strip()
        duration = scene.get("duration")
        # Compute duration if not provided (150 wpm ≈ 2.5 words per second).
        if not duration:
            words = narration_text.split()
            duration = round((len(words) / 150) * 60)
            scene["duration"] = duration

        scene["start_time"] = cumulative_time
        cumulative_time += duration

        # Collect the scene narration text.
        scene_texts.append(narration_text)

        # Generate an image for this scene using DALL·E 3.
        if openai_api_key and image_prompt:
            image_filename = f"{topic}_scene_{scene_number}_{date_str}.png"
            image_path = os.path.join(folder_path, image_filename)
            result_img = generate_dalle_image(image_prompt, image_path, openai_api_key)
            scene["image_file"] = result_img if result_img else ""
        else:
            scene["image_file"] = ""

    # Reassemble the full narration from scenes.
    combined_narration = "\n".join(scene_texts).strip()
    # Check if the combined narration exactly matches the original story.
    if combined_narration != story.strip():
        logging.warning("The combined scene narration does not exactly match the original story text.")

    logging.info(f"Generated storyboard for '{topic}' with {len(storyboard_data['scenes'])} scenes.")
    logging.info(f"Total duration: {cumulative_time} seconds")
    logging.info(f"Length: {len(combined_narration)}. Combined_narration: {combined_narration}")
    # Generate a single combined audio file for all scenes using ElevenLabs TTS with timing info.
    if elevenlabs_api_key and combined_narration:
        audio_filename = f"{topic}_combined_{date_str}.mp3"
        audio_path = os.path.join(folder_path, audio_filename)
        audio_result = generate_elevenlabs_audio(
            text=combined_narration,
            api_key=elevenlabs_api_key,
            output_mp3_path=audio_path,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            stability=0.5,
            similarity_boost=0.75,
            return_timestamps=True  # Request precise character-level timing info.
        )
        if audio_result:
            storyboard_data["audio_file"] = audio_path
            if isinstance(audio_result, dict) and "timestamps" in audio_result:
                storyboard_data["scene_timestamps"] = audio_result["timestamps"]
            else:
                storyboard_data["scene_timestamps"] = {}
        else:
            storyboard_data["audio_file"] = ""
            storyboard_data["scene_timestamps"] = {}
    else:
        storyboard_data["audio_file"] = ""
        storyboard_data["scene_timestamps"] = {}

    # Save the complete storyboard JSON.
    storyboard_json_filename = f"{topic}_storyboard_{date_str}.json"
    storyboard_json_path = os.path.join(folder_path, storyboard_json_filename)
    try:
        with open(storyboard_json_path, "w", encoding="utf-8") as f:
            json.dump(storyboard_data, f, indent=4)
        logging.info(f"Storyboard JSON saved to {storyboard_json_path}")
    except Exception as e:
        logging.error(f"Failed to save storyboard JSON: {e}")

    return storyboard_data
