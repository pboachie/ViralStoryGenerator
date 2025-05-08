# viralStoryGenerator/src/cli.py
import argparse
import time
import os
import datetime
import json

from viralStoryGenerator.src.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config
from viralStoryGenerator.utils.storage_manager import storage_manager

# Directory where failed audio generations are queued
AUDIO_QUEUE_DIR = os.environ.get("AUDIO_QUEUE_DIR", "Output/AudioQueue")
appconfig = config.config

def _save_story_output(result, topic, voice_id=None):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar().week

    story_base_dir = storage_manager._get_storage_dir("story")
    folder_path = os.path.join(story_base_dir, f"Week{week_num}")
    os.makedirs(folder_path, exist_ok=True)

    # Build text filename
    safe_topic = topic.replace("/", "_").replace("\\", "_")
    txt_file_name = f"{safe_topic} - {date_str}.txt"
    txt_file_path = os.path.join(folder_path, txt_file_name)

    # Write out the story & description
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("### Story Script:\n")
        f.write(result.get("story", ""))
        f.write("\n\n### Video Description:\n")
        f.write(result.get("video_description", ""))

    _logger.info(f"Story saved to {txt_file_path}")

    # Also generate audio from the story (if we have content)
    story_text = result.get("story", "")
    if story_text.strip():
        # Build MP3 filename (same base name, .mp3 extension)
        base_name = os.path.splitext(txt_file_name)[0]  # e.g. "MyTopic - 2025-02-02"
        mp3_file_path = os.path.join(folder_path, f"{base_name}.mp3")

        # We'll assume your ElevenLabs API key is stored somewhere
        api_key = appconfig.elevenLabs.API_KEY
        if not api_key:
            _logger.warning("No ElevenLabs API Key found. Skipping TTS generation.")
            return

        # Use the provided voice_id (if any) or default to None
        success = generate_elevenlabs_audio(
            text=story_text,
            api_key=api_key,
            output_mp3_path=mp3_file_path,
            voice_id=voice_id,
            model_id=appconfig.elevenLabs.DEFAULT_MODEL_ID,
            stability=appconfig.elevenLabs.DEFAULT_STABILITY,
            similarity_boost=appconfig.elevenLabs.DEFAULT_SIMILARITY_BOOST
        )
        if success:
            _logger.info(f"Audio TTS saved to {mp3_file_path}")
        else:
            _logger.warning("Audio generation failed. Queueing for later re-generation.")
            # Prepare metadata for retrying audio generation later
            metadata = {
                "topic": topic,
                "story": story_text,
                "mp3_file_path": mp3_file_path,
                "voice_id": voice_id,
                "model_id": appconfig.elevenLabs.DEFAULT_MODEL_ID,
                "stability": appconfig.elevenLabs.DEFAULT_STABILITY,
                "similarity_boost": appconfig.elevenLabs.DEFAULT_SIMILARITY_BOOST,
                "attempts": 1,
                "timestamp": datetime.datetime.now().isoformat()
            }
            queue_failed_audio(metadata)


def queue_failed_audio(metadata):
    """
    Saves the metadata for a failed audio generation attempt into the AUDIO_QUEUE_DIR.
    """
    os.makedirs(AUDIO_QUEUE_DIR, exist_ok=True)
    # Create a safe filename using the topic and current timestamp.
    safe_topic = metadata.get("topic", "untitled").replace("/", "_").replace("\\", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{safe_topic}_{timestamp}.json"
    file_path = os.path.join(AUDIO_QUEUE_DIR, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        _logger.info(f"Queued failed audio generation to {file_path}")
    except Exception as e:
        _logger.error(f"Failed to write queue file {file_path}: {e}")


def process_audio_queue():
    """
    Scans the AUDIO_QUEUE_DIR for queued audio jobs and attempts to re-generate audio.
    On success, the queued file is removed.
    """
    if not os.path.isdir(AUDIO_QUEUE_DIR):
        return

    for filename in os.listdir(AUDIO_QUEUE_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(AUDIO_QUEUE_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                _logger.error(f"Error reading queued file {file_path}: {e}")
                continue

            api_key = appconfig.elevenLabs.API_KEY
            if not api_key:
                _logger.error("No ElevenLabs API Key found. Skipping queued audio generation.")
                break

            _logger.info(f"Attempting queued audio generation for {metadata.get('mp3_file_path')}")
            success = generate_elevenlabs_audio(
                text=metadata["story"],
                api_key=api_key,
                output_mp3_path=metadata["mp3_file_path"],
                voice_id=metadata.get("voice_id"),
                model_id=metadata.get("model_id", appconfig.elevenLabs.DEFAULT_MODEL_ID),
                stability=metadata.get("stability", appconfig.elevenLabs.DEFAULT_STABILITY),
                similarity_boost=metadata.get("similarity_boost", appconfig.elevenLabs.DEFAULT_SIMILARITY_BOOST)
            )
            if success:
                _logger.info(f"Queued audio generated successfully for {metadata.get('mp3_file_path')}. Removing from queue.")
                try:
                    os.remove(file_path)
                except Exception as e:
                    _logger.error(f"Could not remove queue file {file_path}: {e}")
            else:
                _logger.warning(f"Queued audio generation failed for {metadata.get('mp3_file_path')}. Will retry on next run.")


def _read_sources_from_folder(folder_path):
    """
    Reads all files from `folder_path` (e.g., .txt, .md, or any extension),
    concatenates their contents, and returns one combined string.
    """
    combined_texts = []
    if not os.path.isdir(folder_path):
        _logger.warning(f"Sources folder does not exist: {folder_path}")
        return ""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        combined_texts.append(text)
            except Exception as e:
                _logger.error(f"Failed to read {file_path}: {e}")

    # Combine them into one big block of text separated by double newlines
    return "\n\n".join(combined_texts)


def cli_main():
    parser = argparse.ArgumentParser(
        description="Generate short, informal story scripts via a local LLM endpoint."
    )
    parser.add_argument("--topic",type=str, required=True, help="Topic for the story script")
    parser.add_argument("--sources-folder", type=str, default=appconfig.SOURCES_FOLDER, help="Folder with source files")

    parser.add_argument("--endpoint", type=str, default=appconfig.llm.ENDPOINT,
                        help="Local LLM API endpoint.")
    parser.add_argument("--model", type=str, default=appconfig.llm.MODEL,
                        help="Which model to use for generating the story.")
    temperature_default = 0.7
    try:
        temperature_default = float(appconfig.llm.TEMPERATURE)
    except (ValueError, TypeError):
        _logger.warning("Invalid LLM_TEMPRATURE configuration variable. Using default value of 0.7.")

    parser.add_argument("--temperature", type=float, default=temperature_default,
                        metavar="T", choices=[0.0, 0.2, 0.5, 0.7, 1.0],
                        help="Sampling temperature (higher => more random).")
    parser.add_argument("--show-thinking", action="store_true", default=appconfig.llm.SHOW_THINKING,
                        help="If passed, print chain-of-thought (if available).")
    parser.add_argument("--chunk-size", type=int, default=os.getenv("LLM_CHUNK_SIZE", 5000),
                        help="Word chunk size for splitting sources.")
    # Optionally add --voice-id if you want user to specify a voice
    parser.add_argument("--voice-id", type=str, default=os.getenv("ELEVENLABS_VOICE_ID", None), help="ElevenLabs voice ID override")
    args = parser.parse_args()

    if not args.model:
        raise ValueError("The --model argument must be provided, either via command line or the LLM_MODEL environment variable.")

    start_exec = time.time()

    # === CLI is now deprecated for story generation as logic moved to API worker ===
    _logger.warning("CLI execution for story generation is deprecated and may not reflect the full API/worker workflow (like RAG).")
    print("\nWARNING: CLI execution is deprecated. Use the API endpoint (/api/generate) for full functionality.\n")

    # --- The following logic is based on the OLD workflow and will likely NOT work correctly ---
    # --- It is kept here for reference or potential future CLI adaptation ---

    # # 1) Read sources (remains the same)
    # raw_sources = None
    # if args.sources_folder and len(args.sources_folder) >= 1:
    #     _logger.debug(f"Reading all files in folder '{args.sources_folder}' for sources...")
    #     raw_sources = _read_sources_from_folder(args.sources_folder)
    # else:
    #      _logger.debug("No sources folder specified.")

    # # 2) Chunkify & Summarize (OLD METHOD - REMOVED)
    # cleansed_sources = ""
    # if raw_sources and raw_sources.strip():
    #     _logger.warning("CLI is using deprecated chunkify_and_summarize. RAG is not implemented here.")
    #     # This function no longer exists or works as expected
    #     # cleansed_sources = chunkify_and_summarize(...)
    #     _logger.error("chunkify_and_summarize is deprecated/removed. Cannot proceed with cleansing in CLI.")
    # else:
    #     _logger.debug("No sources found. Skipping cleansing step.")

    # # 3) Generate story script (OLD METHOD - REMOVED)
    # _logger.warning("CLI is using deprecated generate_story_script. RAG is not implemented here.")
    # # This function no longer exists or works as expected
    # # result = generate_story_script(...)
    # result = {"story": "CLI Generation Disabled", "video_description": "#CLI #Disabled"}
    # print(f"=== STORY GENERATION RESULT (CLI Disabled) ===\n{result}")


    # # 4) Print thinking (OLD METHOD)
    # # ...

    # # 5) Print final story (OLD METHOD)
    # # ...

    # # 6) Save outputs (OLD METHOD)
    # # _save_story_output(result, args.topic, voice_id=args.voice_id)

    total_exec_time = time.time() - start_exec
    _logger.info(f"Total execution time (CLI): {total_exec_time:.2f} seconds")

if __name__ == "__main__":
    cli_main()
