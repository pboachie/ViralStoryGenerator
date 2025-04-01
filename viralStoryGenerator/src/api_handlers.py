# viralStoryGenerator/src/api_handlers.py
import time
import os
import datetime
import json
import uuid
from typing import Dict, Any, Optional, List

from viralStoryGenerator.models import (
    StoryGenerationResult,
    JobStatusResponse
)
from viralStoryGenerator.src.llm import generate_story_script
from viralStoryGenerator.src.source_cleanser import chunkify_and_summarize
from viralStoryGenerator.src.elevenlabs_tts import generate_elevenlabs_audio
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig
from viralStoryGenerator.utils.redis_manager import RedisManager

# Directory where failed audio generations are queued
AUDIO_QUEUE_DIR = appconfig.AUDIO_QUEUE_DIR
# Redis manager for task queue
redis_manager = RedisManager() if appconfig.redis.ENABLED else None

class StoryTask:
    """Class to manage story generation tasks"""
    def __init__(self, task_id: str, topic: str, sources: Optional[str] = None,
                 voice_id: Optional[str] = None, status: str = "pending",
                 created_at: Optional[str] = None):
        self.task_id = task_id
        self.topic = topic
        self.sources = sources
        self.voice_id = voice_id
        self.status = status
        self.created_at = created_at or datetime.datetime.now().isoformat()
        self.result = None
        self.error = None
        self.file_paths = {
            "story": None,
            "audio": None,
            "storyboard": None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "status": self.status,
            "created_at": self.created_at,
            "file_paths": self.file_paths,
            "error": self.error
        }

def _save_story_output(result, topic, voice_id=None) -> Dict[str, str]:
    """
    Save story output to files and return paths to the created files.

    Returns:
        Dict with paths to generated files
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar().week

    folder_path = os.path.join("GeneratedStories", f"Week{week_num}")
    os.makedirs(folder_path, exist_ok=True)

    # Build text filename
    safe_topic = topic.replace("/", "_").replace("\\", "_")
    txt_file_name = f"{safe_topic} - {date_str}.txt"
    txt_file_path = os.path.join(folder_path, txt_file_name)

    # Prepare result paths
    result_paths = {
        "story": txt_file_path,
        "audio": None,
        "storyboard": None
    }

    # Write out the story & description
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("### Story Script:\n")
        f.write(result.get("story", ""))
        f.write("\n\n### Video Description:\n")
        f.write(result.get("video_description", ""))

    _logger.info(f"Story saved to {txt_file_path}")

    # Save storyboard if available
    if result.get("storyboard"):
        storyboard_file = f"{safe_topic} - {date_str}.json"
        storyboard_path = os.path.join(folder_path, storyboard_file)

        with open(storyboard_path, "w", encoding="utf-8") as f:
            json.dump(result.get("storyboard"), f, indent=2)

        _logger.info(f"Storyboard saved to {storyboard_path}")
        result_paths["storyboard"] = storyboard_path

    # Also generate audio from the story (if we have content)
    story_text = result.get("story", "")
    if story_text.strip():
        # Build MP3 filename (same base name, .mp3 extension)
        base_name = os.path.splitext(txt_file_name)[0]  # e.g. "MyTopic - 2025-02-02"
        mp3_file_path = os.path.join(folder_path, f"{base_name}.mp3")
        result_paths["audio"] = mp3_file_path

        # We'll assume your ElevenLabs API key is stored somewhere
        api_key = appconfig.elevenLabs.API_KEY
        if not api_key:
            _logger.warning("No ElevenLabs API Key found. Skipping TTS generation.")
            return result_paths

        # Use the provided voice_id (if any) or default
        success = generate_elevenlabs_audio(
            text=story_text,
            api_key=api_key,
            output_mp3_path=mp3_file_path,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            stability=0.5,
            similarity_boost=0.75
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
                "model_id": "eleven_multilingual_v2",
                "stability": 0.5,
                "similarity_boost": 0.75,
                "attempts": 1,
                "timestamp": datetime.datetime.now().isoformat()
            }
            queue_failed_audio(metadata)

    return result_paths


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
                model_id=metadata.get("model_id", "eleven_multilingual_v2"),
                stability=metadata.get("stability", 0.5),
                similarity_boost=metadata.get("similarity_boost", 0.75)
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


def create_story_task(topic: str, sources_folder: Optional[str] = None,
                     voice_id: Optional[str] = None) -> Dict[str, Any]:
    """
    API handler to create a new story generation task.

    Args:
        topic: The topic for the story
        sources_folder: Optional folder with source files
        voice_id: Optional voice ID for ElevenLabs

    Returns:
        Dict with task information including task_id
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Create task object
    task = StoryTask(
        task_id=task_id,
        topic=topic,
        voice_id=voice_id
    )

    # If using Redis, queue the task for processing
    if redis_manager:
        task_data = {
            "task_id": task_id,
            "topic": topic,
            "sources_folder": sources_folder,
            "voice_id": voice_id
        }
        redis_manager.enqueue_task(task_data)
        _logger.info(f"Task {task_id} queued for processing")
    else:
        # Process the task immediately (synchronously - not ideal for production)
        _logger.warning("Redis disabled. Processing task synchronously (not recommended for production)")
        # Start processing in background thread
        import threading
        thread = threading.Thread(
            target=process_story_task,
            args=(task_id, topic, sources_folder, voice_id)
        )
        thread.daemon = True
        thread.start()

    return task.to_dict()


def process_story_task(task_id: str, topic: str, sources_folder: Optional[str] = None,
                      voice_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a story generation task.

    Args:
        task_id: Unique task ID
        topic: Topic for the story
        sources_folder: Optional folder with source files
        voice_id: Optional voice ID for ElevenLabs

    Returns:
        Dict with task results
    """
    start_exec = time.time()
    try:
        # Update task status
        if redis_manager:
            redis_manager.update_task_status(task_id, "processing")

        # Process any queued failed audio generations
        process_audio_queue()

        raw_sources = None
        # Read all the files from the sources folder into one combined text
        if sources_folder and len(sources_folder) >= 1:
            _logger.debug(f"Reading all files in folder '{sources_folder}' for sources...")
            raw_sources = _read_sources_from_folder(sources_folder)

        # Chunkify & Summarize them into a single cohesive summary
        if raw_sources and raw_sources.strip():
            _logger.debug("Splitting & summarizing sources via LLM (multi-chunk)...")
            cleansed_sources = chunkify_and_summarize(
                raw_sources=raw_sources,
                endpoint=appconfig.llm.ENDPOINT,
                model=appconfig.llm.MODEL,
                temperature=appconfig.llm.TEMPERATURE,
                chunk_size=appconfig.llm.CHUNK_SIZE
            )
            _logger.debug("Sources cleansed. Proceeding with story generation...")
        else:
            _logger.debug("No sources found. Skipping cleansing step.")
            cleansed_sources = ""

        # Generate the story script from these cleansed/merged sources
        result = generate_story_script(
            topic=topic,
            sources=cleansed_sources,
            endpoint=appconfig.llm.ENDPOINT,
            model=appconfig.llm.MODEL,
            temperature=appconfig.llm.TEMPERATURE,
            show_thinking=appconfig.llm.SHOW_THINKING
        )

        # Save the final outputs including audio generation
        file_paths = _save_story_output(result, topic, voice_id=voice_id)

        # Prepare the final result
        task_result = {
            "task_id": task_id,
            "topic": topic,
            "status": "completed",
            "result": {
                "story": result.get("story", ""),
                "video_description": result.get("video_description", ""),
                "storyboard": result.get("storyboard", {}),
            },
            "file_paths": file_paths,
            "processing_time": time.time() - start_exec
        }

        # Convert to StoryGenerationResult model if needed
        story_result = {
            "story_script": result.get("story", ""),
            "storyboard": result.get("storyboard", {}),
            "sources": [sources_folder] if sources_folder else []
        }

        # Add audio URL if available
        if file_paths.get("audio"):
            story_result["audio_url"] = f"/audio/{os.path.basename(file_paths['audio'])}"

        # Update Redis if enabled
        if redis_manager:
            redis_manager.update_task_result(task_id, task_result)

        return task_result

    except Exception as e:
        error_msg = f"Error processing task {task_id}: {str(e)}"
        _logger.error(error_msg)

        # Update task status in Redis
        if redis_manager:
            redis_manager.update_task_error(task_id, error_msg)

        return {
            "task_id": task_id,
            "topic": topic,
            "status": "failed",
            "error": error_msg
        }


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a task.

    Args:
        task_id: The ID of the task to check

    Returns:
        Dict with task status information that can be converted to JobStatusResponse
    """
    if not redis_manager:
        return {"error": "Task tracking unavailable without Redis", "status": "failed"}

    result = redis_manager.get_task_status(task_id)

    # Ensure the result matches the JobStatusResponse model structure
    if result and "status" in result:
        # Make sure all required fields from JobStatusResponse are present
        if "message" not in result:
            result["message"] = None
        if result["status"] == "completed" and "result" in result:
            # Extract story data from the result
            result["story_script"] = result.get("result", {}).get("story")
            result["storyboard"] = result.get("result", {}).get("storyboard")

            # Add audio URL if available
            if "file_paths" in result and result["file_paths"].get("audio"):
                result["audio_url"] = f"/audio/{os.path.basename(result['file_paths']['audio'])}"
            else:
                result["audio_url"] = None

            # Add sources data
            result["sources"] = [result.get("sources_folder")] if result.get("sources_folder") else []

    return result