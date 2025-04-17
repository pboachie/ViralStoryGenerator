# viralStoryGenerator/src/api_handlers.py
import tempfile
import time
import os
import datetime
import json
import uuid
import re # For sanitization
from typing import Dict, Any, Optional, List

from viralStoryGenerator.models import (
    StoryGenerationResult,
    JobStatusResponse
)
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.utils.security import is_file_in_directory, validate_path_component, sanitize_for_filename

def get_message_broker() -> RedisMessageBroker:
    """Get Redis message broker for API handlers"""
    redis_url = "redis://" + appconfig.redis.HOST + ":" + str(appconfig.redis.PORT)
    return RedisMessageBroker(redis_url=redis_url, stream_name="api_jobs")

class StoryTask:
    """Basic representation of a task state for API response."""
    def __init__(self, task_id: str, topic: str, status: str = "pending", created_at: Optional[str] = None):
        self.task_id = task_id
        self.topic = topic
        self.status = status
        self.created_at = created_at or datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "status": self.status,
            "created_at": self.created_at,
        }

# ---- API Facing Functions ----

def create_story_task(topic: str, sources_folder: Optional[str] = None,
                     voice_id: Optional[str] = None) -> Dict[str, Any]:
    """
    API handler logic to create a new story generation task.
    Validates inputs and queues the task via Redis Streams.
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Basic task info for immediate response
    task_response = StoryTask(
        task_id=task_id,
        topic=topic
    ).to_dict()

    # Prepare job data for Redis Streams
    task_data_payload = {
        "job_id": task_id,
        "topic": topic,
        "sources_folder": sources_folder,
        "voice_id": voice_id,
        "request_time": time.time()
    }

    # Use Redis Streams to publish the task
    try:
        message_broker = get_message_broker()

        # Ensure the stream exists
        message_broker.ensure_stream_exists("api_jobs")

        # Publish the message to the stream
        message_id = message_broker.publish_message(task_data_payload)
        success = message_id is not None
    except Exception as e:
        _logger.error(f"Failed to publish task to Redis Stream: {e}")
        success = False

    if success:
        _logger.info(f"Task {task_id} queued successfully via Redis Stream for topic: '{topic}'.")
        task_response["status"] = "queued"
    else:
        _logger.error(f"Failed to queue task {task_id} via Redis Stream.")
        raise RuntimeError("Failed to add task to Redis Stream.")

    return task_response


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a task from Redis Streams.
    Returns a dictionary conforming to JobStatusResponse or None if not found.
    """
    try:
        # Check in Redis Streams
        try:
            message_broker = get_message_broker()
            stream_status = message_broker.get_job_status(task_id)

            if stream_status:
                # Convert stream status format to the expected format
                status_data = {
                    "status": stream_status.get("status", "pending"),
                    "job_id": stream_status.get("job_id"),
                    "message": stream_status.get("message", "Job found in Redis Stream"),
                    "created_at": stream_status.get("timestamp"),
                    "updated_at": stream_status.get("timestamp"),
                    "story_script": stream_status.get("story_script"),
                    "storyboard": stream_status.get("storyboard"),
                    "audio_url": stream_status.get("audio_url"),
                    "sources": stream_status.get("sources"),
                    "error": stream_status.get("error")
                }

                response = JobStatusResponse(**status_data).model_dump(exclude_none=True)
                return response
            else:
                # Task ID not found in Redis
                _logger.debug(f"Task ID {task_id} not found in Redis Streams.")
                return None

        except Exception as e:
            _logger.warning(f"Error checking Redis Stream for job {task_id}: {e}")
            return {"status": "error", "message": f"Error retrieving job status: {str(e)}", "task_id": task_id}

    except Exception as e:
        _logger.exception(f"Error retrieving task status for {task_id}: {e}")
        return {"status": "error", "message": "Failed to retrieve task status.", "task_id": task_id}


# --- Audio Queue Processing (Called periodically or by cleanup task) ---
def process_audio_queue():
    """
    Scans AUDIO_QUEUE_DIR for failed jobs and retries audio generation.
    NOTE: This relies on local filesystem queuing, which is less robust than Redis.
    TODO: Consider integrating audio retry logic into the Redis task flow if possible.
    """
    audio_queue_dir = appconfig.AUDIO_QUEUE_DIR
    if not os.path.isdir(audio_queue_dir):
        _logger.debug("Audio queue directory not found, skipping processing.")
        return

    _logger.info(f"Checking for queued audio jobs in: {audio_queue_dir}")
    api_key = appconfig.elevenLabs.API_KEY
    if not api_key:
        _logger.error("Cannot process audio queue: ElevenLabs API Key not configured.")
        return

    processed_count = 0
    failed_count = 0
    from .elevenlabs_tts import generate_elevenlabs_audio # Local import

    for filename in os.listdir(audio_queue_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(audio_queue_dir, filename)
            _logger.debug(f"Processing queued audio file: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Basic validation of metadata
                if not all(k in metadata for k in ["story", "mp3_file_path"]):
                     _logger.warning(f"Skipping invalid queue file {filename}: missing required keys.")
                     # Optionally move/delete invalid file
                     continue

                # Attempt regeneration
                success = generate_elevenlabs_audio(
                    text=metadata["story"],
                    api_key=api_key,
                    output_mp3_path=metadata["mp3_file_path"], # Assumes original path is still valid
                    voice_id=metadata.get("voice_id"),
                    model_id=metadata.get("model_id", "eleven_multilingual_v2"),
                    stability=metadata.get("stability", 0.5),
                    similarity_boost=metadata.get("similarity_boost", 0.75)
                )

                if success:
                    _logger.info(f"Queued audio generated successfully for {metadata.get('mp3_file_path')}. Removing queue file.")
                    processed_count += 1
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        _logger.error(f"Failed to remove queue file {file_path}: {e}")
                else:
                    _logger.warning(f"Queued audio generation attempt failed for {metadata.get('mp3_file_path')}. Will retry later.")
                    failed_count += 1

            except json.JSONDecodeError:
                 _logger.error(f"Error reading queued file {file_path}: Invalid JSON.")
                 # Optionally move/delete invalid file
            except Exception as e:
                _logger.exception(f"Unexpected error processing queue file {file_path}: {e}")

    if processed_count > 0 or failed_count > 0:
         _logger.info(f"Audio queue processing finished. Successful: {processed_count}, Failed attempts: {failed_count}")
