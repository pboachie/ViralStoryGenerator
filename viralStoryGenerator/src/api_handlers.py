# viralStoryGenerator/src/api_handlers.py
import tempfile
import time
import os
import datetime
import json
import uuid
import re
from typing import Dict, Any, Optional, List

from viralStoryGenerator.models import (
    StoryGenerationResult,
    JobStatusResponse
)
import logging
from viralStoryGenerator.utils.config import config as appconfig
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.utils.security import is_file_in_directory, validate_path_component, sanitize_for_filename

import viralStoryGenerator.src.logger
import asyncio
import inspect
_logger = logging.getLogger(__name__)

async def get_message_broker() -> RedisMessageBroker:
    """Get Redis message broker for API handlers, ensuring async initialization."""
    redis_url = "redis://" + appconfig.redis.HOST + ":" + str(appconfig.redis.PORT)
    broker = RedisMessageBroker(redis_url=redis_url, stream_name="api_jobs")
    await broker.initialize()
    return broker

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
    if not appconfig.storyboard.ENABLE_STORYBOARD_GENERATION:
        _logger.info("Storyboard generation is disabled in the configuration.")
        include_storyboard = False
    else:
        include_storyboard = True

    if not appconfig.ENABLE_IMAGE_GENERATION:
        _logger.info("Image generation is disabled. Skipping related tasks.")
        include_storyboard = False

    if not appconfig.ENABLE_AUDIO_GENERATION:
        _logger.info("Audio generation is disabled. Skipping related tasks.")

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    task_response = StoryTask(
        task_id=task_id,
        topic=topic
    ).to_dict()

    task_data_payload = {
        "job_id": task_id,
        "job_type": "generate_story",
        "topic": topic,
        "sources_folder": sources_folder,
        "voice_id": voice_id,
        "include_storyboard": include_storyboard,
        "request_time": time.time()
    }

    try:
        message_broker = get_message_broker()
        if inspect.isawaitable(message_broker):
            message_broker = asyncio.get_event_loop().run_until_complete(message_broker)
        message_id = message_broker.publish_message(task_data_payload)
        if inspect.isawaitable(message_id):
            message_id = asyncio.get_event_loop().run_until_complete(message_id)
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

async def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a task. Checks Redis Streams first, then final storage
    if the task is marked completed or not found in Redis.
    Returns a dictionary conforming to JobStatusResponse or None if not found.
    """
    status_source = "unknown"
    try:
        message_broker = await get_message_broker()
        stream_status = await message_broker.get_job_progress(task_id)
        final_result_data = {}

        if stream_status:
            status_source = "redis"
            _logger.debug(f"Stream status for task {task_id}: {json.dumps(stream_status, indent=2)}")
            current_status = stream_status.get("status", "pending")

            if (current_status == "completed"):
                metadata_filename = f"{task_id}_metadata.json"
                try:
                    final_result_data = storage_manager.retrieve_file_content_as_json(filename=metadata_filename, file_type="metadata") or {}
                    if not final_result_data:
                         _logger.warning(f"Task {task_id} status is 'completed' in Redis, but final metadata not found/empty in storage (checked {metadata_filename}).")
                    else:
                         _logger.info(f"Fetched final metadata for completed task {task_id} from storage.")
                except FileNotFoundError:
                    _logger.warning(f"Task {task_id} status is 'completed' in Redis, but final metadata file ({metadata_filename}) not found in storage.")
                    final_result_data = {"error": "Completed task metadata file not found in storage."}
                except Exception as storage_err:
                    _logger.error(f"Error fetching final metadata for completed task {task_id} from storage: {storage_err}")
                    final_result_data = {"error": f"Failed to fetch final results from storage: {storage_err}"}

            status_data = {
                "status": current_status,
                "job_id": stream_status.get("job_id", task_id),
                "message": stream_status.get("message", f"Job status from Redis Stream ({current_status})"),
                "created_at": stream_status.get("created_at") or stream_status.get("timestamp"),
                "updated_at": stream_status.get("updated_at") or stream_status.get("timestamp"),
                "error": stream_status.get("error"),
                "story_script": stream_status.get("story_script") or stream_status.get("result"),
                "storyboard": stream_status.get("storyboard"),
                "audio_url": stream_status.get("audio_url"),
                "sources": stream_status.get("sources"),
            }

            # Merge/Overwrite with final data if available
            if final_result_data:
                status_data["story_script"] = final_result_data.get("story_script", status_data["story_script"])
                status_data["storyboard"] = final_result_data.get("storyboard", status_data["storyboard"])
                status_data["audio_url"] = final_result_data.get("audio_url", status_data["audio_url"])
                status_data["sources"] = final_result_data.get("sources", status_data["sources"])
                storage_error = final_result_data.get("error")
                if storage_error:
                     # Combine Redis error (if any) and storage error
                     existing_error = status_data.get("error")
                     combined_error = f"{existing_error}; {storage_error}" if existing_error else storage_error
                     status_data["error"] = combined_error
                     status_data["message"] = f"Job status from Redis ({current_status}). Error fetching final results: {storage_error}"
                elif current_status == "completed":
                     status_data["message"] = final_result_data.get("message", "Job completed and final results retrieved from storage.")

                status_data["created_at"] = final_result_data.get("created_at", status_data["created_at"])
                status_data["updated_at"] = final_result_data.get("updated_at", status_data["updated_at"])


            response = JobStatusResponse(**status_data).model_dump(exclude_none=True)
            return response

        else:
            # Task ID not found in Redis Streams, check final storage directly
            status_source = "storage"
            _logger.debug(f"Task ID {task_id} not found in Redis Streams. Checking final storage.")
            metadata_filename = f"{task_id}_metadata.json"
            try:
                final_result_data = storage_manager.retrieve_file_content_as_json(filename=metadata_filename, file_type="metadata")
                if final_result_data:
                    _logger.info(f"Task {task_id} not found in Redis, but found completed results in storage (checked {metadata_filename}).")
                    status_data = {
                        "status": final_result_data.get("status", "completed"),
                        "job_id": task_id,
                        "message": final_result_data.get("message", "Job status retrieved from final storage (not found in Redis)."),
                        "story_script": final_result_data.get("story_script"),
                        "storyboard": final_result_data.get("storyboard"),
                        "audio_url": final_result_data.get("audio_url"),
                        "sources": final_result_data.get("sources"),
                        "error": final_result_data.get("error"),
                        "created_at": final_result_data.get("created_at"),
                        "updated_at": final_result_data.get("updated_at"),
                    }
                    response = JobStatusResponse(**status_data).model_dump(exclude_none=True)
                    return response
                else:
                    # Not found in Redis or storage
                     _logger.debug(f"Task ID {task_id} not found in Redis Streams or final storage (checked {metadata_filename}).")
                     return None
            except FileNotFoundError:
                 _logger.debug(f"Task ID {task_id} metadata file ({metadata_filename}) not found in storage (after Redis miss).")
                 return None
            except Exception as storage_err:
                 _logger.error(f"Error checking storage for task {task_id} after Redis miss: {storage_err}")
                 error_data_for_response = {
                     "status": "error",
                     "message": f"Error checking storage for job status: {str(storage_err)}",
                     "job_id": task_id,
                     "story_script": None,
                     "storyboard": None,
                     "audio_url": None,
                     "sources": None,
                     "error": f"Error checking storage for job status: {str(storage_err)}",
                     "created_at": None,
                     "updated_at": None
                 }
                 return JobStatusResponse(**error_data_for_response).model_dump(exclude_none=True)

    except Exception as e:
        _logger.exception(f"Error retrieving task status for {task_id} (source: {status_source}): {e}")
        error_response = {
            "status": "error",
            "message": f"Failed to retrieve task status: {str(e)}",
            "job_id": task_id,
            "story_script": None,
            "storyboard": None,
            "audio_url": None,
            "sources": None,
            "error": f"Failed to retrieve task status: {str(e)}",
            "created_at": None,
            "updated_at": None
            }
        try:
            return JobStatusResponse(**error_response).model_dump(exclude_none=True)
        except Exception:
            return error_response

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
    from .elevenlabs_tts import generate_elevenlabs_audio

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
                     # todo:  move/delete invalid file
                     continue

                # Attempt regeneration
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
            except Exception as e:
                _logger.exception(f"Unexpected error processing queue file {file_path}: {e}")

    if processed_count > 0 or failed_count > 0:
         _logger.info(f"Audio queue processing finished. Successful: {processed_count}, Failed attempts: {failed_count}")
