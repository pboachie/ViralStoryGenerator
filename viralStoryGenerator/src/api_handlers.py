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
from viralStoryGenerator.utils.redis_manager import RedisManager
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.utils.security import is_file_in_directory, validate_path_component, sanitize_for_filename


# Initialize Redis manager only if enabled in config
redis_manager = RedisManager() if appconfig.redis.ENABLED else None

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

# ---- Functions related to processing logic ----
# These functions (_save_story_output, _read_sources_from_folder, process_story_task)
# represent the actual work. If Redis is enabled, this logic should primarily
# reside in the dedicated worker process (e.g., api_worker.py), not here in the
# API request handler module.
# Keeping them here only makes sense if Redis is disabled and tasks run synchronously
# or in background threads within the API process (not recommended for scalability).

# def _save_story_output_with_storage_manager(
#     result: Dict[str, Any],
#     topic: str,
#     voice_id: Optional[str] = None
# ) -> Dict[str, Optional[str]]:
#     """
#     Saves story output (script, storyboard, audio) using the StorageManager.
#     Returns a dictionary of relative storage paths or None on failure.
#     """
#     _logger.info(f"Saving outputs for topic: '{topic}' using {appconfig.storage.PROVIDER} storage.")
#     saved_paths: Dict[str, Optional[str]] = {
#         "story": None,
#         "audio": None,
#         "storyboard": None
#     }
#     if not result:
#         _logger.error("Cannot save output: Result data is empty.")
#         return saved_paths

#     # Generate safe base filename
#     safe_topic_base = sanitize_for_filename(topic)
#     date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
#     base_filename = f"{safe_topic_base}_{date_str}"

#     # --- Save Story Script ---
#     story_text = result.get("story", "")
#     video_desc = result.get("video_description", "")
#     if story_text or video_desc: # Save even if one part is missing
#         story_content = f"### Story Script:\n{story_text}\n\n### Video Description:\n{video_desc}"
#         story_filename = f"{base_filename}_story.txt"
#         try:
#             # Use storage_manager.store_file
#             story_info = storage_manager.store_file(
#                 file_data=story_content.encode('utf-8'), # Store as bytes
#                 file_type="story",
#                 filename=story_filename,
#                 content_type="text/plain"
#             )
#             if "error" not in story_info:
#                 saved_paths["story"] = story_info.get("file_path") # Store the relative path/key
#                 _logger.info(f"Story script saved via storage manager. Path/Key: {saved_paths['story']}")
#             else:
#                 _logger.error(f"Failed to save story script via storage manager: {story_info.get('error')}")
#         except Exception as e:
#             _logger.exception(f"Error saving story script: {e}")

#     # --- Save Storyboard ---
#     storyboard_data = result.get("storyboard")
#     if storyboard_data and isinstance(storyboard_data, dict):
#         storyboard_filename = f"{base_filename}_storyboard.json"
#         try:
#             storyboard_content = json.dumps(storyboard_data, indent=2)
#             storyboard_info = storage_manager.store_file(
#                 file_data=storyboard_content.encode('utf-8'),
#                 file_type="storyboard",
#                 filename=storyboard_filename,
#                 content_type="application/json"
#             )
#             if "error" not in storyboard_info:
#                 saved_paths["storyboard"] = storyboard_info.get("file_path")
#                 _logger.info(f"Storyboard saved via storage manager. Path/Key: {saved_paths['storyboard']}")
#             else:
#                 _logger.error(f"Failed to save storyboard via storage manager: {storyboard_info.get('error')}")
#         except Exception as e:
#             _logger.exception(f"Error saving storyboard: {e}")


#     # --- Generate and Save Audio ---
#     # This assumes audio generation happens *after* script generation.
#     # Requires ElevenLabs API key.
#     if story_text.strip() and appconfig.elevenLabs.API_KEY:
#         audio_filename = f"{base_filename}_audio.mp3"
#         _logger.info(f"Attempting audio generation for topic '{topic}'...")
#         # Generate audio to a temporary file first
#         # This needs generate_elevenlabs_audio function available
#         from .elevenlabs_tts import generate_elevenlabs_audio # Local import if needed here

#         temp_audio_path = None
#         try:
#             # Create a secure temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_f:
#                 temp_audio_path = temp_f.name

#             success = generate_elevenlabs_audio(
#                 text=story_text,
#                 api_key=appconfig.elevenLabs.API_KEY,
#                 output_mp3_path=temp_audio_path, # Save to temp file
#                 voice_id=voice_id, # Pass voice_id through
#                 # model_id, stability, similarity_boost from config?
#             )

#             if success:
#                 _logger.info(f"Audio generated successfully to temp file: {temp_audio_path}")
#                 # Now store the temporary file using storage_manager
#                 with open(temp_audio_path, "rb") as audio_f:
#                     audio_info = storage_manager.store_file(
#                         file_data=audio_f,
#                         file_type="audio",
#                         filename=audio_filename,
#                         content_type="audio/mpeg"
#                     )
#                 if "error" not in audio_info:
#                     saved_paths["audio"] = audio_info.get("file_path")
#                     _logger.info(f"Audio saved via storage manager. Path/Key: {saved_paths['audio']}")
#                 else:
#                     _logger.error(f"Failed to store generated audio via storage manager: {audio_info.get('error')}")
#             else:
#                  _logger.warning(f"Audio generation failed for topic '{topic}'. No audio file saved.")
#                  # TODO: Implement robust queuing/retry for failed audio?

#         except Exception as e:
#             _logger.exception(f"Error during audio generation or storage: {e}")
#         finally:
#             # Clean up temporary audio file
#             if temp_audio_path and os.path.exists(temp_audio_path):
#                 try:
#                     os.remove(temp_audio_path)
#                     _logger.debug(f"Cleaned up temporary audio file: {temp_audio_path}")
#                 except OSError as e:
#                     _logger.error(f"Failed to remove temporary audio file {temp_audio_path}: {e}")
#     elif not appconfig.elevenLabs.API_KEY:
#          _logger.warning("Skipping audio generation: ElevenLabs API Key not configured.")

#     return saved_paths


# def _read_sources_from_folder(folder_path: str) -> Optional[str]:
#     """
#     Reads source files from a validated sub-folder within the allowed base path.
#     Returns concatenated content or None if validation fails or read error occurs.
#     """
#     if not folder_path: return "" # No folder provided, return empty

#     # Validate the provided folder path component
#     if not validate_path_component(folder_path):
#          _logger.error(f"Invalid sources folder path component provided: {folder_path}")
#          return None

#     # Construct full path relative to the configured secure base path
#     base_sources_path = os.path.abspath(appconfig.security.SOURCE_MATERIALS_PATH)
#     full_folder_path = os.path.abspath(os.path.join(base_sources_path, folder_path))

#     # Security check: Ensure the path is within the allowed directory
#     if not os.path.isdir(full_folder_path) or not is_file_in_directory(full_folder_path, base_sources_path):
#         _logger.error(f"Sources folder path is invalid or outside allowed directory: {full_folder_path}")
#         return None

#     _logger.info(f"Reading sources from validated path: {full_folder_path}")
#     combined_texts = []
#     try:
#         for filename in os.listdir(full_folder_path):
#             # Basic check for potentially hidden or problematic files (customize as needed)
#             if filename.startswith('.'): continue

#             file_path = os.path.join(full_folder_path, filename)
#             # Ensure we only read files (and avoid symlink traversal issues if needed)
#             if os.path.isfile(file_path) and not os.path.islink(file_path):
#                 # Security: Double-check the file is still within the directory
#                 if not is_file_in_directory(file_path, full_folder_path):
#                      _logger.warning(f"Skipping file outside directory (possible race condition?): {file_path}")
#                      continue
#                 try:
#                     # Specify encoding, handle potential errors
#                     with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
#                         text = f.read().strip()
#                         if text:
#                             combined_texts.append(text)
#                 except Exception as e:
#                     _logger.warning(f"Failed to read source file {file_path}: {e}")
#                     # Decide whether to continue or fail on read error

#         # Combine texts
#         return "\n\n".join(combined_texts) if combined_texts else ""

#     except OSError as e:
#          _logger.error(f"Error accessing sources folder {full_folder_path}: {e}")
#          return None


# ---- API Facing Functions ----

def create_story_task(topic: str, sources_folder: Optional[str] = None,
                     voice_id: Optional[str] = None) -> Dict[str, Any]:
    """
    API handler logic to create a new story generation task.
    Validates inputs and queues the task via Redis if enabled.
    If Redis is disabled, logs a warning (processing should be handled elsewhere).
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Basic task info for immediate response
    task_response = StoryTask(
        task_id=task_id,
        topic=topic
    ).to_dict()

    # If using Redis, queue the task for the worker process
    if redis_manager:
        if not redis_manager.is_available():
            _logger.error("Redis is enabled but currently unavailable. Cannot queue task.")
            raise ConnectionError("Redis service unavailable, cannot queue task.")

        task_data_payload = {
            "id": task_id,
            "data": {
                "task_id": task_id,
                "topic": topic,
                "sources_folder": sources_folder,
                "voice_id": voice_id,
                "request_time": time.time()
            }
        }
        # Queue using RedisManager's method
        success = redis_manager.add_request(task_data_payload)

        if success:
            _logger.info(f"Task {task_id} queued successfully via Redis for topic: '{topic}'.")
            task_response["status"] = "queued"
        else:
            _logger.error(f"Failed to queue task {task_id} via Redis.")
            raise RuntimeError("Failed to add task to Redis queue.")
    else:
        # Redis is disabled - Log warning. Processing needs separate handling.
        _logger.warning(f"Redis is disabled. Task {task_id} created but NOT queued. Synchronous/threaded processing is NOT recommended here.")
        # If sync/thread processing was intended here:
        # 1. Update status to 'processing' immediately.
        # 2. Start background thread:
        #    import threading
        #    thread = threading.Thread(target=process_story_task_logic, args=(task_id, topic, sources_folder, voice_id))
        #    thread.daemon = True
        #    thread.start()
        # For now, just return the 'pending' status.
        task_response["status"] = "pending_no_queue"


    return task_response


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status and potentially result of a task from Redis.
    Returns a dictionary conforming to JobStatusResponse or None if not found.
    """
    if not redis_manager:
        _logger.error("Cannot get task status: Redis is disabled.")
        return {"status": "error", "message": "Task status unavailable: Redis is disabled.", "task_id": task_id}

    if not redis_manager.is_available():
         _logger.error("Cannot get task status: Redis is unavailable.")
         return {"status": "error", "message": "Task status unavailable: Redis service connection failed.", "task_id": task_id}

    try:
        status_data = redis_manager.get_task_status(task_id)

        if status_data:
             response = JobStatusResponse(
                 status=status_data.get("status", "unknown"),
                 message=status_data.get("message"),
                 story_script=status_data.get("story_script"),
                 storyboard=status_data.get("storyboard"),
                 audio_url=status_data.get("audio_url"),
                 sources=status_data.get("sources"),
                 error=status_data.get("error"),
                 created_at=status_data.get("created_at"),
                 updated_at=status_data.get("updated_at")
             ).model_dump(exclude_none=True)
             return response
        else:
             # Task ID not found in Redis
             _logger.debug(f"Task ID {task_id} not found in Redis.")
             return None

    except Exception as e:
        _logger.exception(f"Error retrieving task status for {task_id} from Redis: {e}")
        return {"status": "error", "message": "Failed to retrieve task status.", "task_id": task_id}


# NOTE: process_story_task (the actual work) should be moved entirely to the worker (api_worker.py)
# if Redis is enabled. It should not be called directly by the API handler in that case.
# Leaving a placeholder signature here if needed for non-Redis mode (discouraged).
# def process_story_task_logic(task_id: str, topic: str, sources_folder: Optional[str] = None,
#                       voice_id: Optional[str] = None):
#      """Placeholder for the actual task processing logic - SHOULD RESIDE IN WORKER."""
#      if redis_manager and redis_manager.is_available():
#           _logger.error("process_story_task_logic should not be called when Redis is enabled!")
#           return # Or raise error

#      _logger.warning(f"--- Processing task {task_id} synchronously/threaded (Not Recommended) ---")
#      # ... [Implement the full logic from the original process_story_task here] ...
#      # 1. Update status to processing (if using a direct state mechanism)
#      # 2. Call _read_sources_from_folder
#      # 3. Call chunkify_and_summarize
#      # 4. Call generate_story_script (LLM)
#      # 5. Call _save_story_output_with_storage_manager
#      # 6. Update status to completed/failed
#      # This requires careful state management if not using Redis.
#      _logger.error("Synchronous/threaded task processing logic is not fully implemented here.")
#      pass

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

                # Check attempt count if implemented
                # max_attempts = 3
                # attempts = metadata.get("attempts", 1)
                # if attempts > max_attempts:
                #     _logger.warning(f"Skipping {filename}: maximum retry attempts ({max_attempts}) exceeded.")
                #     # Optionally move to a failed directory
                #     continue

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
                    # Increment attempt count if tracking
                    # metadata["attempts"] = attempts + 1
                    # try:
                    #     with open(file_path, "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
                    # except Exception as e: _logger.error(f"Failed to update attempts in {file_path}: {e}")


            except json.JSONDecodeError:
                 _logger.error(f"Error reading queued file {file_path}: Invalid JSON.")
                 # Optionally move/delete invalid file
            except Exception as e:
                _logger.exception(f"Unexpected error processing queue file {file_path}: {e}")

    if processed_count > 0 or failed_count > 0:
         _logger.info(f"Audio queue processing finished. Successful: {processed_count}, Failed attempts: {failed_count}")
