# viralStoryGenerator/utils/redis_manager.py
"""Redis manager for task queuing and state management."""

import json
import time
import redis # Using redis-py
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from typing import Dict, Any, Optional, List, Union
import uuid

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config

class RedisManager:
    """Manages Redis connection, task queuing, and result storage."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, db: Optional[int] = None,
                 password: Optional[str] = None, queue_name: Optional[str] = None,
                 result_prefix: Optional[str] = None, ttl: Optional[int] = None):
        """Initializes Redis connection using config or provided parameters."""
        self.is_redis_enabled = app_config.redis.ENABLED
        if not self.is_redis_enabled:
            _logger.warning("Redis is disabled in config. RedisManager operations will be no-ops.")
            self.client = None
            return

        # Use provided args or fall back to config
        self.host = host or app_config.redis.HOST
        self.port = port or app_config.redis.PORT
        self.db = db if db is not None else app_config.redis.DB # Allow db=0
        self.password = password or app_config.redis.PASSWORD # Handles None password correctly
        self.queue_name = queue_name or app_config.redis.QUEUE_NAME
        self.result_prefix = result_prefix or app_config.redis.RESULT_PREFIX
        self.ttl = ttl if ttl is not None else app_config.redis.TTL # Time-to-live for result keys

        self.processing_queue_name = f"{self.queue_name}_processing" # Standard name for processing items

        self.client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        """Establishes connection to Redis."""
        if not self.is_redis_enabled: return
        try:
            # TODO: Consider using ConnectionPool for better performance if used heavily/concurrently
            self.client = redis.Redis(
                host=self.host, port=self.port, db=self.db, password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.client.ping() # Verify connection
            _logger.info(f"Connected to Redis at {self.host}:{self.port}/{self.db}")
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
        except Exception as e:
             _logger.exception(f"Unexpected error connecting to Redis: {e}")
             self.client = None

    def is_available(self) -> bool:
        """Checks if Redis connection is active."""
        if not self.client: return False
        try:
            return self.client.ping()
        except (ConnectionError, TimeoutError):
            _logger.warning("Redis connection check failed (ping).")
            return False
        except Exception as e:
             _logger.exception(f"Unexpected error during Redis ping: {e}")
             return False

    def _execute_command(self, command, *args, **kwargs) -> Any:
        """Helper to execute Redis commands with error handling."""
        if not self.is_available():
            _logger.error("Cannot execute Redis command: Redis is unavailable.")
            return None

        try:
            return command(*args, **kwargs)
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis command failed (connection/timeout): {e}")
            self.client = None
            return None
        except RedisError as e:
            _logger.error(f"Redis command failed (RedisError): {e}")
            return None
        except Exception as e:
             _logger.exception(f"Unexpected error executing Redis command: {e}")
             return None

    def add_request(self, request_data: Dict[str, Any]) -> bool:
        """Adds a job request to the main queue."""
        if not self.client: return False
        job_id = request_data.get("id")
        if not job_id:
            _logger.warning("Request data missing 'id'. Assigning a UUID.")
            job_id = str(uuid.uuid4())
            request_data["id"] = job_id

        try:
            # Serialize data to JSON string
            request_json = json.dumps(request_data)
        except TypeError as e:
            _logger.error(f"Failed to serialize request data to JSON: {e}. Data: {request_data}")
            return False

        # TODO: Use pipeline for atomic operations if needed, though LPUSH is atomic itself
        success = self._execute_command(self.client.lpush, self.queue_name, request_json)

        if success is not None: # LPUSH returns queue length on success
             _logger.debug(f"Job {job_id} added to queue '{self.queue_name}'.")
             self.store_result(job_id, {"status": "queued", "created_at": time.time()})
             return True
        else:
             _logger.error(f"Failed to add job {job_id} to queue '{self.queue_name}'.")
             return False

    def get_next_request(self) -> Optional[Dict[str, Any]]:
        """Atomically gets the next request from main queue and moves it to processing queue."""
        if not self.client: return None

        # BRPOPLPUSH blocks until item available or timeout (timeout=1 second)
        item_json = self._execute_command(
            self.client.brpoplpush, self.queue_name, self.processing_queue_name, timeout=1
        )

        if not item_json:
            return None

        try:
            request = json.loads(item_json)
            request['_processing_start_time'] = time.time()
            request['_original_data'] = item_json
            _logger.debug(f"Moved job {request.get('id')} to processing queue '{self.processing_queue_name}'.")
            return request
        except json.JSONDecodeError as e:
            _logger.error(f"Failed to decode JSON from queue item: {e}. Item: {item_json[:100]}...")
            removed = self._execute_command(self.client.lrem, self.processing_queue_name, 0, item_json)
            _logger.warning(f"Removed invalid JSON item from processing queue (removed: {removed}).")
            return None


    def complete_request(self, request: Dict[str, Any], success: bool):
        """Removes a completed/failed request from the processing queue."""
        if not self.client or not isinstance(request, dict): return False

        original_data = request.get('_original_data')
        job_id = request.get('id', 'unknown')
        if not original_data:
            _logger.error(f"Cannot complete job {job_id}: missing '_original_data' for removal.")
            return False

        # Remove the specific item from the processing queue
        removed_count = self._execute_command(self.client.lrem, self.processing_queue_name, 0, original_data)

        if removed_count is not None and removed_count > 0:
             _logger.debug(f"Completed job {job_id} (success={success}). Removed from processing queue.")
             return True
        elif removed_count == 0:
             _logger.warning(f"Job {job_id} (success={success}) not found in processing queue for completion (already removed?).")
             return False # Indicate item wasn't found/removed
        else:
             _logger.error(f"Failed to remove job {job_id} (success={success}) from processing queue.")
             return False


    def store_result(self, job_id: str, result_data: Dict[str, Any], merge: bool = False, ttl: Optional[int] = None) -> bool:
        """Stores job result/status data in Redis with TTL."""
        if not self.client: return False

        result_key = f"{self.result_prefix}{job_id}"
        effective_ttl = ttl if ttl is not None else self.ttl

        current_data = {}
        if merge:
             existing_json = self._execute_command(self.client.get, result_key)
             if existing_json:
                  try: current_data = json.loads(existing_json)
                  except json.JSONDecodeError: _logger.warning(f"Could not decode existing data for merge on key {result_key}")

        # Update data, ensuring timestamps are present
        current_data.update(result_data)
        current_data.setdefault("created_at", time.time())
        current_data["updated_at"] = time.time()

        try:
            # Serialize final data, handling potential non-serializable types
            result_json = json.dumps(current_data, default=str)
        except TypeError as e:
            _logger.error(f"Failed to serialize result data for job {job_id} to JSON: {e}")
            return False

        # Set with expiration
        success = self._execute_command(self.client.setex, result_key, effective_ttl, result_json)

        if success:
             _logger.debug(f"Stored result/status for job {job_id} with TTL {effective_ttl}s.")
             return True
        else:
             _logger.error(f"Failed to store result/status for job {job_id}.")
             return False

    def update_task_status(self, task_id: str, status: str):
        """Updates only the status field for a task."""
        return self.store_result(task_id, {"status": status}, merge=True)

    def update_task_error(self, task_id: str, error_message: str):
        """Updates task status to 'failed' and adds error message."""
        return self.store_result(task_id, {"status": "failed", "error": error_message}, merge=True)

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the result/status data for a job."""
        if not self.client: return None
        result_key = f"{self.result_prefix}{job_id}"
        result_json = self._execute_command(self.client.get, result_key)
        if result_json:
            try:
                return json.loads(result_json)
            except json.JSONDecodeError as e:
                 _logger.error(f"Failed to decode result JSON for job {job_id}: {e}")
                 return {"status": "error", "error": "Failed to decode stored result data."}
        return None

    get_task_status = get_result

    def check_key_exists(self, job_id: str) -> bool:
         """Checks if the result key exists for a job."""
         if not self.client: return False
         result_key = f"{self.result_prefix}{job_id}"
         exists = self._execute_command(self.client.exists, result_key)
         return bool(exists)

    def get_queue_length(self, queue_name: Optional[str] = None) -> int:
        """Gets the length of the specified queue (defaults to main queue)."""
        if not self.client: return 0
        q_name = queue_name or self.queue_name
        length = self._execute_command(self.client.llen, q_name)
        return length if isinstance(length, int) else 0

    def get_processing_queue_length(self) -> int:
         """Gets the length of the processing queue."""
         return self.get_queue_length(self.processing_queue_name)


    def wait_for_result(self, job_id: str, timeout: int = 300, check_interval: float = 0.5) -> Optional[Dict[str, Any]]:
        """Waits for a job result with polling (use with caution, consider websockets/SSE)."""
        if not self.client: return None
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            result = self.get_result(job_id)
            if result and result.get("status") in ["completed", "failed"]:
                return result
            time.sleep(check_interval)

        _logger.warning(f"Timeout ({timeout}s) reached waiting for job {job_id}")
        return {"status": "timeout", "error": f"Timeout waiting for result after {timeout}s."}
