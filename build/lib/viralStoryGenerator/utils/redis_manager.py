#!/usr/bin/env python
# viralStoryGenerator/utils/redis_manager.py

import json
import time
import redis
from typing import Dict, Any, Optional, List
import uuid

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config

class RedisManager:
    """Redis manager for handling task queuing and state management"""

    def __init__(self, host=None, port=None, db=None, password=None, queue_name=None, result_prefix=None, ttl=None):
        """Initialize Redis connection"""
        # Default to values from config if parameters aren't provided
        self.host = host or app_config.redis.HOST
        self.port = port or app_config.redis.PORT
        self.db = db or app_config.redis.DB
        self.password = password or app_config.redis.PASSWORD
        self.queue_name = queue_name or app_config.redis.QUEUE_NAME
        self.result_prefix = result_prefix or app_config.redis.RESULT_PREFIX
        self.ttl = ttl or app_config.redis.TTL

        if not app_config.redis.ENABLED:
            _logger.warning("Redis is disabled in config. Task queuing will not work properly.")
            self.client = None
            return

        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self.client.ping()  # Test connection
            _logger.info(f"Connected to Redis at {self.host}:{self.port}")

        except redis.exceptions.ConnectionError as e:
            _logger.error(f"Failed to connect to Redis: {str(e)}")
            self.client = None

    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False

    def enqueue_task(self, task_data: Dict[str, Any]) -> bool:
        """
        Enqueue a task for processing

        Args:
            task_data: Dictionary with task data

        Returns:
            bool: Success status
        """
        if not self.client:
            return False

        try:
            task_id = task_data.get("task_id")
            if not task_id:
                task_id = str(uuid.uuid4())
                task_data["task_id"] = task_id

            # Store initial task status
            self.update_task_status(task_id, "pending")

            # Add to processing queue
            self.client.lpush(self.queue_name, json.dumps(task_data))
            _logger.debug(f"Task {task_id} added to queue")
            return True
        except Exception as e:
            _logger.error(f"Failed to enqueue task: {str(e)}")
            return False

    def dequeue_task(self) -> Optional[Dict[str, Any]]:
        """
        Dequeue a task for processing

        Returns:
            Dict or None: Task data if available
        """
        if not self.client:
            return None

        try:
            # Pop task with blocking wait for specified timeout
            result = self.client.brpop(self.queue_name, timeout=1)
            if not result:
                return None

            _, task_data_str = result
            task_data = json.loads(task_data_str)
            return task_data
        except Exception as e:
            _logger.error(f"Failed to dequeue task: {str(e)}")
            return None

    def update_task_status(self, task_id: str, status: str) -> bool:
        """
        Update the status of a task

        Args:
            task_id: Task identifier
            status: New status value

        Returns:
            bool: Success status
        """
        if not self.client:
            return False

        try:
            status_key = f"{self.result_prefix}{task_id}"
            # Get existing data if any
            existing_data = self.client.get(status_key)
            if existing_data:
                data = json.loads(existing_data)
                data["status"] = status
                data["updated_at"] = time.time()
            else:
                data = {
                    "task_id": task_id,
                    "status": status,
                    "created_at": time.time(),
                    "updated_at": time.time()
                }

            # Save back to Redis
            self.client.setex(status_key, self.ttl, json.dumps(data))
            return True
        except Exception as e:
            _logger.error(f"Failed to update task status: {str(e)}")
            return False

    def update_task_result(self, task_id: str, result_data: Dict[str, Any]) -> bool:
        """
        Update task with result data

        Args:
            task_id: Task identifier
            result_data: Result data to store

        Returns:
            bool: Success status
        """
        if not self.client:
            return False

        try:
            status_key = f"{self.result_prefix}{task_id}"
            # Ensure status is completed
            result_data["status"] = "completed"
            result_data["updated_at"] = time.time()

            # Save to Redis
            self.client.setex(status_key, self.ttl, json.dumps(result_data))
            return True
        except Exception as e:
            _logger.error(f"Failed to update task result: {str(e)}")
            return False

    def update_task_error(self, task_id: str, error_message: str) -> bool:
        """
        Update task with error information

        Args:
            task_id: Task identifier
            error_message: Error message

        Returns:
            bool: Success status
        """
        if not self.client:
            return False

        try:
            status_key = f"{self.result_prefix}{task_id}"
            # Get existing data if any
            existing_data = self.client.get(status_key)
            if existing_data:
                data = json.loads(existing_data)
                data["status"] = "failed"
                data["error"] = error_message
                data["updated_at"] = time.time()
            else:
                data = {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error_message,
                    "created_at": time.time(),
                    "updated_at": time.time()
                }

            # Save back to Redis
            self.client.setex(status_key, self.ttl, json.dumps(data))
            return True
        except Exception as e:
            _logger.error(f"Failed to update task error: {str(e)}")
            return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a task

        Args:
            task_id: Task identifier

        Returns:
            Dict or None: Task status information
        """
        if not self.client:
            return None

        try:
            status_key = f"{self.result_prefix}{task_id}"
            result = self.client.get(status_key)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            _logger.error(f"Failed to get task status: {str(e)}")
            return None

    def get_queue_length(self) -> int:
        """
        Get the current length of the task queue

        Returns:
            int: Number of tasks in queue
        """
        if not self.client:
            return 0

        try:
            return self.client.llen(self.queue_name)
        except Exception as e:
            _logger.error(f"Failed to get queue length: {str(e)}")
            return 0

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the result of a completed task.

        Args:
            task_id: Task identifier

        Returns:
            Dict or None: Task result information if available
        """
        if not self.client:
            return None

        try:
            status_key = f"{self.result_prefix}{task_id}"
            result = self.client.get(status_key)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            _logger.error(f"Failed to get task result: {str(e)}")
            return None

    def check_key_exists(self, job_id: str) -> bool:
        """
        Check if any key exists for the given job ID.
        Used to determine if a job exists but hasn't completed yet.

        Args:
            job_id: Job ID to check for existence

        Returns:
            bool: True if any key with this job ID exists, False otherwise
        """
        if not self.client:
            return False

        try:
            # Create the specific result key pattern to check
            key = f"{self.result_prefix}{job_id}"
            return bool(self.client.exists(key))
        except Exception as e:
            _logger.error(f"Error checking key existence: {str(e)}")
            return False

    def add_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Add a request to the queue

        Args:
            request_data: Dictionary with request data

        Returns:
            bool: Success status
        """
        if not self.client:
            return False

        try:
            # Get job ID
            job_id = request_data.get("id")
            if not job_id:
                job_id = str(uuid.uuid4())
                request_data["id"] = job_id

            # Add to queue
            self.client.lpush(self.queue_name, json.dumps(request_data))

            # Initialize status
            status_key = f"{self.result_prefix}{job_id}"
            self.client.setex(
                status_key,
                self.ttl,
                json.dumps({
                    "status": "pending",
                    "message": "Job queued, waiting to start processing",
                    "created_at": time.time()
                })
            )

            return True
        except Exception as e:
            _logger.error(f"Failed to add request: {str(e)}")
            return False

    def store_result(self, job_id: str, result_data: Dict[str, Any]) -> bool:
        """
        Store result data for a job

        Args:
            job_id: Job ID
            result_data: Result data to store

        Returns:
            bool: Success status
        """
        if not self.client:
            return False

        try:
            # Create result key
            status_key = f"{self.result_prefix}{job_id}"

            # Add timestamp
            if "created_at" not in result_data:
                result_data["created_at"] = time.time()
            result_data["updated_at"] = time.time()

            # Store in Redis
            self.client.setex(
                status_key,
                self.ttl,
                json.dumps(result_data)
            )

            return True
        except Exception as e:
            _logger.error(f"Failed to store result: {str(e)}")
            return False

    def wait_for_result(self, job_id: str, timeout: int = 300, check_interval: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Wait for a job result with timeout.

        Args:
            job_id: The job ID to wait for
            timeout: Maximum time to wait in seconds (default 300 seconds/5 minutes)
            check_interval: How often to check for results in seconds (default 0.5 seconds)

        Returns:
            Dict or None: Result data if job completed successfully within timeout, None otherwise
        """
        if not self.client:
            return None

        try:
            start_time = time.time()
            status_key = f"{self.result_prefix}{job_id}"

            while (time.time() - start_time) < timeout:
                # Check if result exists
                result_data = self.client.get(status_key)
                if result_data:
                    result = json.loads(result_data)
                    status = result.get("status")

                    # If job completed or failed, return the result
                    if status in ["completed", "failed"]:
                        return result

                # Wait before checking again
                time.sleep(check_interval)

            # Timeout reached
            _logger.warning(f"Timeout reached while waiting for job {job_id}")
            return {
                "status": "failed",
                "error": f"Timeout reached waiting for job result after {timeout} seconds",
                "job_id": job_id
            }
        except Exception as e:
            _logger.error(f"Error waiting for job result: {str(e)}")
            return None

    def get_next_request(self) -> Optional[Dict[str, Any]]:
        """
        Get the next request from the queue without removing it.
        Returns:
            Dict or None: The next request in the queue, or None if queue is empty
        """
        if not self.client:
            return None

        try:
            # Create a processing queue if it doesn't exist
            processing_queue = f"{self.queue_name}_processing"

            # Atomically move item from main queue to processing queue
            item = self.client.brpoplpush(self.queue_name, processing_queue, timeout=1)

            if not item:
                return None

            # Parse the request data
            request = json.loads(item)

            # Store the original data with the request to make cleanup possible
            if isinstance(request, dict):
                request['_original_data'] = item

            return request
        except Exception as e:
            _logger.error(f"Failed to get next request: {str(e)}")
            return None

    def complete_request(self, request: Dict[str, Any], success: bool = True) -> bool:
        """
        Mark a request as completed and remove it from the processing queue.

        Args:
            request: The request object returned by get_next_request
            success: Whether the processing was successful

        Returns:
            bool: Success status of the operation
        """
        if not self.client or not request:
            return False

        try:
            processing_queue = f"{self.queue_name}_processing"

            # Get the original data that was stored
            original_data = request.get('_original_data')
            if not original_data:
                _logger.warning("Cannot complete request: missing original data")
                return False

            # Remove the item from the processing queue
            self.client.lrem(processing_queue, 1, original_data)

            # If processing failed, optionally put it back in the main queue
            if not success:
                # Check retry count and re-queue if below threshold (Configurable)
                pass

            return True
        except Exception as e:
            _logger.error(f"Failed to complete request: {str(e)}")
            return False