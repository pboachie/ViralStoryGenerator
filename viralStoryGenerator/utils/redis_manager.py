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

    def __init__(self):
        """Initialize Redis connection"""
        if not app_config.redis.ENABLED:
            _logger.warning("Redis is disabled in config. Task queuing will not work properly.")
            self.client = None
            return

        try:
            self.client = redis.Redis(
                host=app_config.redis.HOST,
                port=app_config.redis.PORT,
                db=app_config.redis.DB,
                password=app_config.redis.PASSWORD,
                decode_responses=True
            )
            self.client.ping()  # Test connection
            _logger.info(f"Connected to Redis at {app_config.redis.HOST}:{app_config.redis.PORT}")

            self.queue_name = app_config.redis.QUEUE_NAME
            self.result_prefix = app_config.redis.RESULT_PREFIX
            self.ttl = app_config.redis.TTL

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