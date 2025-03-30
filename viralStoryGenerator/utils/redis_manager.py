"""
Redis Queue Manager for handling crawl4ai requests.
This module provides a queuing system to prevent overloading the system
with too many simultaneous web scraping requests.
"""
import json
import time
import logging
import uuid
from typing import Dict, Any, Optional, List, Union
import redis
from redis.exceptions import ConnectionError, RedisError
from .config import config

logger = logging.getLogger(__name__)

class RedisQueueManager:
    """
    Redis-based queue manager for crawl4ai requests.
    Handles queueing and processing of web scraping requests to avoid
    system overload by controlling request flow.
    """

    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 db: Optional[int] = None,
                 password: Optional[str] = None,
                 queue_name: Optional[str] = None,
                 result_prefix: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: int = 2,
                 ttl: Optional[int] = None):  # Default 1-hour TTL for results
        """
        Initialize the Redis Queue Manager.

        Args:
            host: Redis server hostname (defaults to config value)
            port: Redis server port (defaults to config value)
            db: Redis database number (defaults to config value)
            password: Redis password if required (defaults to config value)
            queue_name: Name of the queue for crawl4ai requests (defaults to config value)
            result_prefix: Prefix for result keys in Redis (defaults to config value)
            max_retries: Maximum number of connection retries
            retry_delay: Delay between retries in seconds
            ttl: Time-to-live for results in seconds (defaults to config value)
        """
        # Use provided values or fall back to config
        self.queue_name = queue_name or config.redis.QUEUE_NAME
        self.result_prefix = result_prefix or config.redis.RESULT_PREFIX
        self.ttl = ttl or config.redis.TTL
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Try to establish Redis connection with retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.redis = redis.Redis(
                    host=host or config.redis.HOST,
                    port=port or config.redis.PORT,
                    db=db if db is not None else config.redis.DB,
                    password=password or config.redis.PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=10
                )
                # Test connection
                self.redis.ping()
                logger.info(f"Connected to Redis at {host or config.redis.HOST}:{port or config.redis.PORT}")
                break
            except ConnectionError as e:
                retry_count += 1
                logger.warning(f"Redis connection attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to Redis after maximum retries")
                    raise ConnectionError(f"Could not connect to Redis after {max_retries} attempts") from e
            except RedisError as e:
                logger.error(f"Redis error during initialization: {str(e)}")
                raise

    def add_request(self, request_data: Dict[str, Any]) -> str:
        """
        Add a crawl4ai request to the queue.

        Args:
            request_data: Dictionary containing request parameters for crawl4ai
                          (e.g., {'url': 'https://example.com', 'depth': 1})

        Returns:
            request_id: Unique ID for the queued request
        """
        request_id = str(uuid.uuid4())
        payload = {
            'id': request_id,
            'timestamp': time.time(),
            'status': 'queued',
            'data': request_data
        }

        try:
            # Add to the queue
            self.redis.rpush(self.queue_name, json.dumps(payload))
            logger.info(f"Request {request_id} added to queue")
            return request_id
        except RedisError as e:
            logger.error(f"Failed to add request to queue: {str(e)}")
            raise

    def get_next_request(self) -> Optional[Dict[str, Any]]:
        """
        Get the next request from the queue.
        Used by worker processes to retrieve pending requests.

        Returns:
            Request data or None if queue is empty
        """
        try:
            # Pop item from the left of the queue (FIFO)
            item = self.redis.lpop(self.queue_name)
            if item:
                return json.loads(item)
            return None
        except RedisError as e:
            logger.error(f"Failed to get next request from queue: {str(e)}")
            raise

    def store_result(self, request_id: str, result: Any) -> bool:
        """
        Store the result of a processed request.

        Args:
            request_id: Unique ID of the request
            result: Result data to store

        Returns:
            success: True if the result was stored successfully
        """
        result_key = f"{self.result_prefix}{request_id}"
        try:
            self.redis.setex(
                result_key,
                self.ttl,
                json.dumps({
                    'id': request_id,
                    'timestamp': time.time(),
                    'data': result
                })
            )
            logger.info(f"Result stored for request {request_id}")
            return True
        except RedisError as e:
            logger.error(f"Failed to store result for request {request_id}: {str(e)}")
            raise

    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a processed request.

        Args:
            request_id: Unique ID of the request

        Returns:
            result: Result data if available, None otherwise
        """
        result_key = f"{self.result_prefix}{request_id}"
        try:
            result = self.redis.get(result_key)
            if result:
                return json.loads(result)
            return None
        except RedisError as e:
            logger.error(f"Failed to get result for request {request_id}: {str(e)}")
            raise

    def get_queue_length(self) -> int:
        """
        Get the current length of the queue.

        Returns:
            length: Number of items in the queue
        """
        try:
            return self.redis.llen(self.queue_name)
        except RedisError as e:
            logger.error(f"Failed to get queue length: {str(e)}")
            raise

    def peek_queue(self, start: int = 0, end: int = -1) -> List[Dict[str, Any]]:
        """
        Peek at items in the queue without removing them.

        Args:
            start: Start index
            end: End index (-1 for all items)

        Returns:
            items: List of queue items
        """
        try:
            items = self.redis.lrange(self.queue_name, start, end)
            return [json.loads(item) for item in items]
        except RedisError as e:
            logger.error(f"Failed to peek queue: {str(e)}")
            raise

    def clear_queue(self) -> bool:
        """
        Clear the entire queue.

        Returns:
            success: True if queue was cleared successfully
        """
        try:
            self.redis.delete(self.queue_name)
            logger.info("Queue cleared")
            return True
        except RedisError as e:
            logger.error(f"Failed to clear queue: {str(e)}")
            raise

    def wait_for_result(self, request_id: str, timeout: int = 300, poll_interval: int = 1) -> Optional[Dict[str, Any]]:
        """
        Wait for a result to become available, with timeout.

        Args:
            request_id: Unique ID of the request
            timeout: Maximum time to wait in seconds
            poll_interval: Time between polling attempts in seconds

        Returns:
            result: Result data if available within timeout, None otherwise
        """
        end_time = time.time() + timeout
        result = None

        while time.time() < end_time:
            result = self.get_result(request_id)
            if result:
                return result
            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for result of request {request_id}")
        return None