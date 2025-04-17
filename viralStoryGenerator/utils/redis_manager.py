# viralStoryGenerator/utils/redis_manager.py
"""Redis message broker for job processing and status tracking using Redis Streams."""

import json
import time
import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from typing import Dict, Any, Optional, List, Union
import uuid

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config

class RedisMessageBroker:
    """A Redis-based message broker using Redis Streams."""

    def __init__(self, redis_url: str, stream_name: str):
        """
        Initialize a Redis message broker using Redis Streams.

        Args:
            redis_url: Redis connection URL (redis://host:port/db)
            stream_name: Name of the Redis Stream to use
        """
        self.redis = redis.StrictRedis.from_url(redis_url)
        self.stream_name = stream_name

    def publish_message(self, message: Dict[str, Any]) -> str:
        """
        Publish a message to the Redis stream.

        Args:
            message: Dictionary containing message data

        Returns:
            str: ID of the published message
        """
        try:
            return self.redis.xadd(self.stream_name, message)
        except Exception as e:
            _logger.error(f"Failed to publish message to stream {self.stream_name}: {e}")
            return None

    def create_consumer_group(self, group_name: str) -> None:
        """
        Create a consumer group for the Redis stream.

        Args:
            group_name: Name of the consumer group
        """
        try:
            self.redis.xgroup_create(self.stream_name, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                pass  # Group already exists
            else:
                raise

    def consume_messages(self, group_name: str, consumer_name: str, count: int = 1, block: int = 5000) -> List[Dict[str, Any]]:
        """
        Consume messages from the Redis stream.

        Args:
            group_name: Name of the consumer group
            consumer_name: Name of the consumer
            count: Maximum number of messages to retrieve
            block: Time to block waiting for messages in milliseconds

        Returns:
            List of messages
        """
        messages = self.redis.xreadgroup(group_name, consumer_name, {self.stream_name: '>'}, count=count, block=block)
        return messages

    def acknowledge_message(self, group_name: str, message_id: str) -> None:
        """
        Acknowledge a message in the Redis stream.

        Args:
            group_name: Name of the consumer group
            message_id: ID of the message to acknowledge
        """
        self.redis.xack(self.stream_name, group_name, message_id)

    def pending_messages(self, group_name: str) -> List[Dict[str, Any]]:
        """
        Get pending messages for a consumer group.

        Args:
            group_name: Name of the consumer group

        Returns:
            List of pending messages
        """
        return self.redis.xpending(self.stream_name, group_name)

    def track_job_progress(self, job_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Track the progress of a job by updating its status in the stream.

        Args:
            job_id: ID of the job
            status: Status of the job (e.g., "pending", "processing", "completed", "failed")
            details: Additional details about the job status
        """
        try:
            message = {"job_id": job_id, "status": status, "timestamp": str(time.time())}

            if details:
                for key, value in details.items():
                    try:
                        if value is None:
                            message[key] = ""
                        elif isinstance(value, (dict, list)):
                            message[key] = json.dumps(value)
                        else:
                            message[key] = str(value)
                    except Exception as e:
                        _logger.warning(f"Could not serialize job detail {key}: {e}")
                        message[key] = str(value) if value is not None else ""

            result = self.publish_message(message)
            if not result:
                _logger.error(f"Failed to publish status update for job {job_id}")
        except Exception as e:
            _logger.error(f"Error tracking job progress for {job_id}: {e}")
            try:
                minimal_message = {
                    "job_id": str(job_id),
                    "status": str(status),
                    "timestamp": str(time.time()),
                    "error": "Failed to publish complete status update"
                }
                self.publish_message(minimal_message)
            except Exception as inner_e:
                _logger.error(f"Critical failure publishing minimal status for job {job_id}: {inner_e}")

    def get_job_status(self, job_id: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest status of a job from the stream.

        Args:
            job_id: ID of the job
            limit: Maximum number of messages to check

        Returns:
            Dictionary containing job status or None if not found
        """
        try:
            messages = self.redis.xrevrange(self.stream_name, count=limit)
            for message_id, message_data in messages:
                msg_job_id = message_data.get(b"job_id") or message_data.get("job_id")
                if isinstance(msg_job_id, bytes):
                    msg_job_id = msg_job_id.decode()

                if msg_job_id == job_id:
                    # Convert bytes to strings in the result
                    result = {}
                    for key, value in message_data.items():
                        key_str = key.decode() if isinstance(key, bytes) else key
                        value_str = value.decode() if isinstance(value, bytes) else value

                        # Try to parse JSON fields
                        if isinstance(value_str, str) and (value_str.startswith('{') or value_str.startswith('[')):
                            try:
                                result[key_str] = json.loads(value_str)
                            except:
                                result[key_str] = value_str
                        else:
                            result[key_str] = value_str

                    # Add the message ID
                    result["message_id"] = message_id.decode() if isinstance(message_id, bytes) else message_id
                    return result
        except Exception as e:
            _logger.error(f"Error retrieving job status for {job_id}: {e}")
        return None

    def get_job_history(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve the full history of a job from the stream.

        Args:
            job_id: ID of the job
            limit: Maximum number of messages to check

        Returns:
            List of job status updates in chronological order
        """
        try:
            all_messages = self.redis.xrange(self.stream_name, count=limit)
            job_messages = []

            for message_id, message_data in all_messages:
                msg_job_id = message_data.get(b"job_id") or message_data.get("job_id")
                if isinstance(msg_job_id, bytes):
                    msg_job_id = msg_job_id.decode()

                if msg_job_id == job_id:
                    # Convert bytes to strings in the result
                    result = {}
                    for key, value in message_data.items():
                        key_str = key.decode() if isinstance(key, bytes) else key
                        value_str = value.decode() if isinstance(value, bytes) else value

                        # Try to parse JSON fields
                        if isinstance(value_str, str) and (value_str.startswith('{') or value_str.startswith('[')):
                            try:
                                result[key_str] = json.loads(value_str)
                            except:
                                result[key_str] = value_str
                        else:
                            result[key_str] = value_str

                    # Add the message ID and timestamp
                    result["message_id"] = message_id.decode() if isinstance(message_id, bytes) else message_id
                    result["timestamp"] = float(result.get("timestamp", 0))
                    job_messages.append(result)

            # Sort by timestamp
            job_messages.sort(key=lambda x: x.get("timestamp", 0))
            return job_messages

        except Exception as e:
            _logger.error(f"Error retrieving job history for {job_id}: {e}")
        return []

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the stream.

        Returns:
            Dictionary containing stream status information
        """
        try:
            # Get stream information
            stream_info = self.redis.xinfo_stream(self.stream_name)

            # Convert bytes to strings
            info = {k.decode() if isinstance(k, bytes) else k:
                   v.decode() if isinstance(v, bytes) else v
                   for k, v in stream_info.items()}

            # Get recent messages for analysis
            messages = self.redis.xrevrange(self.stream_name, count=100)
            recent_jobs = set()
            job_statuses = {}

            for _, message_data in messages:
                job_id = message_data.get(b"job_id")
                status = message_data.get(b"status")

                if job_id and status:
                    job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                    status = status.decode() if isinstance(status, bytes) else status

                    recent_jobs.add(job_id)
                    job_statuses[job_id] = status

            # Count job statuses
            status_counts = {}
            for status in job_statuses.values():
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "stream_name": self.stream_name,
                "length": info.get("length", 0),
                "groups": info.get("groups", 0),
                "last_generated_id": info.get("last-generated-id", "0-0"),
                "recent_jobs_count": len(recent_jobs),
                "status_counts": status_counts
            }

        except Exception as e:
            _logger.error(f"Error getting queue status: {e}")
            return {"status": "error", "error": str(e)}

    def clear_stalled_jobs(self, group_name: str, consumer_name: str = None, max_idle_time: int = 600000) -> Dict[str, int]:
        """
        Clear stalled jobs from the pending entries list.

        Args:
            group_name: Name of the consumer group
            consumer_name: Optional consumer name to filter by
            max_idle_time: Maximum idle time in milliseconds

        Returns:
            Dictionary with counts of cleared jobs
        """
        try:
            # Get pending entries information
            pending_info = self.redis.xpending_range(
                self.stream_name,
                group_name,
                min='-',
                max='+',
                count=100,
                consumername=consumer_name
            )

            claimed = 0
            failed = 0

            for entry in pending_info:
                message_id = entry.get('message_id')
                idle_time = entry.get('idle')

                if idle_time > max_idle_time:
                    try:
                        # Claim the message and mark it as processed
                        self.redis.xclaim(
                            self.stream_name,
                            group_name,
                            "cleanup-worker",
                            min_idle_time=idle_time,
                            message_ids=[message_id]
                        )

                        # Acknowledge the message to remove from PEL
                        self.redis.xack(self.stream_name, group_name, message_id)
                        claimed += 1

                        # Get the message data and publish a failure status
                        messages = self.redis.xrange(self.stream_name, start=message_id, end=message_id)
                        if messages:
                            _, message_data = messages[0]
                            job_id = message_data.get(b"job_id")

                            if job_id:
                                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                                self.track_job_progress(
                                    job_id,
                                    "failed",
                                    {"error": f"Job stalled and cleared after {max_idle_time/1000} seconds"}
                                )
                                failed += 1

                    except Exception as e:
                        _logger.error(f"Error handling stalled message {message_id}: {e}")

            return {"claimed": claimed, "failed": failed}

        except Exception as e:
            _logger.error(f"Error clearing stalled jobs: {e}")
            return {"claimed": 0, "failed": 0, "error": str(e)}

    def purge_stream(self) -> bool:
        """
        Purge the entire stream. This is a destructive operation.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete the stream
            self.redis.delete(self.stream_name)

            # Recreate an empty stream
            self.redis.xadd(self.stream_name, {"purged": "true"})

            return True
        except Exception as e:
            _logger.error(f"Error purging stream {self.stream_name}: {e}")
            return False

    def ensure_stream_exists(self, stream_name: str) -> bool:
        """
        Ensure the Redis stream exists and is of the correct type.

        Args:
            stream_name: Name of the stream to ensure

        Returns:
            bool: True if stream exists or was created, False on error
        """
        try:
            stream_type = self.redis.type(stream_name)

            if stream_type != b'stream' and stream_type != 'stream':
                self.redis.delete(stream_name)  # Delete the key if it exists with the wrong type
                self.redis.xadd(stream_name, {"initialized": "true"})  # Initialize the stream
                _logger.info(f"Created new stream: {stream_name}")
            elif self.redis.xlen(stream_name) == 0:
                # Only add initialization message if the stream is completely empty (no messages)
                # This should happen very rarely
                self.redis.xadd(stream_name, {"initialized": "true"})
                _logger.info(f"Added initialization message to empty stream: {stream_name}")

            # Check if a consumer group exists before creating one
            try:
                # Try to get info about consumer groups - will fail if none exist
                self.redis.xinfo_groups(stream_name)
            except redis.exceptions.ResponseError:
                # No consumer groups exist, create default one
                try:
                    self.redis.xgroup_create(stream_name, "default-group", id="0", mkstream=True)
                    _logger.info(f"Created default consumer group for stream: {stream_name}")
                except redis.exceptions.ResponseError as e:
                    if "BUSYGROUP" not in str(e):
                        _logger.warning(f"Could not create consumer group: {e}")

            return True
        except Exception as e:
            _logger.error(f"Failed to ensure stream {stream_name} exists: {e}")
            return False
