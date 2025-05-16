# viralStoryGenerator/utils/redis_manager.py
"""Redis message broker for job processing and status tracking using Redis Streams."""

import json
import time
import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from typing import Dict, Any, Optional, List, Union
import uuid

import logging
from viralStoryGenerator.utils.config import config as app_config

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

class RedisMessageBroker:
    """A Redis-based message broker using Redis Streams."""

    def __init__(self, redis_url: str, stream_name: str):
        """
        Initialize a Redis message broker using Redis Streams.

        Args:
            redis_url: Redis connection URL (redis://host:port/db)
            stream_name: Name of the Redis Stream to use
        """
        self.redis = redis.StrictRedis.from_url(redis_url, decode_responses=False)
        self.stream_name = stream_name
        self.job_status_prefix = getattr(app_config.redis, "JOB_STATUS_PREFIX", "job_status:")

    def _serialize_value(self, value: Any) -> bytes:
        """Serializes a value for storage in Redis. Returns bytes."""
        if isinstance(value, bytes):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value).encode('utf-8')
            except TypeError:
                _logger.warning(f"Could not JSON serialize value of type {type(value)}, falling back to string.")
                return str(value).encode('utf-8') # Fallback for non-serializable complex types
        elif value is None:
            return b''  # Represent None as an empty byte string
        return str(value).encode('utf-8')

    def _deserialize_value(self, value_bytes: Optional[bytes]) -> Any:
        """Deserializes a value retrieved from Redis (bytes)."""
        if value_bytes is None:
            return None
        if not value_bytes:  # Assuming empty byte string was used for None
            return None

        value_str = value_bytes.decode('utf-8')
        try:
            # Attempt to parse as JSON if it looks like a dict or list
            if (value_str.startswith('{') and value_str.endswith('}')) or \
               (value_str.startswith('[') and value_str.endswith(']')):
                return json.loads(value_str)
        except json.JSONDecodeError:
            # Not JSON, or malformed JSON, return as string
            pass
        return value_str

    def _get_latest_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Internal helper to get the raw latest data for a job_id from its status stream."""
        stream_key = f"{self.job_status_prefix}{job_id}"
        try:
            # Fetch the latest message (newest first)
            # XREVRANGE returns a list of tuples: (message_id_bytes, fields_dict_bytes)
            messages = self.redis.xrevrange(stream_key, count=1)
            if messages:
                _, data_bytes_dict = messages[0] # data_bytes_dict is Dict[bytes, bytes]
                return {k.decode('utf-8'): self._deserialize_value(v) for k, v in data_bytes_dict.items()}
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error getting latest job data for {job_id} from stream {stream_key}: {e}")
        return None

    def publish_message(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Publish a message to the Redis stream. Serializes complex types and None correctly.
        """
        _logger.debug(f"RedisMessageBroker: Attempting to publish to stream '{self.stream_name}'. Initial message keys: {list(message.keys())}")
        # Values in the dict passed to xadd must be bytes, str, int, or float.
        # Our _serialize_value returns bytes.
        safe_message = {k.encode('utf-8'): self._serialize_value(v) for k, v in message.items()}

        _logger.debug(f"RedisMessageBroker: Prepared safe_message for stream '{self.stream_name}'. Keys: {[k.decode('utf-8') for k in safe_message.keys()]}")
        try:
            message_id_bytes = self.redis.xadd(self.stream_name, safe_message)
            if message_id_bytes:
                return message_id_bytes.decode('utf-8')
            return None
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error publishing to stream {self.stream_name}: {e}")
            return None

    def create_consumer_group(self, group_name: str) -> None:
        """Create a consumer group for the stream if it doesn't exist."""
        try:
            self.redis.xgroup_create(name=self.stream_name, groupname=group_name, id='0', mkstream=True)
            _logger.info(f"Consumer group '{group_name}' created for stream '{self.stream_name}'.")
        except RedisError as e:
            if "BUSYGROUP" in str(e):
                _logger.info(f"Consumer group '{group_name}' already exists for stream '{self.stream_name}'.")
            else:
                _logger.error(f"Failed to create consumer group '{group_name}' for stream '{self.stream_name}': {e}")
                raise # Re-raise if it's an unexpected error

    def consume_messages(self, group_name: str, consumer_name: str, count: int = 1, block: int = 5000) -> List[Dict[str, Any]]:
        """Consume messages from the stream for a specific consumer group."""
        decoded_messages = []
        try:
            # XREADGROUP returns a list of streams, each stream is a list of messages
            # For one stream: [[b'stream_name', [(b'msg_id', {b'key': b'val'})]]]
            response = self.redis.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={self.stream_name: '>'}, # '>' means new messages only
                count=count,
                block=block # milliseconds
            )
            if response:
                for stream_name_bytes, messages_list in response:
                    for msg_id_bytes, data_bytes_dict in messages_list:
                        decoded_data = {k.decode('utf-8'): self._deserialize_value(v) for k, v in data_bytes_dict.items()}
                        decoded_messages.append({
                            "message_id": msg_id_bytes.decode('utf-8'),
                            "data": decoded_data
                        })
            # _logger.debug(f"Consumer {consumer_name} in group {group_name} received {len(decoded_messages)} messages.")
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error consuming messages for group {group_name}, consumer {consumer_name}: {e}")
        return decoded_messages

    def acknowledge_message(self, group_name: str, message_id: Union[str, List[str]]) -> int:
        """Acknowledge one or more messages as processed."""
        try:
            # Ensure message_ids are passed correctly
            ids_to_ack = [message_id] if isinstance(message_id, str) else message_id
            if not ids_to_ack:
                return 0
            ack_result = self.redis.xack(self.stream_name, group_name, *ids_to_ack)
            _logger.debug(f"Acknowledged {ack_result} message(s) {ids_to_ack} in group {group_name}.")
            return ack_result if isinstance(ack_result, int) else 0
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error acknowledging message(s) {message_id} in group {group_name}: {e}")
            return 0 # Indicate failure or no messages acknowledged

    def pending_messages(self, group_name: str) -> Dict[str, Any]: # Adjusted return type for more detail
        """Get information about pending messages for a consumer group."""
        try:
            # XPENDING without start/end/count returns summary:
            # [count, min_id, max_id, [[consumer_name, pending_count]]]
            pending_info = self.redis.xpending(self.stream_name, group_name)
            if isinstance(pending_info, dict): # Newer redis-py might return a dict directly
                 return pending_info
            # For older redis-py versions that return a tuple:
            if isinstance(pending_info, (list, tuple)) and len(pending_info) >= 4:
                consumers_detail = []
                if pending_info[3]: # Check if consumer details list is not empty
                    for consumer_data in pending_info[3]:
                        if isinstance(consumer_data, (list, tuple)) and len(consumer_data) == 2:
                             consumers_detail.append({"consumer_name": consumer_data[0].decode('utf-8'), "pending_count": consumer_data[1]})
                return {
                    "count": pending_info[0],
                    "min_id": pending_info[1].decode('utf-8') if pending_info[1] else None,
                    "max_id": pending_info[2].decode('utf-8') if pending_info[2] else None,
                    "consumers": consumers_detail
                }
            _logger.warning(f"Unexpected format from xpending for group {group_name}: {pending_info}")
            return {"count": 0, "min_id": None, "max_id": None, "consumers": []}

        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error getting pending messages for group {group_name}: {e}")
            return {"count": 0, "min_id": None, "max_id": None, "consumers": [], "error": str(e)}

    def track_job_progress(self, job_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Tracks the progress of a job by adding a new entry to its dedicated stream.
        The new entry carries forward previous details, updating them with new ones.
        """
        stream_key = f"{self.job_status_prefix}{job_id}"

        current_data = self._get_latest_job_data(job_id)

        new_progress_data: Dict[str, Any] = {}
        if current_data:
            new_progress_data.update(current_data)

        new_progress_data["job_id"] = job_id
        new_progress_data["status"] = status
        new_progress_data["timestamp"] = str(time.time())

        if details:
            # Carefully update, ensuring new details overwrite old ones for the same keys
            for key, value in details.items():
                new_progress_data[key] = value

        if "created_at" not in new_progress_data and new_progress_data.get("request_time"):
            new_progress_data["created_at"] = new_progress_data["request_time"]
        elif "created_at" not in new_progress_data: # If no request_time and no created_at, set it now
             new_progress_data["created_at"] = new_progress_data["timestamp"]


        new_progress_data.pop("message_id", None)

        safe_progress_data_to_add = {
            k.encode('utf-8'): self._serialize_value(v) for k, v in new_progress_data.items()
        }

        try:
            message_id_bytes = self.redis.xadd(stream_key, safe_progress_data_to_add)
            msg_id_str = message_id_bytes.decode('utf-8') if message_id_bytes else 'N/A'
            _logger.debug(f"Tracked progress for job {job_id} on stream {stream_key} (ID: {msg_id_str}): Status '{status}'. Keys: {list(new_progress_data.keys())}")

            maxlen = getattr(app_config.redis, "JOB_STATUS_MAXLEN", 100) # Keep last 100 entries
            if maxlen and isinstance(maxlen, int) and maxlen > 0:
               self.redis.xtrim(stream_key, maxlen=maxlen, approximate=True)

        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error tracking job progress for {job_id} on stream {stream_key}: {e}")

    def get_job_status(self, job_id: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest consolidated status for a job.
        """
        latest_status = self._get_latest_job_data(job_id)

        if latest_status:
            if "job_id" not in latest_status:
                latest_status["job_id"] = job_id
            _logger.debug(f"Retrieved latest consolidated status for job {job_id}: {list(latest_status.keys())}")
            return latest_status
        else:
            _logger.info(f"No status found in Redis stream for job_id: {job_id} using prefix {self.job_status_prefix}")
            return None

    def get_job_history(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves the historical sequence of statuses for a job."""
        stream_key = f"{self.job_status_prefix}{job_id}"
        history = []
        try:
            # For history, usually newest first is good.
            messages = self.redis.xrevrange(stream_key, count=limit)
            for msg_id_bytes, data_bytes_dict in messages:
                entry = {k.decode('utf-8'): self._deserialize_value(v) for k, v in data_bytes_dict.items()}
                entry['message_id'] = msg_id_bytes.decode('utf-8')
                history.append(entry)
            _logger.debug(f"Retrieved {len(history)} history entries for job {job_id}")
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Redis error getting job history for {job_id}: {e}")
        return history

    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed status of the main job queue stream."""
        try:
            stream_info = self.redis.xinfo_stream(self.stream_name)
            groups_info = self.redis.xinfo_groups(self.stream_name)

            decoded_stream_info = {
                k.decode('utf-8') if isinstance(k, bytes) else k:
                (v.decode('utf-8') if isinstance(v, bytes) else
                 (self._deserialize_value(v) if isinstance(v, bytes) else v)) # Handle first/last entry data
                for k, v in stream_info.items()
            }
            if 'first-entry' in decoded_stream_info and decoded_stream_info['first-entry']:
                decoded_stream_info['first-entry'] = (
                    decoded_stream_info['first-entry'][0].decode('utf-8'),
                    {k.decode('utf-8'): self._deserialize_value(v_bytes) for k, v_bytes in decoded_stream_info['first-entry'][1].items()}
                )
            if 'last-entry' in decoded_stream_info and decoded_stream_info['last-entry']:
                decoded_stream_info['last-entry'] = (
                    decoded_stream_info['last-entry'][0].decode('utf-8'),
                    {k.decode('utf-8'): self._deserialize_value(v_bytes) for k, v_bytes in decoded_stream_info['last-entry'][1].items()}
                )


            decoded_groups_info = []
            for group in groups_info:
                decoded_group = {
                    k.decode('utf-8') if isinstance(k, bytes) else k:
                    (v.decode('utf-8') if isinstance(v, bytes) else v)
                    for k, v in group.items()
                }
                decoded_groups_info.append(decoded_group)

            return {
                "stream_name": self.stream_name,
                "length": decoded_stream_info.get("length"),
                "radix_tree_keys": decoded_stream_info.get("radix-tree-keys"),
                "radix_tree_nodes": decoded_stream_info.get("radix-tree-nodes"),
                "last_generated_id": decoded_stream_info.get("last-generated-id"),
                "first_entry": decoded_stream_info.get("first-entry"),
                "last_entry": decoded_stream_info.get("last-entry"),
                "groups": decoded_stream_info.get("groups"),
                "groups_detail": decoded_groups_info
            }
        except RedisError as e:
            if "ERR no such key" in str(e).lower():
                 _logger.warning(f"Stream '{self.stream_name}' not found when getting queue status.")
                 return {"stream_name": self.stream_name, "error": "Stream not found", "length": 0, "groups_detail": []}
            _logger.error(f"Redis error getting queue status for stream '{self.stream_name}': {e}")
            return {"stream_name": self.stream_name, "error": str(e)}

    def clear_stalled_jobs(self, group_name: str, consumer_name: str, max_idle_time: int = 600000) -> Dict[str, int]:
        """Claim and process (or log) stalled jobs from a group's pending entries list (PEL)."""
        claimed_count = 0
        failed_to_claim_count = 0

        try:
            pending_entries = self.redis.xpending_range(
                name=self.stream_name,
                groupname=group_name,
                min_idle_time=max_idle_time,
                min='-', # Special ID: all pending messages
                max='+', # Special ID: all pending messages
                count=100 # Process in batches
                # consumername=consumer_name # This would filter, not what we want for general cleanup
            )

            if not pending_entries:
                _logger.info(f"No stalled jobs found for group '{group_name}' with idle time > {max_idle_time}ms.")
                return {"claimed": 0, "failed": 0}

            message_ids_to_claim = [entry['message_id'] for entry in pending_entries if isinstance(entry, dict) and 'message_id' in entry]

            if not message_ids_to_claim:
                 _logger.info(f"No message IDs extracted from pending entries for group '{group_name}'.")
                 return {"claimed": 0, "failed": 0}

            # 2. Claim these messages for the current worker/consumer
            # XCLAIM <key> <group> <consumer> <min-idle-time> <ID-1> ... <ID-N>
            claimed_messages = self.redis.xclaim(
                name=self.stream_name,
                groupname=group_name,
                consumername=consumer_name, # The consumer who will now own these messages
                min_idle_time=0, # min_idle_time for XCLAIM is different, 0 means claim regardless of current idle time
                message_ids=message_ids_to_claim,
                # justid=True # If you only want IDs and not the message bodies
            )

            claimed_count = len(claimed_messages)
            _logger.info(f"Claimed {claimed_count} stalled job(s) for consumer '{consumer_name}' in group '{group_name}'.")

        except RedisError as e:
            _logger.error(f"Error clearing stalled jobs for group '{group_name}': {e}")
            failed_to_claim_count = len(pending_entries if 'pending_entries' in locals() else []) # Best guess
            return {"claimed": claimed_count, "failed": failed_to_claim_count, "error": str(e)}

        return {"claimed": claimed_count, "failed": failed_to_claim_count}

    def purge_stream(self) -> bool:
        """Deletes the entire stream. Use with caution."""
        try:
            self.redis.delete(self.stream_name)
            # Also delete associated job status streams if a pattern is known
            # This is more complex and depends on how job_status_prefix is used.
            _logger.warning(f"Purged stream '{self.stream_name}'. All data lost.")
            return True
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Failed to purge stream '{self.stream_name}': {e}")
            return False

    def ensure_stream_exists(self, stream_name: str) -> bool:
        """Ensures a stream exists by trying to add a dummy message if it doesn't.
           More robustly, check type or use XINFO STREAM.
        """
        try:
            if not self.redis.exists(stream_name):
                self.redis.xadd(stream_name, {"_ensure": "1"}, maxlen=1, approximate=True)
                self.redis.xtrim(stream_name, maxlen=0, approximate=True) # Trim the dummy message
                _logger.info(f"Stream '{stream_name}' did not exist and was created.")
                return True
            # Check if it's actually a stream
            if self.redis.type(stream_name).decode('utf-8') != "stream":
                _logger.error(f"Key '{stream_name}' exists but is not a stream.")
                return False
            return True
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Error ensuring stream '{stream_name}' exists: {e}")
            return False
