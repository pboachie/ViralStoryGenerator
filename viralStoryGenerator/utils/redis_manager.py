# viralStoryGenerator/utils/redis_manager.py
import json
from datetime import date, datetime
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError, ResponseError, RedisError
from typing import Dict, Any, Optional, List, Union, cast, Tuple
import uuid
from pydantic import HttpUrl
import logging
import asyncio

_logger = logging.getLogger(__name__)


class RedisManager:
    """Manages async Redis connections and operations."""
    def __init__(self, redis_url: str):
        """Initialize RedisManager."""
        self.redis_url = redis_url
        self.pool: Optional[aioredis.ConnectionPool] = None
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        """Establish an async Redis connection."""
        try:
            self.pool = aioredis.ConnectionPool.from_url(self.redis_url, decode_responses=True)
            self.client = aioredis.Redis(connection_pool=self.pool)
            await self.client.ping()
            _logger.debug("Successfully connected to Redis asynchronously.")
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Async Redis connection failed: {e}")
            self.pool = None
            self.client = None
            raise
        except RedisError as e:
            _logger.error(f"Async Redis error: {e}")
            self.pool = None
            self.client = None
            raise

    def get_client(self) -> aioredis.Redis:
        """Return the Redis client instance. Ensure connect() was awaited."""
        if not self.client:
            _logger.error("Redis client not initialized. Call connect() first.")
            raise RuntimeError("Redis client not initialized. Call and await connect() first.")
        return self.client

    async def close(self):
        """Close the Redis connection pool."""
        if self.client:
            await self.client.close()
            _logger.debug("Async Redis client closed.")
        if self.pool:
            await self.pool.disconnect()
            _logger.debug("Async Redis connection pool disconnected.")
        self.client = None
        self.pool = None


class RedisMessageBroker:
    """A Redis-based message broker using Redis Streams with asyncio."""

    def __init__(self, redis_url: str, stream_name: str):
        """Initialize the message broker."""
        self.redis_manager = RedisManager(redis_url)
        self.client: Optional[aioredis.Redis] = None
        self.stream_name = stream_name
        self.consumer_group_name = f"{stream_name}-group"
        self.consumer_name = f"consumer-{uuid.uuid4()}"
        self._initialized = False

    async def initialize(self, ensure_group: bool = True):
        """Connects to Redis and optionally creates the consumer group. Must be called and awaited before use."""
        if self._initialized:
            _logger.debug("RedisMessageBroker already initialized.")
            return

        await self.redis_manager.connect()
        self.client = self.redis_manager.get_client()
        if not self.client:
             _logger.error("Failed to get Redis client after connect.")
             raise RuntimeError("Failed to get Redis client after connect.")

        if ensure_group:
            try:
                await self.client.xgroup_create(name=self.stream_name, groupname=self.consumer_group_name, id='0', mkstream=True)
                _logger.debug(f"Consumer group '{self.consumer_group_name}' ensured for stream '{self.stream_name}'.")
            except ResponseError as e:
                if "BUSYGROUP" in str(e):
                    _logger.debug(f"Consumer group '{self.consumer_group_name}' already exists for stream '{self.stream_name}'.")
                else:
                    _logger.error(f"Failed to create/verify consumer group: {e}")
                    await self.redis_manager.close()
                    raise
            except (ConnectionError, TimeoutError) as e:
                _logger.error(f"Redis connection error during initialization of consumer group: {e}")
                await self.redis_manager.close()
                raise
        else:
            _logger.debug(f"Skipping consumer group creation for stream '{self.stream_name}' as ensure_group is False.")

        self._initialized = True
        _logger.debug(f"RedisMessageBroker initialized successfully (group ensured: {ensure_group}).")


    def _ensure_initialized(self):
        if not self._initialized or not self.client:
            raise RuntimeError("RedisMessageBroker is not initialized. Call and await initialize() first.")

    @staticmethod
    def _make_json_serializable(item: Any) -> Any:
        if isinstance(item, list):
            return [RedisMessageBroker._make_json_serializable(i) for i in item]
        if isinstance(item, dict):
            return {k: RedisMessageBroker._make_json_serializable(v) for k, v in item.items()}
        if isinstance(item, HttpUrl):
            return str(item)
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if isinstance(item, uuid.UUID):
            return str(item)
        if hasattr(item, 'model_dump') and callable(getattr(item, 'model_dump')):
            try:
                dumped_model = item.model_dump()
                return RedisMessageBroker._make_json_serializable(dumped_model)
            except Exception as e:
                _logger.warning(f"Could not model_dump object of type {type(item)}: {e}")
                return str(item)
        if hasattr(item, 'dict') and callable(getattr(item, 'dict')):
            try:
                dumped_model = item.dict()
                return RedisMessageBroker._make_json_serializable(dumped_model)
            except Exception as e:
                _logger.warning(f"Could not .dict() object of type {type(item)}: {e}")
                return str(item)
        return item


    def _serialize_value(self, value: Any) -> str:
        """Serialize a Python object to a JSON string."""
        serializable_value = RedisMessageBroker._make_json_serializable(value)
        try:
            return json.dumps(serializable_value)
        except TypeError as e:
            _logger.error(f"Serialization error (json.dumps): {e} for value: {serializable_value}")
            raise

    def _deserialize_value(self, json_data: Optional[Union[str, bytes]]) -> Any:
        """Deserialize a JSON string (from Redis) to a Python object."""
        if json_data is None:
            return None
        try:
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8')
            return json.loads(json_data)
        except json.JSONDecodeError as e:
            _logger.error(f"Deserialization error (JSONDecodeError): {e} for json_string: {json_data}")
            return None
        except UnicodeDecodeError as e:
            _logger.error(f"Deserialization error (UnicodeDecodeError): {e} for json_data: {json_data}")
            return None


    async def publish_message(self, message_data: Dict[str, Any], job_id: Optional[str] = None) -> Optional[str]:
        """Publish a message to the Redis Stream."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)

        if not isinstance(message_data, dict):
            _logger.error("Message data must be a dictionary.")
            return None

        current_job_id = job_id or str(uuid.uuid4())
        payload_to_serialize = {"job_id": current_job_id, "data": message_data}

        try:
            serialized_payload_str: str = self._serialize_value(payload_to_serialize)
            message_id: Optional[str] = await client.xadd(self.stream_name, {"payload": serialized_payload_str})

            if message_id:
                _logger.debug(f"Message {message_id} published to stream '{self.stream_name}' with job_id: {current_job_id}")
            else:
                _logger.warning(f"Failed to publish message to stream '{self.stream_name}' (no message ID returned).")
            return message_id
        except TypeError as e:
            _logger.error(f"Serialization failed before publishing: {e}. Data: {payload_to_serialize}")
            return None
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis connection error during publish: {e}")
            return None
        except RedisError as e:
            _logger.error(f"Redis error during publish: {e}")
            return None

    async def consume_messages(self, count: int = 1, block_ms: Optional[int] = None) -> List[Tuple[str, Dict[str, str]]]:
        """Consume messages from the stream using the configured consumer group and name."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)

        try:
            messages_raw: Optional[List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]] = await client.xreadgroup(
                groupname=self.consumer_group_name,
                consumername=self.consumer_name,
                streams={self.stream_name: '>'},
                count=count,
                block=block_ms
            )

            processed_messages: List[Tuple[str, Dict[str, str]]] = []
            if messages_raw:
                for stream_name_str, message_list_tuples in messages_raw:
                    if stream_name_str == self.stream_name:
                        for message_id_str, message_data_dict_str_str in message_list_tuples:
                            processed_messages.append((message_id_str, message_data_dict_str_str))
            return processed_messages
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis connection error while consuming messages: {e}")
            raise
        except ResponseError as e:
            _logger.error(f"Redis response error while consuming messages: {e}")
            if "NOGROUP" in str(e):
                _logger.warning(f"Consumer group '{self.consumer_group_name}' not found for stream '{self.stream_name}'. Re-initialization might be needed.")
                self._initialized = False
            raise
        except Exception as e:
            _logger.error(f"Unexpected error while consuming messages: {e}", exc_info=True)
            raise

    async def acknowledge_message(self, message_id: str) -> int:
        """Acknowledge a message in the Redis Stream."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)
        try:
            ack_result: int = await client.xack(self.stream_name, self.consumer_group_name, message_id)
            if ack_result == 1:
                _logger.debug(f"Message {message_id} acknowledged in stream '{self.stream_name}'.")
            else:
                _logger.warning(f"Message {message_id} not acknowledged (result: {ack_result}). It might not exist in PEL for this consumer or was already acked.")
            return ack_result
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis connection error during acknowledge for {message_id}: {e}")
            return 0
        except RedisError as e:
            _logger.error(f"Redis error during acknowledge for message {message_id}: {e}")
            return 0

    async def track_job_progress(self, job_id: str, status: str, data: Optional[Dict[str, Any]] = None, ttl_seconds: int = 3600):
        """Track the progress of a job by storing its status and data in Redis with a TTL."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)

        if not job_id:
            _logger.error("job_id is required to track progress.")
            return

        progress_key = f"job_progress:{job_id}"
        serializable_data = RedisMessageBroker._make_json_serializable(data) if data else {}
        progress_info = {
            "status": status,
            "last_updated": datetime.utcnow().isoformat(),
            "data": serializable_data
        }

        try:
            serialized_progress_info_str: str = json.dumps(progress_info)
            await client.setex(progress_key, ttl_seconds, serialized_progress_info_str)
            _logger.debug(f"Job {job_id} progress updated: Status - {status}. Key: {progress_key}")
        except TypeError as e:
            _logger.error(f"Serialization failed for job progress {job_id} using json.dumps: {e}. Data attempted: {progress_info}")
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis connection error tracking job {job_id}: {e}")
        except RedisError as e:
            _logger.error(f"Redis error tracking job {job_id}: {e}")

    async def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the progress of a job from Redis."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)

        if not job_id:
            _logger.error("job_id is required to get progress.")
            return None

        progress_key = f"job_progress:{job_id}"
        serialized_progress_info_str: Optional[str] = None
        try:
            serialized_progress_info_str = await client.get(progress_key)

            if serialized_progress_info_str is not None:
                return self._deserialize_value(serialized_progress_info_str)
            else:
                _logger.debug(f"No progress information found for job {job_id} with key {progress_key}.")
                return None
        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis connection error getting job progress for {job_id}: {e}")
            return None
        except RedisError as e:
            _logger.error(f"Redis error getting job progress for {job_id}: {e}")
            return None
        except Exception as e:
            _logger.error(f"Unexpected error getting job progress for {job_id}: {e}. Data: '{serialized_progress_info_str}'")
            return None

    async def get_queue_information(self, count: int = 10) -> Dict[str, Any]:
        """Retrieve detailed information about the Redis Stream, its consumer groups, and recent messages."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)

        base_error_response = lambda msg: {
            "status": "error", "stream_length": 0, "consumer_groups": [],
            "recent_messages": [], "error_message": msg
        }

        try:
            stream_info = await client.xinfo_stream(self.stream_name)
            stream_length = stream_info.get('length', 0)

            groups_info_raw = await client.xinfo_groups(self.stream_name)
            consumer_groups_list = []
            for group_raw in groups_info_raw:
                group_name = group_raw.get('name')
                if not group_name: continue

                consumers_details_list = []
                try:
                    consumers_raw_list = await client.xinfo_consumers(self.stream_name, group_name)
                    for cons_raw in consumers_raw_list:
                        consumers_details_list.append({
                            "name": cons_raw.get('name'),
                            "pending": cons_raw.get('pending'),
                            "idle": cons_raw.get('idle')
                        })
                except Exception as e_cons:
                    _logger.error(f"Failed to get consumers for group {group_name}: {e_cons}")

                consumer_groups_list.append({
                    "group_name": group_name,
                    "pending": group_raw.get('pending'),
                    "consumers_count": group_raw.get('consumers'),
                    "last_delivered_id": group_raw.get('last-delivered-id'),
                    "consumer_details": consumers_details_list
                })

            recent_messages_raw: List[Tuple[str, Dict[str, str]]] = await client.xrevrange(self.stream_name, count=count)
            recent_messages_list = []
            for msg_id, msg_fields_raw_dict in recent_messages_raw:
                payload_str = msg_fields_raw_dict.get('payload')
                job_id_in_msg, status_in_msg, is_system = "N/A", "unknown", False

                if payload_str:
                    try:
                        deserialized_payload = self._deserialize_value(payload_str)
                        if isinstance(deserialized_payload, dict):
                            job_id_in_msg = deserialized_payload.get("job_id", "N/A")
                            if job_id_in_msg != "N/A":
                                job_progress = await self.get_job_progress(job_id_in_msg)
                                if job_progress and isinstance(job_progress, dict):
                                    status_in_msg = job_progress.get("status", "unknown")
                            if job_id_in_msg and job_id_in_msg.startswith("system_"):
                                is_system = True
                    except Exception as e_payload:
                        _logger.warning(f"Could not process payload for msg {msg_id} in get_queue_information: {e_payload}")

                timestamp_ms_str = msg_id.split('-')[0]
                try:
                    timestamp_dt = datetime.fromtimestamp(int(timestamp_ms_str) / 1000.0)
                    timestamp_iso = timestamp_dt.isoformat()
                except ValueError:
                    timestamp_iso = "N/A"

                recent_messages_list.append({
                    "id": msg_id, "timestamp": timestamp_iso, "job_id": job_id_in_msg,
                    "status": status_in_msg, "is_system_message": is_system,
                })

            return {
                "status": "available", "stream_length": stream_length,
                "consumer_groups": consumer_groups_list, "recent_messages": recent_messages_list
            }

        except (ConnectionError, TimeoutError) as e:
            _logger.error(f"Redis connection error in get_queue_information: {e}")
            return base_error_response(f"Redis connection error: {e}")
        except ResponseError as e:
             _logger.error(f"Redis command error in get_queue_information (e.g. stream NX): {e}")
             if "ERR no such key" in str(e) or (hasattr(e, 'args') and e.args and "no such key" in e.args[0].lower()):
                 _logger.debug(f"Stream {self.stream_name} does not exist yet.")
                 return {
                     "status": "stream_not_found", "stream_name": self.stream_name,
                     "stream_length": 0, "consumer_groups": [], "recent_messages": []
                 }
             return base_error_response(f"Redis command error: {e}")
        except RedisError as e:
            _logger.error(f"Generic Redis error in get_queue_information: {e}")
            return base_error_response(f"Redis error: {e}")
        except Exception as e:
            _logger.error(f"Unexpected error in get_queue_information: {e}", exc_info=True)
            return base_error_response(f"Unexpected error: {e}")

    async def close(self):
        """Close the Redis connection through RedisManager."""
        if self.redis_manager:
            await self.redis_manager.close()
        self._initialized = False
        _logger.debug("RedisMessageBroker connections closed.")

    async def purge_stream_messages(self) -> int:
        """Deletes all messages from the stream. Use with extreme caution."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)
        try:
            result = await client.xtrim(self.stream_name, maxlen=0)
            _logger.debug(f"Purged messages from stream '{self.stream_name}'. Result: {result}")
            return cast(int, result)
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Error purging stream '{self.stream_name}': {e}")
            return 0

    async def get_stream_length(self) -> int:
        """Gets the current length of the stream."""
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)
        try:
            length = await client.xlen(self.stream_name)
            return cast(int, length)
        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Error getting stream length for '{self.stream_name}': {e}")
            return 0

    async def get_pending_messages_info(self, group_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves information about pending messages for a consumer group.
        Format: [total_pending, min_message_id, max_message_id, [(consumer_name, num_pending_for_consumer), ...]]
        """
        self._ensure_initialized()
        client = cast(aioredis.Redis, self.client)
        target_group = group_name or self.consumer_group_name
        try:
            pending_summary = await client.xpending(name=self.stream_name, groupname=target_group)
            return cast(Dict[str, Any], pending_summary)

        except (ConnectionError, TimeoutError, RedisError) as e:
            _logger.error(f"Error getting pending messages info for group '{target_group}' on stream '{self.stream_name}': {e}")
            return None
