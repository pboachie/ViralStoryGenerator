import pytest
import asyncio
import json
import uuid
from datetime import datetime
from pydantic import BaseModel, HttpUrl
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming the module is viralStoryGenerator.utils.redis_manager
from viralStoryGenerator.utils import redis_manager as rm_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from redis.asyncio.exceptions import ConnectionError, TimeoutError, ResponseError # For Redis exceptions

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_redis_defaults(monkeypatch):
    """Set default app_config values for redis_manager tests."""
    monkeypatch.setattr(app_config.redis, 'HOST', "mock_redis_mgr_host")
    monkeypatch.setattr(app_config.redis, 'PORT', 6380) # Different port for clarity
    monkeypatch.setattr(app_config.redis, 'PASSWORD', "mock_password")
    monkeypatch.setattr(app_config.redis, 'DB', 0)
    monkeypatch.setattr(app_config.redis, 'USE_SENTINEL', False)
    monkeypatch.setattr(app_config.redis, 'SENTINEL_MASTER_NAME', "mymaster")
    monkeypatch.setattr(app_config.redis, 'SENTINEL_HOSTS', "mock_sentinel_host:26379")
    monkeypatch.setattr(app_config.redis, 'SOCKET_TIMEOUT', 5)
    monkeypatch.setattr(app_config.redis, 'SOCKET_CONNECT_TIMEOUT', 5)
    
    # For RedisMessageBroker defaults if used
    monkeypatch.setattr(app_config.redis, 'STREAM_NAME', "default_test_stream")
    monkeypatch.setattr(app_config.redis, 'CONSUMER_GROUP_NAME', "default_test_group")
    monkeypatch.setattr(app_config.redis, 'JOB_STATUS_KEY_PREFIX', "job_status:")
    monkeypatch.setattr(app_config.redis, 'JOB_STATUS_EXPIRY_SECONDS', 3600)
    monkeypatch.setattr(app_config.redis, 'STREAM_MAX_LEN', 10000)

@pytest.fixture
def mock_redis_manager_logger():
    """Fixture to mock the _logger in redis_manager.py."""
    with patch('viralStoryGenerator.utils.redis_manager._logger') as mock_logger:
        yield mock_logger

# --- Tests for RedisManager ---

# Scenario 1: Successful connection
@pytest.mark.asyncio
@patch('redis.asyncio.ConnectionPool.from_url')
@patch('redis.asyncio.Redis') # Patch the Redis class itself
async def test_redis_manager_connect_successful(
    MockRedisClass, MockConnPoolFromUrl, mock_redis_manager_logger, mock_appconfig_redis_defaults
):
    mock_pool_instance = MagicMock()
    MockConnPoolFromUrl.return_value = mock_pool_instance
    
    mock_redis_client_instance = AsyncMock() # This is what Redis(...) returns
    mock_redis_client_instance.ping = AsyncMock(return_value=True)
    MockRedisClass.return_value = mock_redis_client_instance

    manager = rm_module.RedisManager(
        host=app_config.redis.HOST,
        port=app_config.redis.PORT,
        password=app_config.redis.PASSWORD,
        db=app_config.redis.DB
    )
    
    await manager.connect()

    assert manager.pool is mock_pool_instance
    assert manager.client is mock_redis_client_instance
    
    expected_redis_url = f"redis://:{app_config.redis.PASSWORD}@{app_config.redis.HOST}:{app_config.redis.PORT}/{app_config.redis.DB}"
    MockConnPoolFromUrl.assert_called_once_with(
        expected_redis_url,
        socket_timeout=app_config.redis.SOCKET_TIMEOUT,
        socket_connect_timeout=app_config.redis.SOCKET_CONNECT_TIMEOUT,
        decode_responses=False # Default for manager, broker might override
    )
    MockRedisClass.assert_called_once_with(connection_pool=mock_pool_instance)
    mock_redis_client_instance.ping.assert_called_once()
    mock_redis_manager_logger.info.assert_any_call(f"Successfully connected to Redis at {expected_redis_url}")


# Scenario 2: Connection failure
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage, exception_raised", [
    ("pool_from_url", ConnectionError("Pool creation failed")),
    ("redis_ping", TimeoutError("Ping timed out")),
    ("redis_ping_conn_error", ConnectionError("Ping connection failed")),
])
@patch('redis.asyncio.ConnectionPool.from_url')
@patch('redis.asyncio.Redis')
async def test_redis_manager_connect_failure(
    MockRedisClass, MockConnPoolFromUrl, mock_redis_manager_logger, 
    mock_appconfig_redis_defaults, failure_stage, exception_raised
):
    if failure_stage == "pool_from_url":
        MockConnPoolFromUrl.side_effect = exception_raised
    elif failure_stage in ["redis_ping", "redis_ping_conn_error"]:
        mock_pool_instance = MagicMock()
        MockConnPoolFromUrl.return_value = mock_pool_instance
        
        mock_redis_client_instance = AsyncMock()
        mock_redis_client_instance.ping = AsyncMock(side_effect=exception_raised)
        MockRedisClass.return_value = mock_redis_client_instance

    manager = rm_module.RedisManager(host="fail_host", port=1234) # Use different host to avoid config interference
    
    with pytest.raises(type(exception_raised)) as exc_info:
        await manager.connect()
    
    assert manager.pool is None
    assert manager.client is None
    assert str(exception_raised) in str(exc_info.value) # Check if original exception message is part of the raised one
    
    if failure_stage == "pool_from_url":
        mock_redis_manager_logger.error.assert_any_call(
            f"Failed to create Redis connection pool for redis://:None@fail_host:1234/0. Error: {exception_raised}"
        )
    else: # redis_ping failures
        mock_redis_manager_logger.error.assert_any_call(
             f"Failed to connect to Redis (ping failed) for redis://:None@fail_host:1234/0. Error: {exception_raised}"
        )

# Scenario 3: Get client
@pytest.mark.asyncio
async def test_redis_manager_get_client(mock_appconfig_redis_defaults):
    manager = rm_module.RedisManager(host="test_host", port=1234)
    
    # Client not initialized
    with pytest.raises(RuntimeError, match="Redis client not initialized. Call connect() first."):
        manager.get_client()
        
    # Simulate successful connection
    mock_client = AsyncMock()
    manager.client = mock_client
    
    client_instance = manager.get_client()
    assert client_instance is mock_client


# Scenario 4: Close
@pytest.mark.asyncio
async def test_redis_manager_close(mock_redis_manager_logger, mock_appconfig_redis_defaults):
    manager = rm_module.RedisManager(host="close_test_host", port=1234)
    
    # Case 1: Client and pool are None (connect was never called or failed early)
    manager.client = None
    manager.pool = None
    await manager.close()
    mock_redis_manager_logger.debug.assert_any_call("Redis client or pool not initialized, nothing to close.")
    
    # Case 2: Client and pool are set
    mock_client_instance_to_close = AsyncMock()
    mock_pool_instance_to_close = AsyncMock()
    manager.client = mock_client_instance_to_close
    manager.pool = mock_pool_instance_to_close
    
    mock_redis_manager_logger.reset_mock() # Reset logger before the call
    await manager.close()
    
    mock_client_instance_to_close.close.assert_called_once()
    mock_pool_instance_to_close.disconnect.assert_called_once()
    mock_redis_manager_logger.info.assert_any_call("Redis connection closed.")
    assert manager.client is None
    assert manager.pool is None

    # Case 3: Client.close() raises exception
    mock_client_instance_exc = AsyncMock()
    mock_client_instance_exc.close = AsyncMock(side_effect=Exception("Client close error"))
    mock_pool_instance_exc_client = AsyncMock()
    manager.client = mock_client_instance_exc
    manager.pool = mock_pool_instance_exc_client
    
    mock_redis_manager_logger.reset_mock()
    await manager.close() # Should not propagate, but log
    mock_redis_manager_logger.error.assert_any_call(
        "Error closing Redis client: Client close error", exc_info=True
    )
    # Pool disconnect should still be attempted
    mock_pool_instance_exc_client.disconnect.assert_called_once()


    # Case 4: Pool.disconnect() raises exception
    mock_client_instance_exc_pool = AsyncMock()
    mock_pool_instance_exc = AsyncMock()
    mock_pool_instance_exc.disconnect = AsyncMock(side_effect=Exception("Pool disconnect error"))
    manager.client = mock_client_instance_exc_pool
    manager.pool = mock_pool_instance_exc
    
    mock_redis_manager_logger.reset_mock()
    await manager.close() # Should not propagate, but log
    mock_client_instance_exc_pool.close.assert_called_once() # Client close was called
    mock_redis_manager_logger.error.assert_any_call(
        "Error disconnecting Redis connection pool: Pool disconnect error", exc_info=True
    )

# --- Tests for RedisMessageBroker ---

# Scenario 1: Initialization

@pytest.mark.asyncio
@patch.object(rm_module.RedisManager, 'connect', new_callable=AsyncMock) # Patch RedisManager's connect
@patch.object(rm_module.RedisManager, 'get_client') # Patch get_client
async def test_redis_message_broker_initialize_successful(
    mock_get_client, mock_manager_connect, 
    mock_redis_manager_logger, # Use the logger fixture from the module
    mock_appconfig_redis_defaults # Ensure app_config is mocked
):
    mock_redis_client_instance = AsyncMock() # This is what manager.get_client() will return
    mock_redis_client_instance.xgroup_create = AsyncMock() # Mock xgroup_create on this client
    mock_get_client.return_value = mock_redis_client_instance

    broker = rm_module.RedisMessageBroker(
        stream_name="test_stream_init_ok",
        consumer_group_name="test_group_init_ok",
        consumer_name="test_consumer_init_ok"
    )
    
    await broker.initialize()

    mock_manager_connect.assert_called_once() # RedisManager.connect was called
    mock_get_client.assert_called_once() # client was retrieved
    
    mock_redis_client_instance.xgroup_create.assert_called_once_with(
        name="test_stream_init_ok", 
        groupname="test_group_init_ok", 
        id="0",  # Default starting ID
        mkstream=True
    )
    assert broker._initialized is True
    mock_redis_manager_logger.info.assert_any_call(
        f"Consumer group 'test_group_init_ok' created/ensured for stream 'test_stream_init_ok'."
    )


@pytest.mark.asyncio
@patch.object(rm_module.RedisManager, 'connect', new_callable=AsyncMock)
@patch.object(rm_module.RedisManager, 'get_client')
async def test_redis_message_broker_initialize_busygroup_handled(
    mock_get_client, mock_manager_connect, 
    mock_redis_manager_logger, mock_appconfig_redis_defaults
):
    mock_redis_client_instance = AsyncMock()
    # Simulate BUSYGROUP error from xgroup_create
    mock_redis_client_instance.xgroup_create = AsyncMock(
        side_effect=ResponseError("BUSYGROUP Consumer Group name already exists")
    )
    mock_get_client.return_value = mock_redis_client_instance

    broker = rm_module.RedisMessageBroker("stream_busy", "group_busy", "consumer_busy")
    await broker.initialize() # Should not raise an exception

    assert broker._initialized is True # Still considered initialized
    mock_redis_manager_logger.info.assert_any_call(
        "Consumer group 'group_busy' already exists for stream 'stream_busy'. Proceeding."
    )

@pytest.mark.asyncio
@patch.object(rm_module.RedisManager, 'connect', new_callable=AsyncMock)
@patch.object(rm_module.RedisManager, 'get_client')
async def test_redis_message_broker_initialize_xgroup_create_other_error(
    mock_get_client, mock_manager_connect, 
    mock_redis_manager_logger, mock_appconfig_redis_defaults
):
    mock_redis_client_instance = AsyncMock()
    other_redis_error = ResponseError("Some other Redis error")
    mock_redis_client_instance.xgroup_create = AsyncMock(side_effect=other_redis_error)
    mock_get_client.return_value = mock_redis_client_instance

    broker = rm_module.RedisMessageBroker("stream_err", "group_err", "consumer_err")
    
    with pytest.raises(ResponseError, match="Some other Redis error"):
        await broker.initialize()
    
    assert broker._initialized is False
    mock_redis_manager_logger.error.assert_any_call(
        f"Failed to create consumer group 'group_err' for stream 'stream_err': {other_redis_error}",
        exc_info=True
    )


@pytest.mark.asyncio
@patch.object(rm_module.RedisManager, 'connect', new_callable=AsyncMock)
async def test_redis_message_broker_initialize_connect_fails(
    mock_manager_connect_fails, mock_redis_manager_logger, mock_appconfig_redis_defaults
):
    connect_exception = ConnectionError("Cannot connect to Redis server")
    mock_manager_connect_fails.side_effect = connect_exception

    broker = rm_module.RedisMessageBroker("stream_conn_fail", "group_conn_fail", "consumer_conn_fail")
    
    with pytest.raises(ConnectionError, match="Cannot connect to Redis server"):
        await broker.initialize()
        
    assert broker._initialized is False
    # The error logging for connect failure is handled by RedisManager itself,
    # but initialize might also log its failure.
    # Based on current code, initialize itself doesn't log the connect error again if connect() propagates it.


def test_redis_message_broker_ensure_initialized_raises(mock_appconfig_redis_defaults):
    broker = rm_module.RedisMessageBroker("stream_not_init", "group_not_init", "consumer_not_init")
    # _initialized is False by default
    
    with pytest.raises(RuntimeError, match="RedisMessageBroker is not initialized. Call initialize() first."):
        broker._ensure_initialized() # Call the protected method for testing
    
    # Try calling a method that uses it
    with pytest.raises(RuntimeError, match="RedisMessageBroker is not initialized. Call initialize() first."):
        # Need an async context to await publish_message
        async def try_publish():
            await broker.publish_message({"data": "test"})
        asyncio.run(try_publish())

# Scenario 2: Serialization/Deserialization (_serialize_value, _deserialize_value)

class DummyPydanticModel(BaseModel):
    id: int
    name: str
    url: HttpUrl # Test HttpUrl serialization

@pytest.mark.parametrize("input_value, expected_serialized", [
    ({"key": "value", "num": 123}, '{"key": "value", "num": 123}'),
    (["item1", 2, False], '["item1", 2, false]'),
    ("a simple string", '"a simple string"'), # JSON spec requires strings to be quoted
    (123.45, "123.45"),
    (True, "true"),
    (None, "null"),
    (
        DummyPydanticModel(id=1, name="test", url=HttpUrl("http://example.com")),
        '{"id": 1, "name": "test", "url": "http://example.com/"}' # Pydantic model should be serialized to dict then JSON
    ),
    (
        {"dt": datetime(2023, 1, 1, 12, 30, 0), "uid": uuid.UUID("123e4567-e89b-12d3-a456-426614174000")},
        '{"dt": "2023-01-01T12:30:00", "uid": "123e4567-e89b-12d3-a456-426614174000"}'
    )
])
def test_serialize_value(input_value, expected_serialized, mock_appconfig_redis_defaults):
    broker = rm_module.RedisMessageBroker("s", "g", "c") # Instance needed to call method
    serialized = broker._serialize_value(input_value)
    # Compare loaded JSON for dicts/lists to ignore key order issues if any
    if isinstance(input_value, (dict, list)):
        assert json.loads(serialized) == json.loads(expected_serialized)
    else:
        assert serialized == expected_serialized

def test_serialize_value_unserializable(mock_appconfig_redis_defaults):
    broker = rm_module.RedisMessageBroker("s", "g", "c")
    class Unserializable: pass
    with pytest.raises(TypeError):
        broker._serialize_value(Unserializable())

@pytest.mark.parametrize("serialized_value, expected_deserialized", [
    ('{"key": "value", "num": 123}', {"key": "value", "num": 123}),
    ('["item1", 2, false]', ["item1", 2, False]),
    ('"a simple string"', "a simple string"),
    ("123.45", 123.45),
    ("true", True),
    ("null", None),
    # Pydantic models are not automatically deserialized back to model instances by _deserialize_value
    # It will just return a dict.
    ('{"id": 1, "name": "test", "url": "http://example.com/"}', 
     {"id": 1, "name": "test", "url": "http://example.com/"}),
    # Datetime and UUID are not automatically converted back from string by json.loads
    ('{"dt": "2023-01-01T12:30:00", "uid": "123e4567-e89b-12d3-a456-426614174000"}',
     {"dt": "2023-01-01T12:30:00", "uid": "123e4567-e89b-12d3-a456-426614174000"})
])
def test_deserialize_value(serialized_value, expected_deserialized, mock_appconfig_redis_defaults):
    broker = rm_module.RedisMessageBroker("s", "g", "c")
    deserialized = broker._deserialize_value(serialized_value)
    assert deserialized == expected_deserialized

@pytest.mark.parametrize("invalid_json_string", [
    "not a json string",
    "{'key': 'value', 'malformed_json': True", # Malformed
])
def test_deserialize_value_invalid_json(invalid_json_string, mock_redis_manager_logger, mock_appconfig_redis_defaults):
    broker = rm_module.RedisMessageBroker("s", "g", "c")
    deserialized = broker._deserialize_value(invalid_json_string)
    assert deserialized is None # Should return None on JSONDecodeError
    mock_redis_manager_logger.error.assert_called_once()
    # Check that the log message contains info about the decode error and the problematic string
    args, _ = mock_redis_manager_logger.error.call_args
    assert "Failed to deserialize value from Redis" in args[0]
    assert invalid_json_string in args[0]

def test_deserialize_value_none_input(mock_appconfig_redis_defaults):
    broker = rm_module.RedisMessageBroker("s", "g", "c")
    assert broker._deserialize_value(None) is None

# Scenario 3: RedisMessageBroker.publish_message

@pytest.mark.asyncio
async def test_publish_message_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.xadd = AsyncMock(return_value="12345-0") # Successful xadd returns message ID
    
    # Simulate initialized broker
    broker = rm_module.RedisMessageBroker("test_stream_pub", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance) # Assign mock client

    payload_data = {"key": "value", "number": 1}
    message_id = await broker.publish_message(payload_data)

    assert message_id == "12345-0"
    mock_client_instance.xadd.assert_called_once()
    args, kwargs = mock_client_instance.xadd.call_args
    
    assert args[0] == "test_stream_pub" # Stream name
    # Payload should be wrapped: {'job_id': 'xxx', 'data': 'json_string_of_payload_data'}
    # job_id is auto-generated if not in payload_data.
    # Let's check the structure of the fields passed to xadd (kwargs['fields'])
    xadd_fields = kwargs['fields']
    assert "job_id" in xadd_fields
    assert isinstance(uuid.UUID(xadd_fields["job_id"]), uuid.UUID) # Check it's a valid UUID string
    
    # data field should be the JSON string of original payload_data
    deserialized_data_field = json.loads(xadd_fields["data"])
    assert deserialized_data_field == payload_data
    
    assert kwargs['maxlen'] == app_config.redis.STREAM_MAX_LEN
    assert kwargs['approximate'] is True


@pytest.mark.asyncio
async def test_publish_message_with_provided_job_id(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.xadd = AsyncMock(return_value="12346-0")
    broker = rm_module.RedisMessageBroker("test_stream_pub_jobid", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    custom_job_id = "my_custom_job_id_789"
    payload_data = {"job_id": custom_job_id, "info": "extra"}
    
    message_id = await broker.publish_message(payload_data)
    assert message_id == "12346-0"
    
    xadd_fields = mock_client_instance.xadd.call_args[1]['fields']
    assert xadd_fields["job_id"] == custom_job_id # Provided job_id is used
    deserialized_data_field = json.loads(xadd_fields["data"])
    assert deserialized_data_field == payload_data # Entire original payload becomes 'data'


@pytest.mark.asyncio
@pytest.mark.parametrize("xadd_failure_return", [None, ResponseError("XADD FAILED")])
async def test_publish_message_xadd_fails(
    xadd_failure_return, mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    if isinstance(xadd_failure_return, Exception):
        mock_client_instance.xadd = AsyncMock(side_effect=xadd_failure_return)
    else: # None
        mock_client_instance.xadd = AsyncMock(return_value=xadd_failure_return)
        
    broker = rm_module.RedisMessageBroker("test_stream_xadd_fail", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    payload_data = {"key": "value"}
    message_id = await broker.publish_message(payload_data)

    assert message_id is None
    mock_client_instance.xadd.assert_called_once()
    
    # Check for error log
    error_log_found = False
    expected_error_str = str(xadd_failure_return) if isinstance(xadd_failure_return, Exception) else "xadd returned None"
    for call_arg in mock_redis_manager_logger.error.call_args_list:
        if f"Failed to publish message to stream test_stream_xadd_fail" in call_arg[0][0] and \
           expected_error_str in call_arg[0][0]: # Error message should be part of the log
            error_log_found = True
            break
    assert error_log_found, f"Expected error log for xadd failure not found. Actual logs: {mock_redis_manager_logger.error.call_args_list}"


@pytest.mark.asyncio
async def test_publish_message_serialization_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock() # xadd should not be called
    broker = rm_module.RedisMessageBroker("test_stream_ser_fail", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    class UnserializableObject: pass
    payload_data = {"data": UnserializableObject()} # This will fail _serialize_value

    message_id = await broker.publish_message(payload_data)

    assert message_id is None
    mock_client_instance.xadd.assert_not_called()
    mock_redis_manager_logger.error.assert_any_call(
        f"Failed to serialize message payload for stream test_stream_ser_fail: Object of type UnserializableObject is not JSON serializable",
        exc_info=True
    )

# Scenario 4: RedisMessageBroker.consume_messages

@pytest.mark.asyncio
async def test_consume_messages_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    stream_name = "test_stream_consume_ok"
    group_name = "test_group_consume_ok"
    consumer_name = "test_consumer_consume_ok"
    
    # Raw response from xreadgroup: {stream_name: [(message_id, {field1: val1, ...})]}
    # Note: fields and values are bytes.
    raw_messages_from_redis = {
        stream_name.encode('utf-8'): [
            (b"12345-0", {b"job_id": b"job1", b"data": b'{"key": "val1"}'}),
            (b"12346-0", {b"job_id": b"job2", b"data": b'{"key": "val2"}'}),
        ]
    }
    mock_client_instance.xreadgroup = AsyncMock(return_value=raw_messages_from_redis)
    
    broker = rm_module.RedisMessageBroker(stream_name, group_name, consumer_name)
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    batch_size = 5
    block_ms = 1000
    messages = await broker.consume_messages(batch_size, block_ms)

    assert len(messages) == 2
    # First message
    msg_id1, job_data1_raw = messages[0]
    assert msg_id1 == "12345-0"
    assert job_data1_raw == {b"job_id": b"job1", b"data": b'{"key": "val1"}'} # consume_messages returns raw dict
    
    # Second message
    msg_id2, job_data2_raw = messages[1]
    assert msg_id2 == "12346-0"
    assert job_data2_raw == {b"job_id": b"job2", b"data": b'{"key": "val2"}'}

    mock_client_instance.xreadgroup.assert_called_once_with(
        groupname=group_name,
        consumername=consumer_name,
        streams={stream_name: ">"}, # ">" means new messages only
        count=batch_size,
        block=block_ms
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("xreadgroup_empty_return", [None, {}, {b"other_stream": []}])
async def test_consume_messages_empty_batch(
    xreadgroup_empty_return, mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.xreadgroup = AsyncMock(return_value=xreadgroup_empty_return)
    
    stream_name = "test_stream_consume_empty"
    broker = rm_module.RedisMessageBroker(stream_name, "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    messages = await broker.consume_messages(5, 1000)

    assert messages == [] # Should be an empty list
    mock_client_instance.xreadgroup.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("redis_exception, log_fragment", [
    (ConnectionError("Consume connection failed"), "Connection error during message consumption"),
    (ResponseError("NOGROUP No such consumer group"), "NOGROUP error during message consumption"),
    (TimeoutError("Consume timed out"), "Timeout error during message consumption"),
])
async def test_consume_messages_redis_errors(
    redis_exception, log_fragment, mock_redis_manager_logger, 
    mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.xreadgroup = AsyncMock(side_effect=redis_exception)
    
    broker = rm_module.RedisMessageBroker("test_stream_consume_err", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    messages = await broker.consume_messages(5, 1000)

    assert messages == [] # Should return empty list on error
    mock_client_instance.xreadgroup.assert_called_once()
    
    error_log_found = False
    for call_arg in mock_redis_manager_logger.error.call_args_list:
        if log_fragment in call_arg[0][0] and str(redis_exception) in call_arg[0][0]:
            error_log_found = True
            break
    assert error_log_found, f"Expected log fragment '{log_fragment}' with exception details not found."

# Scenario 5: RedisMessageBroker.acknowledge_message

@pytest.mark.asyncio
async def test_acknowledge_message_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.xack = AsyncMock(return_value=1) # 1 means message acknowledged
    
    stream_name = "test_stream_ack_ok"
    group_name = "test_group_ack_ok"
    broker = rm_module.RedisMessageBroker(stream_name, group_name, "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    message_id_to_ack = "12345-0"
    ack_result = await broker.acknowledge_message(message_id_to_ack) # Group name taken from broker's attribute

    assert ack_result is True
    mock_client_instance.xack.assert_called_once_with(stream_name, group_name, message_id_to_ack)
    mock_redis_manager_logger.debug.assert_any_call(
        f"Acknowledged message ID {message_id_to_ack} from group {group_name} in stream {stream_name}."
    )


@pytest.mark.asyncio
async def test_acknowledge_message_non_existent_or_already_acked(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.xack = AsyncMock(return_value=0) # 0 means message did not exist or was already acked
    
    broker = rm_module.RedisMessageBroker("s_ack_0", "g_ack_0", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    message_id_to_ack = "non_existent_msg_id"
    ack_result = await broker.acknowledge_message(message_id_to_ack)

    assert ack_result is False # Returns False if xack returns 0
    mock_client_instance.xack.assert_called_once()
    mock_redis_manager_logger.warning.assert_any_call(
        f"Failed to acknowledge message ID {message_id_to_ack} from group g_ack_0 in stream s_ack_0 (XACK returned 0). Message might not exist or already be ACKed."
    )


@pytest.mark.asyncio
async def test_acknowledge_message_redis_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    redis_error = ResponseError("XACK error")
    mock_client_instance.xack = AsyncMock(side_effect=redis_error)
    
    broker = rm_module.RedisMessageBroker("s_ack_err", "g_ack_err", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    message_id_to_ack = "msg_id_for_ack_error"
    ack_result = await broker.acknowledge_message(message_id_to_ack)

    assert ack_result is False
    mock_client_instance.xack.assert_called_once()
    mock_redis_manager_logger.error.assert_any_call(
        f"Error acknowledging message ID {message_id_to_ack} from group g_ack_err in stream s_ack_err: {redis_error}",
        exc_info=True
    )

# Scenario 6: RedisMessageBroker.track_job_progress and get_job_progress

@pytest.mark.asyncio
async def test_track_job_progress_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.setex = AsyncMock(return_value=True) # setex returns True on success
    
    broker = rm_module.RedisMessageBroker("s_track", "g_track", "c_track")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    job_id = "job_track_progress_001"
    progress_info = {"status": "processing", "percentage": 50, "message": "Halfway there"}
    
    await broker.track_job_progress(job_id, progress_info)

    expected_key = f"{app_config.redis.JOB_STATUS_KEY_PREFIX}{job_id}"
    expected_value = broker._serialize_value(progress_info) # Use internal serialization
    expected_ttl = app_config.redis.JOB_STATUS_EXPIRY_SECONDS
    
    mock_client_instance.setex.assert_called_once_with(
        expected_key, expected_ttl, expected_value
    )
    mock_redis_manager_logger.debug.assert_any_call(
        f"Tracked progress for job {job_id}. Status: processing, Percentage: 50"
    )


@pytest.mark.asyncio
async def test_track_job_progress_redis_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    redis_error = ResponseError("SETEX error")
    mock_client_instance.setex = AsyncMock(side_effect=redis_error)
    
    broker = rm_module.RedisMessageBroker("s_track_err", "g_track_err", "c_track_err")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    job_id = "job_track_err_002"
    progress_info = {"status": "failed", "error": "Something broke"}

    await broker.track_job_progress(job_id, progress_info) # Should handle exception

    mock_client_instance.setex.assert_called_once() # Attempted
    mock_redis_manager_logger.error.assert_any_call(
        f"Error tracking progress for job {job_id} in Redis: {redis_error}", exc_info=True
    )


@pytest.mark.asyncio
async def test_get_job_progress_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    job_id = "job_get_prog_ok_003"
    expected_progress_info = {"status": "completed", "result": "All done!"}
    serialized_progress = json.dumps(expected_progress_info) # Assume simple serialization for test
    
    mock_client_instance.get = AsyncMock(return_value=serialized_progress.encode('utf-8')) # GET returns bytes
    
    broker = rm_module.RedisMessageBroker("s_get_prog", "g_get_prog", "c_get_prog")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    progress_info = await broker.get_job_progress(job_id)

    assert progress_info == expected_progress_info
    expected_key = f"{app_config.redis.JOB_STATUS_KEY_PREFIX}{job_id}"
    mock_client_instance.get.assert_called_once_with(expected_key)


@pytest.mark.asyncio
async def test_get_job_progress_not_found(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=None) # Key not found
    
    broker = rm_module.RedisMessageBroker("s_get_prog_none", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    job_id = "job_get_prog_not_found_004"
    progress_info = await broker.get_job_progress(job_id)

    assert progress_info is None
    mock_client_instance.get.assert_called_once()


@pytest.mark.asyncio
async def test_get_job_progress_invalid_json(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    invalid_json_str = "this is not json".encode('utf-8')
    mock_client_instance.get = AsyncMock(return_value=invalid_json_str)
    
    broker = rm_module.RedisMessageBroker("s_get_prog_badjson", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    job_id = "job_get_prog_bad_json_005"
    progress_info = await broker.get_job_progress(job_id)

    assert progress_info is None # Should return None on deserialization error
    mock_client_instance.get.assert_called_once()
    # Check that _deserialize_value logged an error (which it does)
    # The error log comes from _deserialize_value.
    # We can check if _deserialize_value was called and returned None,
    # and that it logged. For this test, checking the return is sufficient.
    # If we want to check the log, we'd need to ensure _deserialize_value's logger is the same.
    # The logger is on the broker instance.
    mock_redis_manager_logger.error.assert_any_call(
        f"Failed to deserialize value from Redis: {invalid_json_str.decode('utf-8')}. Error: Expecting value: line 1 column 1 (char 0)"
    )


@pytest.mark.asyncio
async def test_get_job_progress_redis_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    redis_error = ConnectionError("GET connection failed")
    mock_client_instance.get = AsyncMock(side_effect=redis_error)
    
    broker = rm_module.RedisMessageBroker("s_get_prog_err", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    job_id = "job_get_prog_err_006"
    progress_info = await broker.get_job_progress(job_id)

    assert progress_info is None
    mock_client_instance.get.assert_called_once()
    mock_redis_manager_logger.error.assert_any_call(
        f"Error retrieving job progress for job {job_id} from Redis: {redis_error}", exc_info=True
    )

# Scenario 7: RedisMessageBroker.get_queue_information

@pytest.mark.asyncio
async def test_get_queue_information_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    stream_name = "test_stream_qinfo_ok"
    group_name = "test_group_qinfo_ok"
    
    # Mock data for xinfo_stream
    # Example from redis-py: {'length': 2, 'radix-tree-keys': 1, 'radix-tree-nodes': 2, 'last-generated-id': b'1678886400001-0', 'groups': 1, 'first-entry': (b'1678886400000-0', {b'field1': b'value1'}), 'last-entry': (b'1678886400001-0', {b'field2': b'value2'})}
    mock_client_instance.xinfo_stream = AsyncMock(return_value={
        "length": 10, "last-generated-id": "12345-0", "groups": 1
    })
    # Mock data for xinfo_groups
    # Example: [{'name': b'mygroup', 'consumers': 2, 'pending': 2, 'last-delivered-id': b'1678886400001-0', 'entries-read': 10, 'lag': 0}]
    mock_client_instance.xinfo_groups = AsyncMock(return_value=[
        {"name": group_name.encode('utf-8'), "consumers": 2, "pending": 5, "last-delivered-id": "12340-0"}
    ])
    # Mock data for xinfo_consumers
    # Example: [{'name': b'consumer1', 'pending': 1, 'idle': 10000}, {'name': b'consumer2', 'pending': 1, 'idle': 15000}]
    mock_client_instance.xinfo_consumers = AsyncMock(return_value=[
        {"name": "consumer_a".encode('utf-8'), "pending": 3, "idle": 1000},
        {"name": "consumer_b".encode('utf-8'), "pending": 2, "idle": 2000},
    ])
    # Mock data for xrevrange (recent messages)
    # Example: [(b'1678886400001-0', {b'job_id': b'job1', b'data': b'{"info":"recent1"}'})]
    mock_client_instance.xrevrange = AsyncMock(return_value=[
        ("12345-0".encode('utf-8'), {b"job_id": b"job_recent1", b"data": b'{"detail": "recent data 1"}'}),
        ("12344-0".encode('utf-8'), {b"job_id": b"job_recent2", b"data": b'{"detail": "recent data 2"}'}),
    ])
    # Mock data for get_job_progress (DLQ simulation - not directly part of get_queue_information for main stream)
    # This test focuses on main stream info. DLQ length is separate.
    
    broker = rm_module.RedisMessageBroker(stream_name, group_name, "c_qinfo")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    # Mock internal get_pending_messages_info to simulate DLQ check or simplify
    # For this test, let's assume DLQ info is not part of this specific call path,
    # or get_pending_messages_info for DLQ (if any) returns empty or is mocked.
    # The function primarily uses xinfo_*, xrevrange.

    queue_info = await broker.get_queue_information()

    assert queue_info["status"] == "ok"
    assert queue_info["stream_name"] == stream_name
    assert queue_info["stream_length"] == 10
    assert queue_info["pending_messages_count"] == 5 # From xinfo_groups[0]['pending']
    
    assert len(queue_info["consumer_groups"]) == 1
    cg_info = queue_info["consumer_groups"][0]
    assert cg_info["name"] == group_name
    assert cg_info["consumers_count"] == 2 # From xinfo_groups[0]['consumers']
    assert cg_info["pending_messages"] == 5 # From xinfo_groups[0]['pending']
    
    # xinfo_consumers data is directly used if available
    assert len(cg_info["consumers"]) == 2
    assert cg_info["consumers"][0]["name"] == "consumer_a"
    assert cg_info["consumers"][0]["pending_messages"] == 3
    assert cg_info["consumers"][0]["inactive_time_ms"] == 1000
    
    assert len(queue_info["recent_messages"]) == 2
    assert queue_info["recent_messages"][0]["message_id"] == "12345-0"
    assert queue_info["recent_messages"][0]["data"] == {"job_id": "job_recent1", "detail": "recent data 1"} # Deserialized
    
    mock_client_instance.xinfo_stream.assert_called_once_with(stream_name)
    mock_client_instance.xinfo_groups.assert_called_once_with(stream_name)
    mock_client_instance.xinfo_consumers.assert_any_call(stream_name, group_name) # Called for each group
    mock_client_instance.xrevrange.assert_called_once_with(stream_name, count=10) # Default count


@pytest.mark.asyncio
async def test_get_queue_information_stream_not_exist(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    # xinfo_stream raises ResponseError for "no such key"
    mock_client_instance.xinfo_stream = AsyncMock(side_effect=ResponseError("ERR no such key"))
    
    broker = rm_module.RedisMessageBroker("non_existent_stream", "g", "c")
    monkeypatch.setattr(broker, '_initialized', True) # Assume broker is initialized for this test
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    queue_info = await broker.get_queue_information()

    assert queue_info["status"] == "error"
    assert "Stream 'non_existent_stream' does not exist" in queue_info["error_message"]
    assert queue_info["stream_name"] == "non_existent_stream"
    # Other fields should be default/empty
    assert queue_info["stream_length"] == 0
    assert queue_info["consumer_groups"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_command_mock, raised_exception", [
    ("xinfo_groups", ConnectionError("XINFO GROUPS failed")),
    ("xinfo_consumers", TimeoutError("XINFO CONSUMERS timed out")),
    ("xrevrange", ResponseError("XREVRANGE other error")),
])
async def test_get_queue_information_other_redis_errors(
    failing_command_mock, raised_exception,
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    stream_name = "test_stream_qinfo_err"
    
    # Default successful mocks, one will be overridden
    mock_client_instance.xinfo_stream = AsyncMock(return_value={"length": 5, "groups": 1})
    mock_client_instance.xinfo_groups = AsyncMock(return_value=[{"name": b"g1", "consumers":1, "pending":1}])
    mock_client_instance.xinfo_consumers = AsyncMock(return_value=[{"name": b"c1", "pending":1, "idle":100}])
    mock_client_instance.xrevrange = AsyncMock(return_value=[])

    # Make one command fail
    setattr(mock_client_instance, failing_command_mock, AsyncMock(side_effect=raised_exception))
    
    broker = rm_module.RedisMessageBroker(stream_name, "g1", "c1")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    queue_info = await broker.get_queue_information()

    assert queue_info["status"] == "error"
    assert f"Error during {failing_command_mock}: {raised_exception}" in queue_info["error_message"]
    assert queue_info["stream_name"] == stream_name # Still includes stream name
    
    mock_redis_manager_logger.error.assert_any_call(
        f"Error fetching queue information for stream {stream_name}: {raised_exception}", exc_info=True
    )

# Scenario 8: RedisMessageBroker.purge_stream_messages

@pytest.mark.asyncio
async def test_purge_stream_messages_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    num_deleted_entries = 100
    mock_client_instance.xtrim = AsyncMock(return_value=num_deleted_entries)
    
    stream_name = "test_stream_purge_ok"
    broker = rm_module.RedisMessageBroker(stream_name, "g_purge", "c_purge")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    result = await broker.purge_stream_messages()

    assert result == num_deleted_entries
    mock_client_instance.xtrim.assert_called_once_with(stream_name, 0, approximate=False)
    mock_redis_manager_logger.info.assert_any_call(
        f"Successfully purged {num_deleted_entries} messages from stream '{stream_name}'."
    )


@pytest.mark.asyncio
async def test_purge_stream_messages_redis_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    redis_error = ResponseError("XTRIM error")
    mock_client_instance.xtrim = AsyncMock(side_effect=redis_error)
    
    stream_name = "test_stream_purge_err"
    broker = rm_module.RedisMessageBroker(stream_name, "g_purge_err", "c_purge_err")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    result = await broker.purge_stream_messages()

    assert result == 0 # Returns 0 on error
    mock_client_instance.xtrim.assert_called_once_with(stream_name, 0, approximate=False)
    mock_redis_manager_logger.error.assert_any_call(
        f"Error purging messages from stream '{stream_name}': {redis_error}", exc_info=True
    )

# Scenario 9: RedisMessageBroker.get_stream_length

@pytest.mark.asyncio
async def test_get_stream_length_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    expected_length = 42
    mock_client_instance.xlen = AsyncMock(return_value=expected_length)
    
    stream_name = "test_stream_xlen_ok"
    broker = rm_module.RedisMessageBroker(stream_name, "g_xlen", "c_xlen")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    length = await broker.get_stream_length()

    assert length == expected_length
    mock_client_instance.xlen.assert_called_once_with(stream_name)


@pytest.mark.asyncio
async def test_get_stream_length_redis_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    redis_error = ResponseError("XLEN error")
    mock_client_instance.xlen = AsyncMock(side_effect=redis_error)
    
    stream_name = "test_stream_xlen_err"
    broker = rm_module.RedisMessageBroker(stream_name, "g_xlen_err", "c_xlen_err")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    length = await broker.get_stream_length()

    assert length == 0 # Returns 0 on error
    mock_client_instance.xlen.assert_called_once_with(stream_name)
    mock_redis_manager_logger.error.assert_any_call(
        f"Error getting length for stream '{stream_name}': {redis_error}", exc_info=True
    )


# Scenario 10: RedisMessageBroker.get_pending_messages_info

@pytest.mark.asyncio
async def test_get_pending_messages_info_successful(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    group_name = "test_group_pending_ok"
    consumer_name = "test_consumer_pending_ok" # Specific consumer
    stream_name = "test_stream_pending_ok"
    
    # Example raw response from redis-py for XPENDING with consumer:
    # [{'message_id': b'1678886400000-0', 'consumer': b'consumer-123', 'idle_time': 12345, 'delivery_count': 1}]
    raw_pending_data = [
        {'message_id': "1700000000000-0".encode('utf-8'), 'consumer': consumer_name.encode('utf-8'), 'idle_time': 5000, 'delivery_count': 1},
        {'message_id': "1700000001000-0".encode('utf-8'), 'consumer': consumer_name.encode('utf-8'), 'idle_time': 15000, 'delivery_count': 3},
    ]
    mock_client_instance.xpending_range = AsyncMock(return_value=raw_pending_data) # xpending_range is used in code
    
    broker = rm_module.RedisMessageBroker(stream_name, group_name, consumer_name) # Consumer name on broker not directly used by this method
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    min_idle_time_ms = 1000
    count = 10
    pending_info = await broker.get_pending_messages_info(group_name, consumer_name, min_idle_time_ms, count)

    assert len(pending_info) == 2
    assert pending_info[0]["message_id"] == "1700000000000-0"
    assert pending_info[0]["consumer"] == consumer_name
    assert pending_info[0]["idle_time_ms"] == 5000
    assert pending_info[0]["delivery_count"] == 1
    
    mock_client_instance.xpending_range.assert_called_once_with(
        name=stream_name, 
        groupname=group_name,
        min="-", # Start ID
        max="+", # End ID
        count=count,
        consumername=consumer_name, # Consumer specified
        idle=min_idle_time_ms
    )

@pytest.mark.asyncio
async def test_get_pending_messages_info_no_consumer(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    group_name = "test_group_pending_no_consumer"
    stream_name = "test_stream_pending_no_consumer"
    
    # If consumername is None, xpending_range returns a dict summary:
    # {'pending': 2, 'min_id': b'123-0', 'max_id': b'124-0', 'consumers': [{'name': b'c1', 'pending': 1}, ...]}
    # The function as written expects the list format, so this test needs to align with how xpending_range is called.
    # The function `get_pending_messages_info` *always* passes consumername=None to `xpending_range` if its own `consumer_name` arg is None.
    # When `consumername` is `None` to `xpending_range`, the return is a summary dict.
    # The code seems to expect a list of dicts from `xpending_range`, which happens when `consumername` *is* provided to `xpending_range`.
    # The current code is: `client.xpending_range(name=self.stream_name, groupname=group_name, min="-", max="+", count=count, consumername=consumer_name, idle=min_idle_time_ms)`
    # So, if `consumer_name` arg to `get_pending_messages_info` is `None`, it passes `None` to `xpending_range`.
    
    # Let's test the case where consumer_name is None for the method call
    # and xpending_range returns the summary dict (as per redis-py docs if consumername not specified in xpending_range call)
    # However, the code *always* passes a consumername to xpending_range if the method's consumer_name is not None.
    # If the method's consumer_name *is* None, it then calls client.xpending (not xpending_range) which has a different signature.
    # The code's logic for `consumer_name is None`:
    # `data = await client.xpending(name=self.stream_name, groupname=group_name)`
    # `return [{"consumer": cname.decode('utf-8'), "pending_messages": pcount} for cname, pcount in data["consumers"].items()]`
    
    mock_xpending_summary = {
        "pending": 5, "min": "1700000000000-0", "max": "1700000005000-0",
        "consumers": {
            "consumer_x".encode('utf-8'): 2, # name: pending_count
            "consumer_y".encode('utf-8'): 3
        }
    }
    mock_client_instance.xpending = AsyncMock(return_value=mock_xpending_summary)
    
    broker = rm_module.RedisMessageBroker(stream_name, group_name, "c_default_ignored")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    pending_info = await broker.get_pending_messages_info(group_name, consumer_name=None, min_idle_time_ms=0, count=0) # consumer_name=None

    assert len(pending_info) == 2
    assert {"consumer": "consumer_x", "pending_messages": 2} in pending_info
    assert {"consumer": "consumer_y", "pending_messages": 3} in pending_info
    
    mock_client_instance.xpending.assert_called_once_with(name=stream_name, groupname=group_name)
    mock_client_instance.xpending_range.assert_not_called()


@pytest.mark.asyncio
async def test_get_pending_messages_info_redis_error(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    mock_client_instance = AsyncMock()
    redis_error = ResponseError("XPENDING error")
    mock_client_instance.xpending_range = AsyncMock(side_effect=redis_error) # Test with consumer_name path
    
    broker = rm_module.RedisMessageBroker("s_pend_err", "g_pend_err", "c_pend_err")
    monkeypatch.setattr(broker, '_initialized', True)
    monkeypatch.setattr(broker.manager, 'client', mock_client_instance)

    pending_info = await broker.get_pending_messages_info("g_pend_err", "c_pend_err", 1000, 10)

    assert pending_info == [] # Returns empty list on error
    mock_client_instance.xpending_range.assert_called_once()
    mock_redis_manager_logger.error.assert_any_call(
        f"Error getting pending messages info for stream s_pend_err, group g_pend_err: {redis_error}", exc_info=True
    )


# Scenario 11: RedisMessageBroker.close
@pytest.mark.asyncio
async def test_redis_message_broker_close_calls_manager_close(
    mock_redis_manager_logger, mock_appconfig_redis_defaults, monkeypatch
):
    # manager.close is an async method. Ensure it's awaited.
    mock_manager_instance = MagicMock(spec=rm_module.RedisManager)
    mock_manager_instance.close = AsyncMock() # Make it an AsyncMock
    
    broker = rm_module.RedisMessageBroker("s_close", "g_close", "c_close")
    # Manually assign the mocked manager to the broker's _manager attribute
    monkeypatch.setattr(broker, '_manager', mock_manager_instance)
    monkeypatch.setattr(broker, '_initialized', True) # Assume it was initialized

    await broker.close()

    mock_manager_instance.close.assert_called_once()
    mock_redis_manager_logger.info.assert_any_call("RedisMessageBroker resources released.")
