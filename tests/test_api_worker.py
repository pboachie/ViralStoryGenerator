import pytest
import asyncio
import sys
import signal
from unittest.mock import patch, MagicMock

import viralStoryGenerator.src.api_worker as api_worker

@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    # Reset global state before each test
    api_worker._message_broker = None
    api_worker._vector_db_client = None
    yield

def test_handle_shutdown(monkeypatch):
    called = {}
    def fake_exit(code):
        called['exit'] = code
        raise SystemExit()
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    monkeypatch.setattr(sys, "exit", fake_exit)
    # Should set shutdown_event and call sys.exit
    with pytest.raises(SystemExit):
        api_worker.handle_shutdown(signal.SIGINT, None)
    assert called['exit'] == 0

def test_preload_components_creates_group(monkeypatch):
    mock_broker = MagicMock()
    monkeypatch.setattr(api_worker, "RedisMessageBroker", lambda **kwargs: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    mock_broker.create_consumer_group.return_value = None
    mock_broker.ensure_stream_exists.return_value = None
    api_worker.preload_components("testgroup")
    mock_broker.create_consumer_group.assert_called_once_with(group_name="testgroup")
    mock_broker.ensure_stream_exists.assert_called_once()

def test_preload_components_group_exists(monkeypatch):
    mock_broker = MagicMock()
    monkeypatch.setattr(api_worker, "RedisMessageBroker", lambda **kwargs: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    # Simulate BUSYGROUP error
    mock_broker.create_consumer_group.side_effect = Exception("BUSYGROUP Consumer Group name already exists")
    mock_broker.ensure_stream_exists.return_value = None
    api_worker.preload_components("testgroup")
    mock_broker.ensure_stream_exists.assert_called_once()

def test_get_message_broker_initializes(monkeypatch):
    mock_broker = MagicMock()
    monkeypatch.setattr(api_worker, "RedisMessageBroker", lambda **kwargs: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    api_worker._message_broker = None
    broker = api_worker.get_message_broker()
    assert broker is mock_broker

def test_get_message_broker_returns_existing(monkeypatch):
    mock_broker = MagicMock()
    api_worker._message_broker = mock_broker
    assert api_worker.get_message_broker() is mock_broker

@pytest.mark.asyncio
async def test_process_api_jobs_handles_no_broker(monkeypatch):
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: None)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    # Patch shutdown_event to stop after one loop
    api_worker.shutdown_event.clear()
    async def stop_event():
        api_worker.shutdown_event.set()
    asyncio.get_event_loop().call_later(0.1, lambda: asyncio.create_task(stop_event()))
    await api_worker.process_api_jobs("group", "consumer")

@pytest.mark.asyncio
async def test_process_api_jobs_consumes_and_acknowledges(monkeypatch):
    # Setup a fake message broker
    mock_broker = MagicMock()
    # Simulate one message with job_id and job_type
    message_id = "123-0"
    message_data = {
        b"job_id": b"jid",
        b"job_type": b"story"
    }
    mock_broker.consume_messages.return_value = [
        ("stream", [(message_id, message_data)])
    ]
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: mock_broker)
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    # Patch process_api_job to just awaitable dummy
    monkeypatch.setattr(api_worker, "process_api_job", lambda *a, **k: asyncio.sleep(0))
    # Patch shutdown_event to stop after one loop
    api_worker.shutdown_event.clear()
    async def stop_event():
        api_worker.shutdown_event.set()
    asyncio.get_event_loop().call_later(0.1, lambda: asyncio.create_task(stop_event()))
    await api_worker.process_api_jobs("group", "consumer")
    mock_broker.acknowledge_message.assert_called_with("group", message_id)

def test_main_runs(monkeypatch):
    # Patch run_worker to raise KeyboardInterrupt to exit loop
    monkeypatch.setattr(api_worker, "run_worker", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(api_worker, "_logger", MagicMock())
    monkeypatch.setattr(api_worker, "asyncio", api_worker.asyncio)
    monkeypatch.setattr(api_worker, "os", api_worker.os)
    # Patch event loop methods
    loop = MagicMock()
    monkeypatch.setattr(api_worker, "sys", api_worker.sys)
    monkeypatch.setattr(api_worker, "signal", api_worker.signal)
    monkeypatch.setattr(api_worker, "uuid", api_worker.uuid)
    monkeypatch.setattr(api_worker, "get_vector_db_client", lambda: None)
    monkeypatch.setattr(api_worker, "get_vector_db", lambda: None)
    monkeypatch.setattr(api_worker, "get_message_broker", lambda: None)
    monkeypatch.setattr(api_worker.asyncio, "get_event_loop", lambda: loop)
    loop.run_until_complete.side_effect = KeyboardInterrupt()
    loop.shutdown_asyncgens.return_value = None
    loop.is_running.return_value = False
    loop.is_closed.return_value = True
    # Should not raise
    api_worker.main()
