import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import time

from viralStoryGenerator.src import api_handlers

@pytest.fixture
def mock_appconfig(monkeypatch):
    class DummyConfig:
        class storyboard:
            ENABLE_STORYBOARD_GENERATION = True
        ENABLE_IMAGE_GENERATION = True
        ENABLE_AUDIO_GENERATION = True
        AUDIO_QUEUE_DIR = "/tmp/audio_queue"
        class elevenLabs:
            API_KEY = "dummy"
        redis = type("redis", (), {"HOST": "localhost", "PORT": 6379})()
    monkeypatch.setattr(api_handlers, "appconfig", DummyConfig)

def test_create_story_task_success(mock_appconfig):
    with patch.object(api_handlers, "get_message_broker") as mock_broker:
        mock_instance = MagicMock()
        mock_instance.publish_message.return_value = "123-0"
        mock_broker.return_value = mock_instance

        result = api_handlers.create_story_task("Test Topic", sources_folder="folder", voice_id="voice1")
        assert result["topic"] == "Test Topic"
        assert result["status"] == "queued"
        assert "task_id" in result

def test_create_story_task_publish_fail(mock_appconfig):
    with patch.object(api_handlers, "get_message_broker") as mock_broker:
        mock_instance = MagicMock()
        mock_instance.publish_message.side_effect = Exception("fail")
        mock_broker.return_value = mock_instance

        with pytest.raises(RuntimeError):
            api_handlers.create_story_task("Test Topic")

@pytest.mark.asyncio
async def test_get_task_status_redis_completed(mock_appconfig):
    with patch.object(api_handlers, "get_message_broker") as mock_broker, \
         patch.object(api_handlers.storage_manager, "retrieve_file_content_as_json") as mock_storage:
        # Simulate Redis returns completed
        mock_instance = MagicMock()
        mock_instance.get_job_progress = AsyncMock(return_value={
            "status": "completed",
            "job_id": "abc",
            "story_script": "script",
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:05:00Z"
        })
        mock_broker.return_value = mock_instance
        # The handler may expect 'story_script', 'storyboard', 'audio_url', 'sources'
        mock_storage.return_value = {
            "story_script": "final_script",
            "storyboard": {"scenes": []},
            "audio_url": "audio.mp3",
            "sources": ["src1"],
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:05:00Z"
        }
        result = await api_handlers.get_task_status("abc")
        assert result is not None
        assert result["status"] == "completed"
        # The handler may return 'story_script', 'storyboard', 'audio_url', 'sources'
        assert result.get("story_script") == "final_script"
        assert result.get("audio_url") == "audio.mp3"
        assert isinstance(result.get("storyboard"), dict)
        assert isinstance(result.get("sources"), list)

@pytest.mark.asyncio
async def test_get_task_status_not_found(mock_appconfig):
    with patch.object(api_handlers, "get_message_broker") as mock_broker, \
         patch.object(api_handlers.storage_manager, "retrieve_file_content_as_json") as mock_storage:
        mock_instance = MagicMock()
        mock_instance.get_job_progress = AsyncMock(return_value=None)
        mock_broker.return_value = mock_instance
        mock_storage.side_effect = FileNotFoundError()
        result = await api_handlers.get_task_status("notfound")
        assert result is None

def test_process_audio_queue_no_dir(mock_appconfig, tmp_path, caplog):
    # Directory does not exist
    with patch("os.path.isdir", return_value=False):
        api_handlers.process_audio_queue()  # Should do nothing

def test_process_audio_queue_success(tmp_path, mock_appconfig):
    queue_dir = tmp_path / "audio_queue"
    queue_dir.mkdir()
    file_path = queue_dir / "job1.json"
    file_path.write_text('{"story": "test", "mp3_file_path": "out.mp3"}')

    with patch.object(api_handlers, "appconfig") as mock_cfg, \
         patch("os.listdir", return_value=["job1.json"]), \
         patch("os.remove") as mock_remove, \
         patch("builtins.open", open), \
         patch.object(api_handlers, "generate_elevenlabs_audio", return_value=True):
        mock_cfg.AUDIO_QUEUE_DIR = str(queue_dir)
        mock_cfg.elevenLabs.API_KEY = "dummy"
        api_handlers.process_audio_queue()
        mock_remove.assert_called_once()
