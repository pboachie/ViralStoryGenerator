import pytest
import asyncio
import json
import os
import uuid # Added
import time # Added
from unittest.mock import patch, MagicMock, AsyncMock, mock_open # Added mock_open

from viralStoryGenerator.src import api_handlers
from viralStoryGenerator.utils.config import config as appconfig # For patching
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For type hints/spec
from viralStoryGenerator.models.models import JobStatusResponse # For asserting response types

# Basic test structure
class TestApiHandlers:

    @pytest.mark.asyncio
    async def test_get_message_broker(self, monkeypatch):
        """
        Tests that get_message_broker instantiates RedisMessageBroker correctly
        and calls initialize.
        """
        mock_redis_host = "testhost"
        mock_redis_port = "1234"
        
        monkeypatch.setattr(appconfig.redis, "HOST", mock_redis_host)
        monkeypatch.setattr(appconfig.redis, "PORT", int(mock_redis_port))

        expected_redis_url = f"redis://{mock_redis_host}:{mock_redis_port}"
        
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        
        with patch('viralStoryGenerator.src.api_handlers.RedisMessageBroker', return_value=mock_broker_instance) as mock_broker_class:
            broker = await api_handlers.get_message_broker()

            mock_broker_class.assert_called_once_with(
                redis_url=expected_redis_url,
                stream_name="api_jobs" 
            )
            mock_broker_instance.initialize.assert_awaited_once()
            assert broker == mock_broker_instance

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.uuid.uuid4')
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_create_story_task_success(self, mock_get_broker, mock_uuid, monkeypatch):
        mock_task_id = "test-uuid-123"
        mock_uuid.return_value = mock_task_id
        
        mock_broker = AsyncMock(spec=RedisMessageBroker)
        mock_broker.publish_message = AsyncMock(return_value="some_message_id") 
        mock_get_broker.return_value = mock_broker


        topic = "Test Topic"
        sources_folder = "test_sources"
        voice_id = "voice_abc"

        monkeypatch.setattr(appconfig.storyboard, "ENABLE_STORYBOARD_GENERATION", True)
        monkeypatch.setattr(appconfig, "ENABLE_IMAGE_GENERATION", True)
        monkeypatch.setattr(appconfig, "ENABLE_AUDIO_GENERATION", True)
        
        with patch('viralStoryGenerator.src.api_handlers.inspect.isawaitable') as mock_isawaitable, \
             patch('asyncio.get_event_loop') as mock_loop_getter:
            
            def isawaitable_side_effect(obj):
                if obj == mock_broker: 
                    return True
                if obj == "some_message_id": 
                     return True 
                return False
            mock_isawaitable.side_effect = isawaitable_side_effect

            def run_until_complete_side_effect(awaitable_obj):
                if awaitable_obj == mock_broker:
                    return mock_broker
                if awaitable_obj == "some_message_id":
                    return "some_message_id"
                if asyncio.iscoroutine(awaitable_obj):
                    async def runner(): return await awaitable_obj
                    # Use a new loop for this specific execution if needed, or ensure existing loop is running
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError: # No running loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return loop.run_until_complete(runner())
                return awaitable_obj 
            mock_loop_getter.return_value.run_until_complete = MagicMock(side_effect=run_until_complete_side_effect)

            response = await asyncio.to_thread(
                api_handlers.create_story_task, 
                topic, 
                sources_folder=sources_folder, 
                voice_id=voice_id
            )

        assert response["task_id"] == mock_task_id
        assert response["topic"] == topic
        assert response["status"] == "queued"

        mock_get_broker.assert_called_once() 
        mock_broker.publish_message.assert_called_once() 
        call_args = mock_broker.publish_message.call_args[0][0]
        
        assert call_args["job_id"] == mock_task_id
        assert call_args["job_type"] == "generate_story"
        assert call_args["topic"] == topic
        assert call_args["sources_folder"] == sources_folder
        assert call_args["voice_id"] == voice_id
        assert call_args["include_storyboard"] is True 
        assert "request_time" in call_args

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.uuid.uuid4')
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_create_story_task_config_toggles(self, mock_get_broker, mock_uuid, monkeypatch):
        mock_task_id = "test-uuid-config"
        mock_uuid.return_value = mock_task_id
        
        mock_broker = AsyncMock(spec=RedisMessageBroker)
        mock_broker.publish_message = AsyncMock(return_value="some_message_id")
        mock_get_broker.return_value = mock_broker

        topic = "Config Toggle Topic"

        test_cases = [
            {"sb_gen": False, "img_gen": True, "audio_gen": True, "expected_include_sb": False},
            {"sb_gen": True, "img_gen": False, "audio_gen": True, "expected_include_sb": False},
            {"sb_gen": True, "img_gen": True, "audio_gen": True, "expected_include_sb": True},
        ]

        for case in test_cases:
            monkeypatch.setattr(appconfig.storyboard, "ENABLE_STORYBOARD_GENERATION", case["sb_gen"])
            monkeypatch.setattr(appconfig, "ENABLE_IMAGE_GENERATION", case["img_gen"])
            monkeypatch.setattr(appconfig, "ENABLE_AUDIO_GENERATION", case["audio_gen"])

            with patch('viralStoryGenerator.src.api_handlers.inspect.isawaitable', return_value=True), \
                 patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_until_complete = MagicMock(side_effect=lambda coro_or_obj: coro_or_obj if not asyncio.iscoroutine(coro_or_obj) else mock_broker)


                await asyncio.to_thread(api_handlers.create_story_task, topic)
            
            call_args = mock_broker.publish_message.call_args[0][0]
            assert call_args["include_storyboard"] == case["expected_include_sb"], f"Failed for case: {case}"
            mock_broker.publish_message.reset_mock()

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.uuid.uuid4')
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_create_story_task_publish_failure(self, mock_get_broker, mock_uuid, monkeypatch):
        mock_task_id = "test-uuid-fail"
        mock_uuid.return_value = mock_task_id
        
        mock_broker = AsyncMock(spec=RedisMessageBroker)
        mock_broker.publish_message = AsyncMock(return_value=None) 
        mock_get_broker.return_value = mock_broker
        
        topic = "Failure Topic"

        monkeypatch.setattr(appconfig.storyboard, "ENABLE_STORYBOARD_GENERATION", True)
        monkeypatch.setattr(appconfig, "ENABLE_IMAGE_GENERATION", True)

        with patch('viralStoryGenerator.src.api_handlers.inspect.isawaitable', return_value=True), \
             patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete = MagicMock(side_effect=lambda coro_or_obj: coro_or_obj if not asyncio.iscoroutine(coro_or_obj) else mock_broker)


            with pytest.raises(RuntimeError, match="Failed to add task to Redis Stream."):
                await asyncio.to_thread(api_handlers.create_story_task, topic)
        
        mock_broker.publish_message.assert_called_once()


    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.uuid.uuid4')
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_create_story_task_optional_params(self, mock_get_broker, mock_uuid, monkeypatch):
        mock_task_id = "test-uuid-optional"
        mock_uuid.return_value = mock_task_id
        
        mock_broker = AsyncMock(spec=RedisMessageBroker)
        mock_broker.publish_message = AsyncMock(return_value="some_message_id")
        mock_get_broker.return_value = mock_broker
        
        topic = "Optional Params Topic"

        monkeypatch.setattr(appconfig.storyboard, "ENABLE_STORYBOARD_GENERATION", True)
        monkeypatch.setattr(appconfig, "ENABLE_IMAGE_GENERATION", True)
        
        with patch('viralStoryGenerator.src.api_handlers.inspect.isawaitable', return_value=True), \
             patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete = MagicMock(side_effect=lambda coro_or_obj: coro_or_obj if not asyncio.iscoroutine(coro_or_obj) else mock_broker)

        
            response = await asyncio.to_thread(api_handlers.create_story_task, topic) 

        assert response["task_id"] == mock_task_id
        assert response["topic"] == topic
        assert response["status"] == "queued"

        mock_broker.publish_message.assert_called_once()
        call_args = mock_broker.publish_message.call_args[0][0]
        
        assert call_args["job_id"] == mock_task_id
        assert call_args["topic"] == topic
        assert call_args["sources_folder"] is None
        assert call_args["voice_id"] is None
        assert call_args["include_storyboard"] is True
        assert "request_time" in call_args

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.uuid.uuid4')
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_create_story_task_publish_message_is_awaitable(self, mock_get_broker, mock_uuid, monkeypatch):
        mock_task_id = "test-uuid-awaitable-publish"
        mock_uuid.return_value = mock_task_id
        
        mock_broker = AsyncMock(spec=RedisMessageBroker)
        awaited_message_id = "awaited_message_id_val"
        mock_publish_result_awaitable = AsyncMock(return_value=awaited_message_id)
        mock_broker.publish_message = MagicMock(return_value=mock_publish_result_awaitable)

        mock_get_broker.return_value = mock_broker

        topic = "Awaitable Publish Topic"
        monkeypatch.setattr(appconfig.storyboard, "ENABLE_STORYBOARD_GENERATION", True)
        monkeypatch.setattr(appconfig, "ENABLE_IMAGE_GENERATION", True)

        with patch('viralStoryGenerator.src.api_handlers.inspect.isawaitable') as mock_isawaitable, \
             patch('asyncio.get_event_loop') as mock_loop_getter:
            
            def isawaitable_side_effect(obj):
                if obj == mock_broker: return True 
                if obj == mock_publish_result_awaitable: return True 
                return False
            mock_isawaitable.side_effect = isawaitable_side_effect

            def run_until_complete_side_effect(awaitable_obj):
                if awaitable_obj == mock_broker: return mock_broker
                if awaitable_obj == mock_publish_result_awaitable: return awaited_message_id 
                if asyncio.iscoroutine(awaitable_obj):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError: 
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return loop.run_until_complete(awaitable_obj) # Actually run it
                return awaitable_obj
            mock_loop_getter.return_value.run_until_complete = MagicMock(side_effect=run_until_complete_side_effect)

            response = await asyncio.to_thread(api_handlers.create_story_task, topic)

        assert response["status"] == "queued"
        mock_broker.publish_message.assert_called_once()
        assert any(call[0][0] == mock_publish_result_awaitable for call in mock_loop_getter.return_value.run_until_complete.call_args_list)


    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.uuid.uuid4')
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_create_story_task_publish_exception(self, mock_get_broker, mock_uuid, monkeypatch):
        mock_task_id = "test-uuid-exception"
        mock_uuid.return_value = mock_task_id
        
        mock_broker = AsyncMock(spec=RedisMessageBroker)
        mock_broker.publish_message = AsyncMock(side_effect=RuntimeError("Redis unavailable"))
        mock_get_broker.return_value = mock_broker
        
        topic = "Exception Topic"

        monkeypatch.setattr(appconfig.storyboard, "ENABLE_STORYBOARD_GENERATION", True)
        monkeypatch.setattr(appconfig, "ENABLE_IMAGE_GENERATION", True)

        with patch('viralStoryGenerator.src.api_handlers.inspect.isawaitable', return_value=True), \
             patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete = MagicMock(side_effect=lambda coro_or_obj: coro_or_obj if not asyncio.iscoroutine(coro_or_obj) else mock_broker)

            with pytest.raises(RuntimeError, match="Failed to add task to Redis Stream."):
                 await asyncio.to_thread(api_handlers.create_story_task, topic)
        
        mock_broker.publish_message.assert_called_once()

    # --- Tests for get_task_status ---

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_found_in_redis_processing(self, mock_storage_manager, mock_get_broker):
        task_id = "task_processing_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        redis_status = {'status': 'processing', 'job_id': task_id, 'message': 'In progress', 'timestamp': time.time()}
        mock_broker_instance.get_job_progress = AsyncMock(return_value=redis_status)
        mock_get_broker.return_value = mock_broker_instance

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response) 
        assert response['job_id'] == task_id
        assert response['status'] == 'processing'
        assert response['message'] == 'Job status from Redis Stream (processing)'
        mock_storage_manager.retrieve_file_content_as_json.assert_not_called()

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_found_in_redis_failed(self, mock_storage_manager, mock_get_broker):
        task_id = "task_failed_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        redis_status = {'status': 'failed', 'job_id': task_id, 'error': 'Something broke', 'timestamp': time.time()}
        mock_broker_instance.get_job_progress = AsyncMock(return_value=redis_status)
        mock_get_broker.return_value = mock_broker_instance

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response)
        assert response['job_id'] == task_id
        assert response['status'] == 'failed'
        assert response['error'] == 'Something broke'
        mock_storage_manager.retrieve_file_content_as_json.assert_not_called()

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_redis_completed_storage_lookup_success(self, mock_storage_manager, mock_get_broker):
        task_id = "task_completed_storage_success_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        redis_status = {'status': 'completed', 'job_id': task_id, 'message': 'From Redis', 'timestamp': time.time()}
        mock_broker_instance.get_job_progress = AsyncMock(return_value=redis_status)
        mock_get_broker.return_value = mock_broker_instance

        storage_data = {'job_id': task_id, 'status': 'completed', 'story_script': 'Final script from storage', 'audio_url': 'http://storage.com/audio.mp3', 'message': 'From Storage'}
        mock_storage_manager.retrieve_file_content_as_json = MagicMock(return_value=storage_data)

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response)
        assert response['job_id'] == task_id
        assert response['status'] == 'completed'
        assert response['story_script'] == 'Final script from storage'
        assert response['audio_url'] == 'http://storage.com/audio.mp3'
        assert response['message'] == 'Job completed and final results retrieved from storage.' 
        mock_storage_manager.retrieve_file_content_as_json.assert_called_once_with(
            filename=f"{task_id}_metadata.json", file_type="metadata"
        )

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_redis_completed_storage_lookup_file_not_found(self, mock_storage_manager, mock_get_broker):
        task_id = "task_completed_storage_fnf_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        redis_status = {'status': 'completed', 'job_id': task_id, 'timestamp': time.time()}
        mock_broker_instance.get_job_progress = AsyncMock(return_value=redis_status)
        mock_get_broker.return_value = mock_broker_instance

        mock_storage_manager.retrieve_file_content_as_json = MagicMock(side_effect=FileNotFoundError("Metadata not found"))

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response)
        assert response['job_id'] == task_id
        assert response['status'] == 'completed' 
        assert "Completed task metadata file not found in storage" in response['error']
        assert "Error fetching final results" in response['message']
        mock_storage_manager.retrieve_file_content_as_json.assert_called_once_with(
            filename=f"{task_id}_metadata.json", file_type="metadata"
        )

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_redis_completed_storage_lookup_other_exception(self, mock_storage_manager, mock_get_broker):
        task_id = "task_completed_storage_ex_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        redis_status = {'status': 'completed', 'job_id': task_id, 'timestamp': time.time()}
        mock_broker_instance.get_job_progress = AsyncMock(return_value=redis_status)
        mock_get_broker.return_value = mock_broker_instance

        mock_storage_manager.retrieve_file_content_as_json = MagicMock(side_effect=Exception("Storage unavailable"))

        response = await api_handlers.get_task_status(task_id)
        
        assert response is not None
        JobStatusResponse(**response)
        assert response['job_id'] == task_id
        assert response['status'] == 'completed' 
        assert "Failed to fetch final results from storage: Storage unavailable" in response['error']
        assert "Error fetching final results: Storage unavailable" in response['message']


    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_not_in_redis_found_in_storage(self, mock_storage_manager, mock_get_broker):
        task_id = "task_notredis_storage_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        mock_broker_instance.get_job_progress = AsyncMock(return_value=None) 
        mock_get_broker.return_value = mock_broker_instance

        storage_data = {'job_id': task_id, 'status': 'completed', 'story_script': 'From storage only', 'message': 'Retrieved from storage'}
        mock_storage_manager.retrieve_file_content_as_json = MagicMock(return_value=storage_data)

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response)
        assert response['job_id'] == task_id
        assert response['status'] == 'completed'
        assert response['story_script'] == 'From storage only'
        assert response['message'] == 'Job status retrieved from final storage (not found in Redis).'
        mock_storage_manager.retrieve_file_content_as_json.assert_called_once_with(
            filename=f"{task_id}_metadata.json", file_type="metadata"
        )

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_not_found_anywhere(self, mock_storage_manager, mock_get_broker):
        task_id = "task_notfound_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        mock_broker_instance.get_job_progress = AsyncMock(return_value=None) 
        mock_get_broker.return_value = mock_broker_instance

        mock_storage_manager.retrieve_file_content_as_json = MagicMock(return_value=None) 

        response = await api_handlers.get_task_status(task_id)

        assert response is None
        mock_storage_manager.retrieve_file_content_as_json.assert_called_once_with(
            filename=f"{task_id}_metadata.json", file_type="metadata"
        )
    
    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_not_found_anywhere_storage_raises_fnf(self, mock_storage_manager, mock_get_broker):
        task_id = "task_notfound_fnf_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        mock_broker_instance.get_job_progress = AsyncMock(return_value=None) 
        mock_get_broker.return_value = mock_broker_instance

        mock_storage_manager.retrieve_file_content_as_json = MagicMock(side_effect=FileNotFoundError) 

        response = await api_handlers.get_task_status(task_id)

        assert response is None 
        mock_storage_manager.retrieve_file_content_as_json.assert_called_once_with(
            filename=f"{task_id}_metadata.json", file_type="metadata"
        )


    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    async def test_get_task_status_broker_exception(self, mock_get_broker):
        task_id = "task_broker_ex_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        mock_broker_instance.get_job_progress = AsyncMock(side_effect=Exception("Redis connection error"))
        mock_get_broker.return_value = mock_broker_instance

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response) 
        assert response['job_id'] == task_id
        assert response['status'] == 'error'
        assert "Failed to retrieve task status: Redis connection error" in response['error']
        assert "Failed to retrieve task status: Redis connection error" in response['message']

    @pytest.mark.asyncio
    @patch('viralStoryGenerator.src.api_handlers.get_message_broker')
    @patch('viralStoryGenerator.src.api_handlers.storage_manager')
    async def test_get_task_status_storage_exception_after_redis_miss(self, mock_storage_manager, mock_get_broker):
        task_id = "task_storage_ex_after_redis_miss_123"
        mock_broker_instance = AsyncMock(spec=RedisMessageBroker)
        mock_broker_instance.get_job_progress = AsyncMock(return_value=None) 
        mock_get_broker.return_value = mock_broker_instance

        mock_storage_manager.retrieve_file_content_as_json = MagicMock(side_effect=Exception("Storage general error"))

        response = await api_handlers.get_task_status(task_id)

        assert response is not None
        JobStatusResponse(**response)
        assert response['job_id'] == task_id
        assert response['status'] == 'error'
        assert "Error checking storage for job status: Storage general error" in response['error']
        assert "Error checking storage for job status: Storage general error" in response['message']
        mock_storage_manager.retrieve_file_content_as_json.assert_called_once_with(
            filename=f"{task_id}_metadata.json", file_type="metadata"
        )

    # --- Tests for process_audio_queue ---

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio') # Patching the imported name
    @patch('viralStoryGenerator.src.api_handlers._logger') # Patching the logger
    def test_process_audio_queue_empty_directory(self, mock_logger, mock_gen_audio, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue"
        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")
        
        mock_isdir.return_value = True
        mock_listdir.return_value = []

        api_handlers.process_audio_queue()

        mock_isdir.assert_called_once_with(mock_audio_queue_dir)
        mock_listdir.assert_called_once_with(mock_audio_queue_dir)
        mock_gen_audio.assert_not_called()
        # Check for a debug log if the queue is empty (this depends on actual log messages in the function)
        # For now, just ensure it runs without error and doesn't process anything.

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    def test_process_audio_queue_no_json_files(self, mock_gen_audio, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue"
        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.txt", "file2.mp3"]

        api_handlers.process_audio_queue()
        mock_gen_audio.assert_not_called()

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.os.path.join')
    @patch('builtins.open', new_callable=mock_open) # Mock open globally for this test
    @patch('viralStoryGenerator.src.api_handlers.json.load')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    @patch('viralStoryGenerator.src.api_handlers.os.remove')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_success_one_file(self, mock_logger, mock_os_remove, mock_gen_audio, mock_json_load, mock_file_open, mock_os_path_join, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue"
        json_filename = "job1.json"
        full_json_path = os.path.join(mock_audio_queue_dir, json_filename) # Use actual os.path.join for mock setup

        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = True
        mock_listdir.return_value = [json_filename]
        mock_os_path_join.return_value = full_json_path # Ensure join returns the expected full path

        mock_json_data = {'story': 'Test story', 'mp3_file_path': '/path/to/audio.mp3', 'voice_id': 'voice1'}
        mock_json_load.return_value = mock_json_data
        
        mock_gen_audio.return_value = True # Success

        api_handlers.process_audio_queue()

        mock_file_open.assert_called_once_with(full_json_path, "r", encoding="utf-8")
        mock_json_load.assert_called_once()
        mock_gen_audio.assert_called_once_with(
            text=mock_json_data['story'],
            api_key="fake_key",
            output_mp3_path=mock_json_data['mp3_file_path'],
            voice_id=mock_json_data['voice_id'],
            model_id="eleven_multilingual_v2", # Default from function
            stability=0.5, # Default
            similarity_boost=0.75 # Default
        )
        mock_os_remove.assert_called_once_with(full_json_path)

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    @patch('viralStoryGenerator.src.api_handlers.json.load')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    @patch('viralStoryGenerator.src.api_handlers.os.remove')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_generate_audio_fails(self, mock_logger, mock_os_remove, mock_gen_audio, mock_json_load, mock_file_open, mock_os_path_join, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue_fail"
        json_filename = "job_fail.json"
        full_json_path = os.path.join(mock_audio_queue_dir, json_filename)

        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = True
        mock_listdir.return_value = [json_filename]
        mock_os_path_join.return_value = full_json_path
        mock_json_data = {'story': 'Test story fail', 'mp3_file_path': '/path/fail.mp3'}
        mock_json_load.return_value = mock_json_data
        mock_gen_audio.return_value = False # Failure

        api_handlers.process_audio_queue()

        mock_gen_audio.assert_called_once()
        mock_os_remove.assert_not_called()
        mock_logger.warning.assert_any_call(f"Queued audio generation attempt failed for {mock_json_data['mp3_file_path']}. Will retry later.")


    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    @patch('viralStoryGenerator.src.api_handlers.json.load')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    @patch('viralStoryGenerator.src.api_handlers.os.remove')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_invalid_json_file(self, mock_logger, mock_os_remove, mock_gen_audio, mock_json_load, mock_file_open, mock_os_path_join, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue_invalid"
        json_filename = "job_invalid.json"
        full_json_path = os.path.join(mock_audio_queue_dir, json_filename)

        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = True
        mock_listdir.return_value = [json_filename]
        mock_os_path_join.return_value = full_json_path
        mock_json_load.side_effect = json.JSONDecodeError("Simulated error", "doc", 0)

        api_handlers.process_audio_queue()

        mock_file_open.assert_called_once_with(full_json_path, "r", encoding="utf-8")
        mock_gen_audio.assert_not_called()
        mock_os_remove.assert_not_called()
        mock_logger.error.assert_any_call(f"Error reading queued file {full_json_path}: Invalid JSON.")

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    @patch('viralStoryGenerator.src.api_handlers.json.load')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    @patch('viralStoryGenerator.src.api_handlers.os.remove')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_missing_keys_in_json(self, mock_logger, mock_os_remove, mock_gen_audio, mock_json_load, mock_file_open, mock_os_path_join, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue_mkeys"
        json_filename = "job_mkeys.json"
        full_json_path = os.path.join(mock_audio_queue_dir, json_filename)
        
        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = True
        mock_listdir.return_value = [json_filename]
        mock_os_path_join.return_value = full_json_path
        mock_json_data = {'story': 'Test story only'} # Missing mp3_file_path
        mock_json_load.return_value = mock_json_data

        api_handlers.process_audio_queue()

        mock_gen_audio.assert_not_called()
        mock_os_remove.assert_not_called()
        mock_logger.warning.assert_any_call(f"Skipping invalid queue file {json_filename}: missing required keys.")


    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_no_api_key(self, mock_logger, mock_gen_audio, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue_no_key"
        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", None) # No API Key

        mock_isdir.return_value = True
        # Let listdir return something to ensure the API key check is the blocker
        mock_listdir.return_value = ["job_some.json"] 

        api_handlers.process_audio_queue()

        mock_gen_audio.assert_not_called()
        mock_logger.error.assert_any_call("Cannot process audio queue: ElevenLabs API Key not configured.")

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers.os.listdir')
    @patch('viralStoryGenerator.src.api_handlers.os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    @patch('viralStoryGenerator.src.api_handlers.json.load')
    @patch('viralStoryGenerator.src.api_handlers.generate_elevenlabs_audio')
    @patch('viralStoryGenerator.src.api_handlers.os.remove')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_os_remove_fails(self, mock_logger, mock_os_remove, mock_gen_audio, mock_json_load, mock_file_open, mock_os_path_join, mock_listdir, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/test/audio_queue_rm_fail"
        json_filename = "job_rm_fail.json"
        full_json_path = os.path.join(mock_audio_queue_dir, json_filename)

        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = True
        mock_listdir.return_value = [json_filename]
        mock_os_path_join.return_value = full_json_path
        mock_json_data = {'story': 'Test story rm fail', 'mp3_file_path': '/path/rm_fail.mp3'}
        mock_json_load.return_value = mock_json_data
        mock_gen_audio.return_value = True # Success
        mock_os_remove.side_effect = OSError("Permission denied")

        api_handlers.process_audio_queue()

        mock_gen_audio.assert_called_once()
        mock_os_remove.assert_called_once_with(full_json_path)
        mock_logger.error.assert_any_call(f"Failed to remove queue file {full_json_path}: Permission denied")

    @patch('viralStoryGenerator.src.api_handlers.os.path.isdir')
    @patch('viralStoryGenerator.src.api_handlers._logger')
    def test_process_audio_queue_dir_not_found(self, mock_logger, mock_isdir, monkeypatch):
        mock_audio_queue_dir = "/non_existent_dir"
        monkeypatch.setattr(appconfig, "AUDIO_QUEUE_DIR", mock_audio_queue_dir)
        monkeypatch.setattr(appconfig.elevenLabs, "API_KEY", "fake_key")

        mock_isdir.return_value = False # Directory does not exist

        api_handlers.process_audio_queue()

        mock_isdir.assert_called_once_with(mock_audio_queue_dir)
        mock_logger.debug.assert_any_call("Audio queue directory not found, skipping processing.")


# To make `asyncio.to_thread` available in Python 3.8 tests if needed, though environment is 3.10
if not hasattr(asyncio, 'to_thread'):
    async def to_thread_polyfill(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    asyncio.to_thread = to_thread_polyfill
