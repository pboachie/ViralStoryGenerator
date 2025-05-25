import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError, AnyHttpUrl, HttpUrl
from typing import List, Dict, Any, Optional, Union, Tuple # Added HttpUrl, Optional, Union, Tuple

from viralStoryGenerator.models.models import (
    URLMetadata,
    StoryGenerationRequest,
    QueueConsumerDetail,
    QueueConsumerGroup,
    QueueRecentMessage,
    QueueStreamStatus,
    AllQueueStatusResponse,
    QueueStatusResponse,
    SingleQueueStatusResponse,
    JobResponse, # Added
    StoryGenerationResult, # Added
    ErrorResponse, # Added
    ServiceStatusDetail, # Added
    HealthResponse, # Added
    ContentDetailItem, # Added
    JobStatusResponse, # Added
    StoryboardScene, # Added
    StoryboardResponse, # Added
    ScrapeJobRequest, # Added
    ScrapeJobResult, # Added
    ClearStalledJobsResponse, # Added
    SuccessResponse, # Added
    FailureResponse # Added
)

class TestModels(unittest.TestCase):

    def test_url_metadata_creation(self):
        data = {
            "url": "http://example.com",
            "final_url": "http://example.com/final",
            "status": "success",
            "markdown_content": "markdown",
            "title": "Test Title",
            "metadata": {"key": "value"}
        }
        metadata = URLMetadata(**data)
        self.assertEqual(str(metadata.url), "http://example.com")

    def test_url_metadata_alias(self):
        metadata = URLMetadata(url="http://example.com/page", metadata={"custom_key": "custom_value"})
        self.assertEqual(metadata.metadata_payload, {"custom_key": "custom_value"})
        self.assertIn("metadata", metadata.model_dump(by_alias=True))
        self.assertNotIn("metadata_payload", metadata.model_dump(by_alias=True))
        self.assertIn("metadata_payload", metadata.model_dump(by_alias=False))


    def test_story_generation_request_creation(self):
        data = {
            "urls": ["http://example.com/story1"],
            "topic": "A great story",
            "include_images": True,
            "temperature": 0.7
        }
        req = StoryGenerationRequest(**data)
        self.assertEqual(req.topic, "A great story")
        self.assertTrue(req.include_images)
        self.assertEqual(len(req.urls), 1)
        self.assertEqual(str(req.urls[0]), "http://example.com/story1")

    def test_story_generation_request_invalid_topic(self):
        with self.assertRaises(ValidationError) as context:
            StoryGenerationRequest(urls=["http://example.com"], topic="A bad topic with ;")
        self.assertIn("Topic contains invalid characters", str(context.exception))

    def test_story_generation_request_too_many_urls(self):
        urls = [f"http://example.com/{i}" for i in range(11)]
        with self.assertRaises(ValidationError) as context:
            StoryGenerationRequest(urls=urls, topic="A valid topic")
        self.assertIn("Maximum 10 URLs allowed", str(context.exception))

    @patch('viralStoryGenerator.models.models.app_config')
    def test_story_generation_request_disallowed_domain(self, mock_app_config):
        mock_http = MagicMock()
        mock_http.ALLOWED_DOMAINS = ['example.org']
        mock_app_config.http = mock_http

        with self.assertRaises(ValidationError) as context:
            StoryGenerationRequest(urls=["http://example.com/disallowed"], topic="Test Topic")
        self.assertIn("URL domain not allowed: example.com", str(context.exception))

    @patch('viralStoryGenerator.models.models.app_config')
    def test_story_generation_request_allowed_domain(self, mock_app_config):
        mock_http = MagicMock()
        mock_http.ALLOWED_DOMAINS = ['example.com']
        mock_app_config.http = mock_http

        try:
            StoryGenerationRequest(urls=["http://example.com/allowed"], topic="Test Topic")
        except ValidationError:
            self.fail("ValidationError raised unexpectedly for allowed domain")

    @patch('viralStoryGenerator.models.models.app_config')
    def test_story_generation_request_urls_no_domain_restriction_applied(self, mock_app_config):
        """
        Tests that URL validation passes for any domain if ALLOWED_DOMAAINS is not configured,
        None, or an empty list.
        """
        urls_to_test = ["http://anydomain.com/path", "https://another.co.uk/resource"]
        topic = "Test Topic No Domain Restriction"

        # Scenario 1: app_config.http.ALLOWED_DOMAINS is None
        mock_http_config_none = MagicMock()
        mock_http_config_none.ALLOWED_DOMAINS = None
        mock_app_config.http = mock_http_config_none

        try:
            StoryGenerationRequest(urls=urls_to_test, topic=topic)
        except ValidationError as e:
            self.fail(f"ValidationError raised unexpectedly when ALLOWED_DOMAINS is None: {e}")

        # Scenario 2: app_config.http.ALLOWED_DOMAINS is an empty list
        mock_http_config_empty = MagicMock()
        mock_http_config_empty.ALLOWED_DOMAINS = []
        mock_app_config.http = mock_http_config_empty

        try:
            StoryGenerationRequest(urls=urls_to_test, topic=topic)
        except ValidationError as e:
            self.fail(f"ValidationError raised unexpectedly when ALLOWED_DOMAINS is an empty list: {e}")

        # Scenario 3: app_config.http does not have the ALLOWED_DOMAINS attribute
        # The model's code `... if hasattr(app_config.http, 'ALLOWED_DOMAINS') else None`
        # handles this by setting internal allowed_domains to None.
        mock_http_config_attr_missing = MagicMock(spec=[]) # spec=[] ensures no attributes unless added
        mock_app_config.http = mock_http_config_attr_missing

        self.assertFalse(hasattr(mock_http_config_attr_missing, 'ALLOWED_DOMAINS'))

        try:
            StoryGenerationRequest(urls=urls_to_test, topic=topic)
        except ValidationError as e:
            self.fail(f"ValidationError raised unexpectedly when ALLOWED_DOMAINS attribute is missing: {e}")

    def test_queue_consumer_detail_creation(self):
        detail = QueueConsumerDetail(name="consumer-1", pending=5, idle=1000)
        self.assertEqual(detail.name, "consumer-1")
        self.assertEqual(detail.pending, 5)

    def test_queue_consumer_group_creation(self):
        consumer_detail_data = {"name": "c1", "pending": 1, "idle": 100}
        group = QueueConsumerGroup(
            group_name="group-alpha",
            pending=10,
            consumers=1,
            consumer_details=[QueueConsumerDetail(**consumer_detail_data)]
        )
        self.assertEqual(group.group_name, "group-alpha")
        self.assertEqual(group.pending, 10)
        self.assertEqual(len(group.consumer_details), 1)
        self.assertEqual(group.consumer_details[0].name, "c1")

    def test_queue_recent_message_creation(self):
        msg = QueueRecentMessage(id="msg-123", timestamp="2023-01-01T12:00:00Z", job_id="job-abc", status="processed")
        self.assertEqual(msg.id, "msg-123")
        self.assertEqual(msg.status, "processed")

    def test_queue_stream_status_creation(self):
        consumer_detail = QueueConsumerDetail(name="c1", pending=1, idle=100)
        consumer_group = QueueConsumerGroup(
            group_name="cg1", pending=5, consumers=1, consumer_details=[consumer_detail]
        )
        recent_message = QueueRecentMessage(id="m1", timestamp="ts", job_id="j1", status="done")

        status = QueueStreamStatus(
            stream_length=100,
            consumer_groups=[consumer_group],
            recent_messages=[recent_message]
        )
        self.assertEqual(status.stream_length, 100)
        self.assertEqual(len(status.consumer_groups), 1)
        self.assertEqual(status.consumer_groups[0].group_name, "cg1")
        self.assertEqual(len(status.recent_messages), 1)
        self.assertEqual(status.recent_messages[0].id, "m1")

    def test_all_queue_status_response_creation(self):
        q_status_data = {
            "stream_name": "queue1",
            "stream_length": 50,
            "consumer_groups": [],
            "recent_messages": []
        }
        response_data = {
            "queue1": {**q_status_data, "stream_name": "queue1"},
            "queue2": {**q_status_data, "stream_name": "queue2"}
        }
        response = AllQueueStatusResponse.model_validate(response_data)
        self.assertIn("queue1", response.root)
        self.assertEqual(response.root["queue1"].stream_length, 50)

    def test_validate_urls(self):
        max_urls = 10
        urls = [f"http://example.com/{i}" for i in range(max_urls + 1)]
        try:
            StoryGenerationRequest(urls=urls, topic="Too Many URLs")
        except ValidationError:
            pass
        else:
            self.fail("ValidationError not raised for too many URLs")

    def test_queue_status_response_creation(self):
        consumer_detail = QueueConsumerDetail(name="c1", pending=1, idle=100)
        consumer_group = QueueConsumerGroup(
            group_name="cg1", pending=5, consumers=1, consumer_details=[consumer_detail]
        )
        recent_message = QueueRecentMessage(id="m1", timestamp="ts", job_id="j1", status="done")

        response = QueueStatusResponse(
            status="healthy",
            stream_length=20,
            consumer_groups=[consumer_group],
            recent_messages=[recent_message]
        )
        self.assertEqual(response.status, "healthy")
        self.assertEqual(response.stream_length, 20)
        self.assertTrue(len(response.consumer_groups) > 0)

    def test_single_queue_status_response_creation(self):
        recent_message_data = {"id": "msg1", "timestamp": "ts1", "job_id": "job1", "status": "pending"}
        consumer_group = {
            "group_name": "group1",
            "pending": 2,
            "consumers": 1,
            "consumer_details": []
        }
        response = SingleQueueStatusResponse(
            stream_name="my-stream",
            stream_length=10,
            consumer_groups=[consumer_group],
            recent_messages=[QueueRecentMessage(**recent_message_data)]
        )
        self.assertEqual(response.status, "available") # Default value
        self.assertEqual(response.stream_name, "my-stream")
        self.assertEqual(response.stream_length, 10)
        self.assertEqual(len(response.recent_messages), 1)
        self.assertEqual(response.recent_messages[0].id, "msg1")

    def test_single_queue_status_response_recent_message_id(self):
        """Test that SingleQueueStatusResponse recent_messages[0].id is set correctly."""
        recent_message_data = {"id": "test-id", "timestamp": "now", "job_id": "job42", "status": "pending"}
        response = SingleQueueStatusResponse(
            stream_name="test-stream",
            stream_length=1,
            consumer_groups=[],
            recent_messages=[QueueRecentMessage(**recent_message_data)]
        )
        self.assertEqual(response.recent_messages[0].id, "test-id")

    # --- Tests for JobResponse ---
    def test_job_response_creation(self):
        data = {"job_id": "job123", "message": "Job created successfully"}
        obj = JobResponse(**data)
        self.assertEqual(obj.job_id, "job123")
        self.assertEqual(obj.message, "Job created successfully")

    def test_job_response_missing_required_fields(self):
        with self.assertRaises(ValidationError):
            JobResponse(message="Job created successfully") # job_id missing
        with self.assertRaises(ValidationError):
            JobResponse(job_id="job123") # message missing

    def test_job_response_invalid_types(self):
        with self.assertRaises(ValidationError):
            JobResponse(job_id=123, message="Job created successfully") # job_id invalid type
        with self.assertRaises(ValidationError):
            JobResponse(job_id="job123", message=123) # message invalid type

    # --- Tests for StoryGenerationResult ---
    def test_story_generation_result_creation(self):
        data = {
            "story_script": "This is the story.",
            "storyboard": {"scene1": "details1"},
            "audio_url": "http://example.com/audio.mp3",
            "sources": ["http://example.com/source1"]
        }
        obj = StoryGenerationResult(**data)
        self.assertEqual(obj.story_script, "This is the story.")
        self.assertEqual(obj.storyboard, {"scene1": "details1"})
        self.assertEqual(obj.audio_url, "http://example.com/audio.mp3")
        self.assertEqual(obj.sources, ["http://example.com/source1"])

    def test_story_generation_result_missing_required_fields(self):
        required_fields = ["story_script", "storyboard", "sources"]
        base_data = {
            "story_script": "Script", "storyboard": {}, "sources": [], "audio_url": "http://example.com/audio.mp3"
        }
        for field in required_fields:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                StoryGenerationResult(**data)

    def test_story_generation_result_invalid_types(self):
        with self.assertRaises(ValidationError): # story_script invalid
            StoryGenerationResult(story_script=123, storyboard={}, sources=[])
        with self.assertRaises(ValidationError): # storyboard invalid
            StoryGenerationResult(story_script="Script", storyboard="not a dict", sources=[])
        with self.assertRaises(ValidationError): # audio_url invalid
            StoryGenerationResult(story_script="Script", storyboard={}, sources=[], audio_url=123)
        with self.assertRaises(ValidationError): # sources invalid
            StoryGenerationResult(story_script="Script", storyboard={}, sources="not a list")

    # --- Tests for ErrorResponse ---
    def test_error_response_creation(self):
        data = {"error": "NotFound", "detail": "Resource not found"}
        obj = ErrorResponse(**data)
        self.assertEqual(obj.error, "NotFound")
        self.assertEqual(obj.detail, "Resource not found")

    def test_error_response_missing_required_fields(self):
        with self.assertRaises(ValidationError):
            ErrorResponse(detail="Resource not found") # error missing
        with self.assertRaises(ValidationError):
            ErrorResponse(error="NotFound") # detail missing

    def test_error_response_invalid_types(self):
        with self.assertRaises(ValidationError):
            ErrorResponse(error=123, detail="Resource not found")
        with self.assertRaises(ValidationError):
            ErrorResponse(error="NotFound", detail=123)

    # --- Tests for ServiceStatusDetail ---
    def test_service_status_detail_creation(self):
        data = {"status": "healthy", "uptime": 12345.6, "response_time": 100.5, "message": "All systems nominal"}
        obj = ServiceStatusDetail(**data)
        self.assertEqual(obj.status, "healthy")
        self.assertEqual(obj.uptime, 12345.6)
        self.assertEqual(obj.response_time, 100.5)
        self.assertEqual(obj.message, "All systems nominal")
        data_str_uptime = {"status": "degraded", "uptime": "3 days", "response_time": "N/A", "message": "High load"}
        obj_str = ServiceStatusDetail(**data_str_uptime)
        self.assertEqual(obj_str.uptime, "3 days")


    def test_service_status_detail_missing_required_fields(self):
        required = ["status", "uptime", "response_time"]
        base_data = {"status": "healthy", "uptime": 1.0, "response_time": 1.0, "message": "OK"}
        for field in required:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                ServiceStatusDetail(**data)

    def test_service_status_detail_invalid_types(self):
        with self.assertRaises(ValidationError): # status invalid
            ServiceStatusDetail(status=123, uptime=1.0, response_time=1.0)
        # uptime/response_time can be str or float, so direct type errors are harder to trigger without specific constraints
        with self.assertRaises(ValidationError): # message invalid
            ServiceStatusDetail(status="ok", uptime=1.0, response_time=1.0, message=[])


    # --- Tests for HealthResponse ---
    def test_health_response_creation(self):
        service_detail_data = {"status": "healthy", "uptime": 10.0, "response_time": 1.0}
        data = {
            "status": "healthy",
            "services": {"service1": ServiceStatusDetail(**service_detail_data)},
            "version": "1.0.0",
            "environment": "production",
            "uptime": 12345.0
        }
        obj = HealthResponse(**data)
        self.assertEqual(obj.status, "healthy")
        self.assertEqual(obj.services["service1"].status, "healthy")
        self.assertEqual(obj.version, "1.0.0")

    def test_health_response_missing_required_fields(self):
        required = ["status", "services", "version", "environment", "uptime"]
        base_data = {
            "status": "healthy", "services": {}, "version": "1.0", "environment": "dev", "uptime": 1.0
        }
        for field in required:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                HealthResponse(**data)

    def test_health_response_invalid_types(self):
        service_detail_data = {"status": "healthy", "uptime": 10.0, "response_time": 1.0}
        base_args = {"services": {"s1": service_detail_data}, "version": "1", "environment": "dev", "uptime": 1.0}
        with self.assertRaises(ValidationError): # status invalid
            HealthResponse(status=123, **base_args)
        with self.assertRaises(ValidationError): # services invalid
            HealthResponse(status="ok", services="not a dict", version="1", environment="dev", uptime=1.0)

        base_args_no_version = base_args.copy()
        del base_args_no_version["version"]
        with self.assertRaises(ValidationError):
            HealthResponse(status="ok", **base_args_no_version, version="1.0")


    # --- Tests for ContentDetailItem ---
    def test_content_detail_item_creation(self):
        data = {"url": "http://example.com/content", "content": "This is the content."}
        obj = ContentDetailItem(**data)
        self.assertEqual(obj.url, "http://example.com/content")
        self.assertEqual(obj.content, "This is the content.")
        data_dict_content = {"url": "http://example.com/json", "content": {"key": "value"}}
        obj_dict = ContentDetailItem(**data_dict_content)
        self.assertEqual(obj_dict.content, {"key": "value"})


    def test_content_detail_item_invalid_types(self):
         # URL should be a string, content can be Any
        with self.assertRaises(ValidationError):
            ContentDetailItem(url=123, content="content")


    # --- Tests for JobStatusResponse ---
    def test_job_status_response_creation(self):
        content_item_data = {"url": "http://example.com/doc", "content": "doc content"}
        data = {
            "job_id": "job789",
            "status": "completed",
            "message": "Job finished.",
            "story": [ContentDetailItem(**content_item_data)],
            "storyboard": [ContentDetailItem(**content_item_data)],
            "audio_url": "http://example.com/audio.mp3",
            "sources": ["http://example.com/source"],
            "error": None,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T01:00:00Z",
            "processing_time_seconds": 60.5
        }
        obj = JobStatusResponse(**data)
        self.assertEqual(obj.job_id, "job789")
        self.assertEqual(obj.status, "completed")
        self.assertEqual(obj.story[0].url, "http://example.com/doc")

    def test_job_status_response_missing_required_fields(self):
        with self.assertRaises(ValidationError):
            JobStatusResponse(status="pending") # job_id missing
        with self.assertRaises(ValidationError):
            JobStatusResponse(job_id="job123") # status missing

    def test_job_status_response_invalid_types(self):
        base_data = {"job_id": "job1", "status": "done"}
        with self.assertRaises(ValidationError): # job_id invalid
            JobStatusResponse(job_id=123, status="pending")
        with self.assertRaises(ValidationError): # status invalid
            JobStatusResponse(job_id="job1", status=123)
        with self.assertRaises(ValidationError): # story invalid
            JobStatusResponse(**base_data, story="not a list")
        with self.assertRaises(ValidationError): # storyboard invalid
            JobStatusResponse(**base_data, storyboard="not a list")
        with self.assertRaises(ValidationError): # audio_url invalid
            JobStatusResponse(**base_data, audio_url=123)
        with self.assertRaises(ValidationError): # sources invalid
            JobStatusResponse(**base_data, sources="not a list")
        with self.assertRaises(ValidationError): # processing_time_seconds invalid
            JobStatusResponse(**base_data, processing_time_seconds="not a float")


    # --- Tests for StoryboardScene ---
    def test_storyboard_scene_creation(self):
        data = {
            "scene_number": 1,
            "narration_text": "The story begins...",
            "image_prompt": "A beautiful landscape.",
            "duration": 10.5,
            "start_time": 0.0
        }
        obj = StoryboardScene(**data)
        self.assertEqual(obj.scene_number, 1)
        self.assertEqual(obj.narration_text, "The story begins...")

    def test_storyboard_scene_missing_required_fields(self):
        required = ["scene_number", "narration_text", "image_prompt", "duration", "start_time"]
        base_data = {"scene_number": 1, "narration_text": "text", "image_prompt": "prompt", "duration": 1.0, "start_time": 0.0}
        for field in required:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                StoryboardScene(**data)

    def test_storyboard_scene_invalid_types(self):
        base_data = {"narration_text": "text", "image_prompt": "prompt", "duration": 1.0, "start_time": 0.0}
        with self.assertRaises(ValidationError): # scene_number invalid
            StoryboardScene(scene_number="not an int", **base_data)
        with self.assertRaises(ValidationError): # narration_text invalid
            StoryboardScene(scene_number=1, narration_text=123, image_prompt="p", duration=1.0, start_time=0.0)
        with self.assertRaises(ValidationError): # duration invalid
            StoryboardScene(scene_number=1, **base_data, duration="not a float", start_time=0.0)


    # --- Tests for StoryboardResponse ---
    def test_storyboard_response_creation(self):
        scene_data = {
            "scene_number": 1, "narration_text": "text", "image_prompt": "prompt", "duration": 5.0, "start_time": 0.0
        }
        data = {"scenes": [StoryboardScene(**scene_data)]}
        obj = StoryboardResponse(**data)
        self.assertEqual(len(obj.scenes), 1)
        self.assertEqual(obj.scenes[0].scene_number, 1)

    def test_storyboard_response_missing_required_fields(self):
        with self.assertRaises(ValidationError):
            StoryboardResponse() # scenes missing

    def test_storyboard_response_invalid_types(self):
        with self.assertRaises(ValidationError): # scenes not a list
            StoryboardResponse(scenes="not a list")
        with self.assertRaises(ValidationError): # scene item invalid
            StoryboardResponse(scenes=[{"invalid_scene_data": True}])


    # --- Tests for ScrapeJobRequest ---
    def test_scrape_job_request_creation(self):
        data = {
            "job_id": "scrape_job_1",
            "urls": ["http://example.com/scrape_this"],
            "browser_config": {"headless": True},
            "run_config": {"timeout": 30},
            "request_time": 1678886400.0
        }
        obj = ScrapeJobRequest(**data)
        self.assertEqual(obj.job_id, "scrape_job_1")
        self.assertEqual(obj.urls, ["http://example.com/scrape_this"])
        self.assertTrue(obj.browser_config["headless"])

    def test_scrape_job_request_missing_required_fields(self):
        required = ["job_id", "urls", "request_time"]
        base_data = {"job_id": "j1", "urls": ["http://example.com"], "request_time": 1.0}
        for field in required:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                ScrapeJobRequest(**data)

    def test_scrape_job_request_invalid_types(self):
        base_data = {"urls": ["http://example.com"], "request_time": 1.0}
        with self.assertRaises(ValidationError): # job_id invalid
            ScrapeJobRequest(job_id=123, **base_data)
        with self.assertRaises(ValidationError): # urls invalid (not list)
            ScrapeJobRequest(job_id="j1", urls="not a list", request_time=1.0)
        with self.assertRaises(ValidationError): # urls invalid (item not str)
            ScrapeJobRequest(job_id="j1", urls=[123], request_time=1.0)
        with self.assertRaises(ValidationError): # browser_config invalid
            ScrapeJobRequest(job_id="j1", **base_data, browser_config="not a dict")
        with self.assertRaises(ValidationError): # request_time invalid
            ScrapeJobRequest(job_id="j1", urls=["http://example.com"], request_time="not a float")

    # --- Tests for ScrapeJobResult ---
    def test_scrape_job_result_creation(self):
        data = {
            "status": "completed",
            "message": "Scraping done.",
            "error": None,
            "data": [("http://example.com", "content")],
            "created_at": 1678886400.0,
            "updated_at": 1678886460.0
        }
        obj = ScrapeJobResult(**data)
        self.assertEqual(obj.status, "completed")
        self.assertEqual(obj.data[0][0], "http://example.com")

    def test_scrape_job_result_missing_required_fields(self):
        required = ["status", "created_at", "updated_at"]
        base_data = {"status": "ok", "created_at": 1.0, "updated_at": 2.0}
        for field in required:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                ScrapeJobResult(**data)

    def test_scrape_job_result_invalid_types(self):
        base_data = {"created_at": 1.0, "updated_at": 2.0}
        with self.assertRaises(ValidationError): # status invalid
            ScrapeJobResult(status=123, **base_data)
        with self.assertRaises(ValidationError): # data invalid (not list)
            ScrapeJobResult(status="ok", **base_data, data="not a list")
        with self.assertRaises(ValidationError): # data invalid (tuple item not str)
            ScrapeJobResult(status="ok", **base_data, data=[(123, "content")])
        with self.assertRaises(ValidationError): # created_at invalid
            ScrapeJobResult(status="ok", created_at="not a float", updated_at=2.0)


    # --- Tests for ClearStalledJobsResponse ---
    def test_clear_stalled_jobs_response_creation(self):
        data = {"message": "Cleared jobs", "claimed_count": 5, "failed_count": 1, "reprocessed_count": 4}
        obj = ClearStalledJobsResponse(**data)
        self.assertEqual(obj.message, "Cleared jobs")
        self.assertEqual(obj.claimed_count, 5)

    def test_clear_stalled_jobs_response_missing_required_fields(self):
        required = ["message", "claimed_count", "failed_count", "reprocessed_count"]
        base_data = {"message":"msg", "claimed_count":0, "failed_count":0, "reprocessed_count":0}
        for field in required:
            data = base_data.copy()
            del data[field]
            with self.assertRaises(ValidationError, msg=f"Missing field: {field}"):
                ClearStalledJobsResponse(**data)

    def test_clear_stalled_jobs_response_invalid_types(self):
        base_data = {"claimed_count":0, "failed_count":0, "reprocessed_count":0}
        with self.assertRaises(ValidationError): # message invalid
            ClearStalledJobsResponse(message=123, **base_data)
        with self.assertRaises(ValidationError): # claimed_count invalid
            ClearStalledJobsResponse(message="msg", claimed_count="not int", failed_count=0, reprocessed_count=0)

    # --- Tests for SuccessResponse ---
    def test_success_response_creation(self):
        data = {"message": "Operation successful", "detail": "Extra details here"}
        obj = SuccessResponse(**data)
        self.assertEqual(obj.message, "Operation successful")
        self.assertEqual(obj.detail, "Extra details here")

    def test_success_response_missing_required_fields(self):
        with self.assertRaises(ValidationError): # message missing
            SuccessResponse(detail="details")

    def test_success_response_invalid_types(self):
        with self.assertRaises(ValidationError): # message invalid
            SuccessResponse(message=123)
        with self.assertRaises(ValidationError): # detail invalid
            SuccessResponse(message="ok", detail=123)

    # --- Tests for FailureResponse ---
    def test_failure_response_creation(self):
        data = {"error": "Operation failed", "detail": "Reason for failure"}
        obj = FailureResponse(**data)
        self.assertEqual(obj.error, "Operation failed")
        self.assertEqual(obj.detail, "Reason for failure")

    def test_failure_response_missing_required_fields(self):
        with self.assertRaises(ValidationError): # error missing
            FailureResponse(detail="details")

    def test_failure_response_invalid_types(self):
        with self.assertRaises(ValidationError): # error invalid
            FailureResponse(error=123)
        with self.assertRaises(ValidationError): # detail invalid
            FailureResponse(error="fail", detail=123)


if __name__ == '__main__':
    unittest.main()
