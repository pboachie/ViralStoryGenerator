import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError, AnyHttpUrl
from typing import List, Dict, Any

from viralStoryGenerator.models.models import (
    URLMetadata,
    StoryGenerationRequest,
    QueueConsumerDetail,
    QueueConsumerGroup,
    QueueRecentMessage,
    QueueStreamStatus,
    AllQueueStatusResponse,
    QueueStatusResponse,
    SingleQueueStatusResponse
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
        # Test serialization with alias
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
        Tests that URL validation passes for any domain if ALLOWED_DOMAINS is not configured,
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

        # Ensure hasattr behaves as expected with this mock
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
            "stream_length": 50,
            "consumer_groups": [],
            "recent_messages": []
        }
        # AllQueueStatusResponse is a RootModel, expects a dict at the root
        response_data = {"queue1": q_status_data, "queue2": q_status_data}
        response = AllQueueStatusResponse.model_validate(response_data)
        self.assertIn("queue1", response.root)
        self.assertEqual(response.root["queue1"].stream_length, 50)

    # test to validate urls
    def test_validate_urls(self):
        max_urls = 10
        # Test maximum URLs
        urls = [f"http://example.com/{i}" for i in range(max_urls + 1)]
        try:
            StoryGenerationRequest(urls=urls, topic="Too Many URLs")
        except ValidationError:
            pass
        else:
            self.fail("ValidationError not raised for too many URLs")

    def test_queue_status_response_creation(self):
        # Reusing QueueStreamStatus components for brevity
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
        response = SingleQueueStatusResponse(
            stream_name="my-stream",
            stream_length=10,
            consumer_groups=[{"name": "group1", "pending": 2, "consumers": 1}], # Simplified for this test
            recent_messages=[QueueRecentMessage(**recent_message_data)]
        )
        self.assertEqual(response.status, "available") # Default value
        self.assertEqual(response.stream_name, "my-stream")
        self.assertEqual(response.stream_length, 10)
        self.assertEqual(len(response.recent_messages), 1)
        self.assertEqual(response.recent_messages[0].id, "msg1")

if __name__ == '__main__':
    unittest.main()
