#!/usr/bin/env python
# viralStoryGenerator/models/models.py

from typing import Dict, Any, Optional, List, Union, Tuple
from pydantic import BaseModel, AnyHttpUrl, Field, HttpUrl, ValidationError, field_validator, RootModel
from viralStoryGenerator.utils.config import config as app_config


class URLMetadata(BaseModel):
    url: Union[AnyHttpUrl, str]
    final_url: Optional[Union[AnyHttpUrl, str]] = None
    status: str = "pending"  # e.g., pending, success, error
    markdown_content: Optional[str] = None
    html_content: Optional[str] = None
    raw_html_snippet: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    metadata_payload: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")
    error: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    screenshot_path: Optional[str] = None
    screenshot_url: Optional[AnyHttpUrl] = None
    screenshot_base64: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_url: Optional[AnyHttpUrl] = None
    mhtml_path: Optional[str] = None
    mhtml_url: Optional[AnyHttpUrl] = None
    job_id: Optional[str] = None
    dispatch_task_id: Optional[str] = None
    dispatch_memory_usage: Optional[Any] = None
    dispatch_duration_seconds: Optional[float] = None
    all_meta_tags: Optional[Dict[str, str]] = None
    image_url: Optional[AnyHttpUrl] = None
    language: Optional[str] = None
    favicon_url: Optional[AnyHttpUrl] = None

    class Config:
        populate_by_name = True

class StoryGenerationRequest(BaseModel):
    """Request model for generating a story from URLs"""
    urls: List[AnyHttpUrl] = Field(..., description="List of URLs to parse for content")
    topic: str = Field(..., description="Topic for the story", min_length=1, max_length=500)
    generate_audio: bool = Field(False, description="Whether to generate audio (overall toggle)")
    include_images: bool = Field(False, description="Whether to include images (overall toggle)")
    temperature: Optional[float] = Field(None, description="LLM temperature", ge=0.0, le=1.0)
    chunk_size: Optional[int] = Field(None, description="Word chunk size for splitting sources", ge=50, le=5000)
    voice_id: Optional[str] = Field(None, description="Voice ID for audio generation, if specific voice is requested.")
    include_storyboard: Optional[bool] = Field(None, description="Whether to generate a storyboard. Overrides global config if set.")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for the LLM.")
    output_format: Optional[str] = Field(None, description="Desired output format (e.g., 'standard', 'script_only').")

    class Config:
        validate_assignment = True

    @field_validator('topic')
    def validate_topic(cls, v):
        if any(char in v for char in ['&', '|', ';', '$', '`', '\\']):
            raise ValueError("Topic contains invalid characters")
        return v

    @field_validator('urls')
    def validate_urls(cls, v):
        if len(v) > 10: #todo: make this configurable
            raise ValueError("Maximum 10 URLs allowed")

        allowed_domains = app_config.http.ALLOWED_DOMAINS if hasattr(app_config.http, 'ALLOWED_DOMAINS') else []
        if allowed_domains:
            for url in v:
                domain = url.host
                if domain not in allowed_domains:
                    raise ValueError(f"URL domain not allowed: {domain}")
        return v

class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str = Field(..., description="Job ID for tracking progress")
    message: str = Field(..., description="Status message")


class StoryGenerationResult(BaseModel):
    """Response model for a completed story generation job"""
    story_script: str = Field(..., description="Generated story script")
    storyboard: Dict[str, Any] = Field(..., description="Generated storyboard")
    audio_url: Optional[str] = Field(None, description="URL to the generated audio file")
    sources: List[str] = Field(..., description="Sources used to generate the story")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")


class ServiceStatusDetail(BaseModel):
    """Model for individual service status details"""
    status: str = Field(..., description="Service status: healthy, degraded, or unhealthy")
    uptime: Union[float, str] = Field(..., description="Service uptime in seconds or uptime text")
    response_time: Union[float, str] = Field(..., description="Service response time in ms or response time text")
    message: Optional[str] = Field(None, description="Additional service status information")


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Overall system status: healthy, degraded, or unhealthy")
    services: Dict[str, ServiceStatusDetail] = Field(..., description="Status of individual services")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    uptime: Union[float, str] = Field(..., description="Server uptime in seconds or uptime text")


class ContentDetailItem(BaseModel):
    """Model for content URL and its fetched content."""
    url: Optional[str] = Field(None, description="URL to the content")
    content: Optional[Any] = Field(None, description="Fetched content from the URL (can be string or dict for JSON)")

class JobStatusResponse(BaseModel):
    """Response model for job status checks"""
    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Job status: pending, processing, completed, or failed")
    message: Optional[str] = Field(None, description="Status message")
    story: Optional[List[ContentDetailItem]] = Field(None, description="Details of the generated story script, including URL and content")
    storyboard: Optional[List[ContentDetailItem]] = Field(None, description="Details of the generated storyboard, including URL and content")
    audio_url: Optional[str] = Field(None, description="URL to the generated audio file if completed")
    sources: Optional[List[str]] = Field(None, description="Sources used to generate the story")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: Optional[str] = Field(None, description="Timestamp when job was created")
    updated_at: Optional[str] = Field(None, description="Timestamp when job was last updated")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time in seconds for the job")

class StoryboardScene(BaseModel):
    scene_number: int = Field(..., description="Scene number")
    narration_text: str = Field(..., description="Narration text")
    image_prompt: str = Field(..., description="Image prompt")
    duration: float = Field(..., description="Duration in seconds")
    start_time: float = Field(..., description="Start time in seconds")

class StoryboardResponse(BaseModel):
    scenes: List[StoryboardScene] = Field(..., description="List of scenes in storyboard")

class ScrapeJobRequest(BaseModel):
    job_id: str = Field(..., description="Unique job identifier for the scrape request")
    urls: List[str] = Field(..., description="List of URLs to scrape")
    browser_config: Optional[Dict[str, Any]] = Field(None, description="Optional browser configuration")
    run_config: Optional[Dict[str, Any]] = Field(None, description="Optional crawler run configuration")
    request_time: float = Field(..., description="Timestamp when the scrape request was created")

class ScrapeJobResult(BaseModel):
    status: str = Field(..., description="Job status: processing, completed, or failed")
    message: Optional[str] = Field(None, description="Update or status message")
    error: Optional[str] = Field(None, description="Error details if the job failed")
    data: Optional[List[Tuple[str, Optional[str]]]] = Field(None, description="List of scraped results (url and content)")
    created_at: float = Field(..., description="Timestamp when the job was created")
    updated_at: float = Field(..., description="Timestamp when the job was last updated")

class ClearStalledJobsResponse(BaseModel):
    message: str
    claimed_count: int
    failed_count: int
    reprocessed_count: int

class SuccessResponse(BaseModel):
    message: str
    detail: Optional[str] = None

class FailureResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class QueueConsumerDetail(BaseModel):
    name: str
    pending: Optional[int]
    idle: Optional[int]

class QueueConsumerGroup(BaseModel):
    group_name: str
    pending: Optional[int]
    consumers: int
    consumer_details: List[QueueConsumerDetail]
    last_delivered_id: Optional[str] = None

class QueueRecentMessage(BaseModel):
    id: str
    timestamp: str
    job_id: Optional[str]
    status: str
    is_system_message: Optional[bool] = False

class QueueStreamStatus(BaseModel):
    stream_length: int
    consumer_groups: List[QueueConsumerGroup]
    recent_messages: List[QueueRecentMessage]

class SingleQueueStatusResponse(BaseModel):
    stream_name: str
    status: str = "available"
    stream_length: int
    consumer_groups: List[QueueConsumerGroup]
    recent_messages: List[QueueRecentMessage]
    consumer_groups_count: Optional[int] = None
    recent_messages_count: Optional[int] = None
    error_message: Optional[str] = None

class AllQueueStatusResponse(RootModel[Dict[str, SingleQueueStatusResponse]]):
    pass

class QueueStatusResponse(BaseModel):
    status: str
    stream_length: int
    consumer_groups: List[QueueConsumerGroup]
    recent_messages: List[QueueRecentMessage]

STORYBOARD_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "storyboard_response",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_number": {"type": "integer"},
                            "scene_start_marker": {"type": "string", "description": "The exact first 5-10 words of the scene text."},
                            "image_prompt": {"type": "string"},
                            "duration": {"type": "number"},
                            "start_time": {"type": "number"}
                        },
                        "required": ["scene_number", "scene_start_marker", "image_prompt", "duration", "start_time"]
                    }
                }
            },
            "required": ["scenes"]
        }
    }
}
