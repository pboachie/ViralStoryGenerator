#!/usr/bin/env python
# viralStoryGenerator/models/models.py

from typing import Dict, Any, Optional, List, Union, Tuple
from pydantic import BaseModel, AnyHttpUrl, Field, field_validator
from viralStoryGenerator.utils.config import config as app_config

class StoryGenerationRequest(BaseModel):
    """Request model for generating a story from URLs"""
    urls: List[AnyHttpUrl] = Field(..., description="List of URLs to parse for content", max_items=10)
    topic: str = Field(..., description="Topic for the story", min_length=1, max_length=500)
    generate_audio: bool = Field(False, description="Whether to generate audio")
    temperature: Optional[float] = Field(None, description="LLM temperature", ge=0.0, le=1.0)
    chunk_size: Optional[int] = Field(None, description="Word chunk size for splitting sources", ge=50, le=5000)

    class Config:
        validate_assignment = True

    @field_validator('topic')
    def validate_topic(cls, v):
        # Check for potentially dangerous characters
        if any(char in v for char in ['&', '|', ';', '$', '`', '\\']):
            raise ValueError("Topic contains invalid characters")
        return v

    @field_validator('urls')
    def validate_urls(cls, v):
        # Limit the number of URLs to prevent abuse
        if len(v) > 10:
            raise ValueError("Maximum 10 URLs allowed")

        # Check for allowed domains if needed
        allowed_domains = app_config.http.ALLOWED_DOMAINS if hasattr(app_config.http, 'ALLOWED_DOMAINS') else []
        if allowed_domains:
            for url in v:
                domain = url.netloc
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


class JobStatusResponse(BaseModel):
    """Response model for job status checks"""
    status: str = Field(..., description="Job status: pending, processing, completed, or failed")
    message: Optional[str] = Field(None, description="Status message")
    story_script: Optional[str] = Field(None, description="Generated story script if completed")
    storyboard: Optional[Dict[str, Any]] = Field(None, description="Generated storyboard if completed")
    audio_url: Optional[str] = Field(None, description="URL to the generated audio file if completed")
    sources: Optional[List[str]] = Field(None, description="Sources used to generate the story")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: Optional[float] = Field(None, description="Timestamp when job was created")
    updated_at: Optional[float] = Field(None, description="Timestamp when job was last updated")

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
                            "narration_text": {"type": "string"},
                            "image_prompt": {"type": "string"},
                            "duration": {"type": "number"},
                            "start_time": {"type": "number"}
                        },
                        "required": ["scene_number", "narration_text", "image_prompt", "duration", "start_time"]
                    }
                }
            },
            "required": ["scenes"]
        }
    }
}
