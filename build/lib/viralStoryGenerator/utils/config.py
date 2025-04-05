#/viralStoryGenerator/util/config.py

from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

class config:
    # General config
    AUDIO_QUEUE_DIR = os.environ.get("AUDIO_QUEUE_DIR", "Output/AudioQueue")
    SOURCES_FOLDER = os.environ.get("SOURCES_FOLDER")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
    VERSION = os.environ.get("APP_VERSION", "0.1.2")
    APP_TITLE = os.environ.get("APP_TITLE", "Viral Story Generator API")
    APP_DESCRIPTION = os.environ.get("APP_DESCRIPTION", "API for generating viral stories from web content")

    class elevenLabs:
        API_KEY = os.environ.get("ELEVENLABS_API_KEY")
        VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

    class llm:
        CHUNK_SIZE = int(os.environ.get("LLM_CHUNK_SIZE", 1000))
        ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
        MODEL = os.environ.get("LLM_MODEL")
        SHOW_THINKING = os.environ.get("LLM_SHOW_THINKING", "True").lower() in ["true", "1", "yes"]
        TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.7))
        MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 4096)) # 4096 is the default for GPT-3.5-turbo

    class openAI:
        API_KEY = os.environ.get("OPENAI_API_KEY")

    class httpOptions:
        TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", 90)) # Use longer timeout for Reasoning LLM Models

    class http:
        # API server configuration
        HOST = os.environ.get("API_HOST", "0.0.0.0")
        PORT = int(os.environ.get("API_PORT", 8000))
        WORKERS = int(os.environ.get("API_WORKERS", os.cpu_count() or 1))

        # Security settings
        API_KEY_ENABLED = os.environ.get("API_KEY_ENABLED", "False").lower() in ["true", "1", "yes"]
        API_KEY = os.environ.get("API_KEY", "")

        # Rate limiting
        RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "False").lower() in ["true", "1", "yes"]
        RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 100))  # requests per window
        RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", 60))  # window in seconds

        # CORS configuration
        CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

        # SSL/TLS settings for production
        SSL_ENABLED = os.environ.get("SSL_ENABLED", "False").lower() in ["true", "1", "yes"]
        SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE", "")
        SSL_KEY_FILE = os.environ.get("SSL_KEY_FILE", "")

        # Request size limits
        MAX_REQUEST_SIZE_MB = int(os.environ.get("MAX_REQUEST_SIZE_MB", 10))

        # File uploads
        UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
        MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 50))

        # Base URL for serving files - used for constructing absolute URLs in responses
        BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

    class storage:
        # Storage configuration
        PROVIDER = os.environ.get("STORAGE_PROVIDER", "local")  # local, s3, azure, gcs

        # Local storage settings
        LOCAL_STORAGE_PATH = os.environ.get("LOCAL_STORAGE_PATH", "./storage")
        AUDIO_STORAGE_PATH = os.environ.get("AUDIO_STORAGE_PATH", "./storage/audio")
        STORY_STORAGE_PATH = os.environ.get("STORY_STORAGE_PATH", "./storage/stories")
        STORYBOARD_STORAGE_PATH = os.environ.get("STORYBOARD_STORAGE_PATH", "./storage/storyboards")

        # File retention policy (in days, 0 = keep forever)
        FILE_RETENTION_DAYS = int(os.environ.get("FILE_RETENTION_DAYS", 30))

        # Cleanup interval in hours
        CLEANUP_INTERVAL_HOURS = int(os.environ.get("CLEANUP_INTERVAL_HOURS", 24))

        # S3 settings (if using S3)
        S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
        S3_REGION = os.environ.get("S3_REGION", "us-east-1")
        S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
        S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
        S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "")  # For non-AWS S3-compatible storage

        # Azure Blob Storage settings (if using Azure)
        AZURE_ACCOUNT_NAME = os.environ.get("AZURE_ACCOUNT_NAME", "")
        AZURE_ACCOUNT_KEY = os.environ.get("AZURE_ACCOUNT_KEY", "")
        AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "viralstories")

    class redis:
        # Redis connection settings
        HOST = os.environ.get("REDIS_HOST", "localhost")
        PORT = int(os.environ.get("REDIS_PORT", 6379))
        DB = int(os.environ.get("REDIS_DB", 0))
        PASSWORD = os.environ.get("REDIS_PASSWORD", None)

        # Queue configuration
        QUEUE_NAME = os.environ.get("REDIS_QUEUE_NAME", "crawl4ai_queue")
        RESULT_PREFIX = os.environ.get("REDIS_RESULT_PREFIX", "crawl4ai_result:")
        TTL = int(os.environ.get("REDIS_RESULT_TTL", 3600))  # 1-hour default TTL for results

        # Worker configuration
        WORKER_BATCH_SIZE = int(os.environ.get("REDIS_WORKER_BATCH_SIZE", 5))
        WORKER_SLEEP_INTERVAL = int(os.environ.get("REDIS_WORKER_SLEEP_INTERVAL", 1))
        WORKER_MAX_CONCURRENT = int(os.environ.get("REDIS_WORKER_MAX_CONCURRENT", 3))

        # Enable/disable Redis queue (fallback to direct scraping if disabled)
        ENABLED = os.environ.get("REDIS_ENABLED", "True").lower() in ["true", "1", "yes"]

    class monitoring:
        # Prometheus monitoring settings
        METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "False").lower() in ["true", "1", "yes"]
        METRICS_ENDPOINT = os.environ.get("METRICS_ENDPOINT", "/metrics")

        # Health check settings
        HEALTH_CHECK_ENABLED = os.environ.get("HEALTH_CHECK_ENABLED", "True").lower() in ["true", "1", "yes"]
        HEALTH_CHECK_ENDPOINT = os.environ.get("HEALTH_CHECK_ENDPOINT", "/health")

        # Tracing configuration (for distributed tracing)
        TRACING_ENABLED = os.environ.get("TRACING_ENABLED", "False").lower() in ["true", "1", "yes"]
        TRACING_EXPORTER = os.environ.get("TRACING_EXPORTER", "jaeger")

    class security:
        # Security settings
        VOICE_ID_PATTERN = os.environ.get("VOICE_ID_PATTERN", "")  # Regex pattern for voice ID validation
        SANITIZE_MAX_LENGTH = int(os.environ.get("SANITIZE_MAX_LENGTH", 1000))
        DANGEROUS_CHARS = list(os.environ.get("DANGEROUS_CHARS", "&|;$`\\"))  # Characters to remove in sanitization
        ALLOWED_FILE_TYPES = os.environ.get("ALLOWED_FILE_TYPES", "mp3,wav,txt,json").split(",")  # Allowed file extensions
        SOURCE_MATERIALS_PATH = os.environ.get("SOURCE_MATERIALS_PATH", "./data/sources")  # Path to source materials
