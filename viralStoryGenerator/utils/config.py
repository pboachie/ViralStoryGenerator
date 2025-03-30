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
