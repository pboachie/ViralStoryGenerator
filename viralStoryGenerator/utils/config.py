# /viralStoryGenerator/util/config.py

from dotenv import load_dotenv
import os
import sys
import logging
from typing import List, Optional, Union

load_dotenv()

_config_logger = logging.getLogger(__name__)
_config_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
if not _config_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _config_logger.addHandler(handler)
_config_logger.propagate = False


class ConfigError(Exception):
    """Custom exception for critical configuration errors."""
    pass

class config:
    # General config
    AUDIO_QUEUE_DIR: str = os.environ.get("AUDIO_QUEUE_DIR", "Output/AudioQueue")
    SOURCES_FOLDER: Optional[str] = os.environ.get("SOURCES_FOLDER")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development").lower()
    VERSION: str = os.environ.get("APP_VERSION", "0.1.2")
    APP_TITLE: str = os.environ.get("APP_TITLE", "Viral Story Generator API")
    APP_DESCRIPTION: str = os.environ.get("APP_DESCRIPTION", "API for generating viral stories from web content")

    class elevenLabs:
        API_KEY: Optional[str] = os.environ.get("ELEVENLABS_API_KEY")
        VOICE_ID: Optional[str] = os.environ.get("ELEVENLABS_VOICE_ID")

    class llm:
        CHUNK_SIZE: int = int(os.environ.get("LLM_CHUNK_SIZE", 1000))
        ENDPOINT: Optional[str] = os.environ.get("LLM_ENDPOINT")
        MODEL: Optional[str] = os.environ.get("LLM_MODEL")
        SHOW_THINKING: bool = os.environ.get("LLM_SHOW_THINKING", "True").lower() in ["true", "1", "yes"]
        TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", 0.7))
        MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", 4096)) # 4096 is the default for GPT-3.5-turbo

    class openAI:
        API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY") # Used for DALL-E

    class httpOptions:
        TIMEOUT: int = int(os.environ.get("HTTP_TIMEOUT", 90))

    class http:
        # API server configuration
        HOST: str = os.environ.get("API_HOST", "0.0.0.0")
        PORT: int = int(os.environ.get("API_PORT", 8000))
        WORKERS: int = max(1, int(os.environ.get("API_WORKERS", os.cpu_count() or 1)))

        # Security settings
        API_KEY_ENABLED: bool = os.environ.get("API_KEY_ENABLED", "False").lower() in ["true", "1", "yes"]
        API_KEY: Optional[str] = os.environ.get("API_KEY")

        # Rate limiting
        RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "False").lower() in ["true", "1", "yes"]
        RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", 100))  # requests per window
        RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", 60))  # window in seconds

        # CORS configuration
        # SECURITY: Defaulting to empty list. '*' is insecure for production.
        # Must be explicitly set via environment variable for frontend access.
        _cors_origins_str = os.environ.get("CORS_ORIGINS", "")
        CORS_ORIGINS: List[str] = [origin.strip() for origin in _cors_origins_str.split(",") if origin.strip()] if _cors_origins_str else []

        # SSL/TLS settings for production (usually handled by reverse proxy)
        SSL_ENABLED: bool = os.environ.get("SSL_ENABLED", "False").lower() in ["true", "1", "yes"]
        SSL_CERT_FILE: Optional[str] = os.environ.get("SSL_CERT_FILE")
        SSL_KEY_FILE: Optional[str] = os.environ.get("SSL_KEY_FILE")

        # Request size limits
        MAX_REQUEST_SIZE_MB: int = int(os.environ.get("MAX_REQUEST_SIZE_MB", 10))

        # File uploads
        UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", "./uploads") # TODO: Use in API for all uploads
        MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 50))

        BASE_URL: str = os.environ.get("BASE_URL", f"http://localhost:{PORT}")

        # Allowed domains for URL scraping (Optional)
        _allowed_domains_str = os.environ.get("ALLOWED_DOMAINS", "")
        ALLOWED_DOMAINS: List[str] = [domain.strip() for domain in _allowed_domains_str.split(",") if domain.strip()] if _allowed_domains_str else []


    class storage:
        # Storage configuration
        PROVIDER: str = os.environ.get("STORAGE_PROVIDER", "local").lower()  # local, s3, azure, gcs

        # Local storage settings
        LOCAL_STORAGE_PATH: str = os.path.abspath(os.environ.get("LOCAL_STORAGE_PATH", "./storage"))
        AUDIO_STORAGE_PATH: str = os.path.abspath(os.environ.get("AUDIO_STORAGE_PATH", os.path.join(LOCAL_STORAGE_PATH, "audio")))
        STORY_STORAGE_PATH: str = os.path.abspath(os.environ.get("STORY_STORAGE_PATH", os.path.join(LOCAL_STORAGE_PATH, "stories")))
        STORYBOARD_STORAGE_PATH: str = os.path.abspath(os.environ.get("STORYBOARD_STORAGE_PATH", os.path.join(LOCAL_STORAGE_PATH, "storyboards")))

        # File retention policy (in days, 0 or negative = keep forever)
        FILE_RETENTION_DAYS: int = int(os.environ.get("FILE_RETENTION_DAYS", 30))

        # Cleanup interval in hours
        CLEANUP_INTERVAL_HOURS: int = int(os.environ.get("CLEANUP_INTERVAL_HOURS", 24))

        # S3 settings (if using S3)
        S3_BUCKET_NAME: Optional[str] = os.environ.get("S3_BUCKET_NAME")
        S3_REGION: Optional[str] = os.environ.get("S3_REGION")
        S3_ACCESS_KEY: Optional[str] = os.environ.get("S3_ACCESS_KEY")
        S3_SECRET_KEY: Optional[str] = os.environ.get("S3_SECRET_KEY")
        S3_ENDPOINT_URL: Optional[str] = os.environ.get("S3_ENDPOINT_URL")

        # Azure Blob Storage settings (if using Azure)
        AZURE_ACCOUNT_NAME: Optional[str] = os.environ.get("AZURE_ACCOUNT_NAME")
        AZURE_ACCOUNT_KEY: Optional[str] = os.environ.get("AZURE_ACCOUNT_KEY")
        AZURE_CONTAINER_NAME: Optional[str] = os.environ.get("AZURE_CONTAINER_NAME")

    class redis:
        # Enable/disable Redis queue (fallback to direct processing if disabled)
        # Critical for async operation via /generate endpoint
        ENABLED: bool = os.environ.get("REDIS_ENABLED", "True").lower() in ["true", "1", "yes"]

        # Redis connection settings
        HOST: str = os.environ.get("REDIS_HOST", "localhost")
        PORT: int = int(os.environ.get("REDIS_PORT", 6379))
        DB: int = int(os.environ.get("REDIS_DB", 0))
        PASSWORD: Optional[str] = os.environ.get("REDIS_PASSWORD", None)

        # Queue configuration (used by api_worker/queue_worker)
        QUEUE_NAME: str = os.environ.get("REDIS_QUEUE_NAME", "viralStoryGenerator_jobs")
        RESULT_PREFIX: str = os.environ.get("REDIS_RESULT_PREFIX", "vs_result:")
        TTL: int = int(os.environ.get("REDIS_RESULT_TTL", 3600 * 24))

        # Worker configuration (used by worker process)
        WORKER_BATCH_SIZE: int = int(os.environ.get("REDIS_WORKER_BATCH_SIZE", 5))
        WORKER_SLEEP_INTERVAL: int = int(os.environ.get("REDIS_WORKER_SLEEP_INTERVAL", 1))
        WORKER_MAX_CONCURRENT: int = int(os.environ.get("REDIS_WORKER_MAX_CONCURRENT", 3))


    class monitoring:
        # Prometheus monitoring settings
        METRICS_ENABLED: bool = os.environ.get("METRICS_ENABLED", "False").lower() in ["true", "1", "yes"]
        METRICS_ENDPOINT: str = os.environ.get("METRICS_ENDPOINT", "/metrics")

        # Health check settings
        HEALTH_CHECK_ENABLED: bool = os.environ.get("HEALTH_CHECK_ENABLED", "True").lower() in ["true", "1", "yes"]
        HEALTH_CHECK_ENDPOINT: str = os.environ.get("HEALTH_CHECK_ENDPOINT", "/health")

        # Tracing configuration (for distributed tracing)
        TRACING_ENABLED: bool = os.environ.get("TRACING_ENABLED", "False").lower() in ["true", "1", "yes"]
        TRACING_EXPORTER: str = os.environ.get("TRACING_EXPORTER", "jaeger") # Options: jaeger, zipkin, otlp

    class security:
        # Security settings
        # Regex pattern for voice ID validation (example: typical ElevenLabs ID)
        VOICE_ID_PATTERN: str = os.environ.get("VOICE_ID_PATTERN", r"^[a-zA-Z0-9]{20}$")
        SANITIZE_MAX_LENGTH: int = int(os.environ.get("SANITIZE_MAX_LENGTH", 1000))
        # Characters to remove in basic sanitization (use with caution)
        DANGEROUS_CHARS: List[str] = list(os.environ.get("DANGEROUS_CHARS", "&|;$`\\<>\"'()"))

        # Allowed file extensions for any potential upload features
        ALLOWED_FILE_TYPES: List[str] = [ext.strip().lower() for ext in os.environ.get("ALLOWED_FILE_TYPES", "mp3,wav,txt,json").split(",") if ext.strip()]

        # Base path for allowed source material folders. Crucial for preventing traversal.
        SOURCE_MATERIALS_PATH: str = os.path.abspath(os.environ.get("SOURCE_MATERIALS_PATH", "./data/sources"))


# --- Configuration Validation ---
def validate_config_on_startup(cfg: config):
    """
    Performs basic checks on the loaded configuration, especially for production.
    Call this function early in the application startup sequence.
    """
    _config_logger.info("Validating application configuration...")
    is_production = cfg.ENVIRONMENT == "production"

    # --- Security Checks ---
    if is_production:
        if not cfg.http.API_KEY_ENABLED:
            _config_logger.warning("SECURITY WARNING: API Key authentication (API_KEY_ENABLED) is DISABLED in production environment!")
        elif not cfg.http.API_KEY:
            raise ConfigError("CRITICAL SECURITY ERROR: API Key authentication (API_KEY_ENABLED) is TRUE but no API_KEY is set in production environment!")
        elif len(cfg.http.API_KEY) < 32:
             _config_logger.warning(f"SECURITY WARNING: API_KEY seems short ({len(cfg.http.API_KEY)} characters). Consider a longer, randomly generated key.")


        if "*" in cfg.http.CORS_ORIGINS:
            _config_logger.critical("CRITICAL SECURITY WARNING: CORS_ORIGINS allows '*' in production. This MUST be restricted to your frontend domain(s).")
        elif not cfg.http.CORS_ORIGINS:
             _config_logger.warning("CORS_ORIGINS is empty in production. Frontend access might be blocked unless hosted on the same origin.")


        if not cfg.http.RATE_LIMIT_ENABLED:
            _config_logger.warning("Rate limiting (RATE_LIMIT_ENABLED) is DISABLED in production. Enabling is recommended.")

        if cfg.http.SSL_ENABLED and (not cfg.http.SSL_CERT_FILE or not cfg.http.SSL_KEY_FILE):
             _config_logger.warning("SSL_ENABLED is true, but SSL_CERT_FILE or SSL_KEY_FILE is missing. SSL might not function correctly (unless handled by proxy).")

    # --- Service Dependency Checks ---
    if cfg.elevenLabs.VOICE_ID and not cfg.elevenLabs.API_KEY:
         _config_logger.warning("ELEVENLABS_VOICE_ID is set, but ELEVENLABS_API_KEY is missing. Audio generation will likely fail.")

    if not cfg.llm.ENDPOINT or not cfg.llm.MODEL:
        _config_logger.warning("LLM_ENDPOINT or LLM_MODEL is not configured. Story generation will likely fail.")

    if cfg.openAI.API_KEY is None:
         _config_logger.warning("OPENAI_API_KEY is not configured. DALL-E image generation for storyboards will fail.")

    if cfg.storage.PROVIDER == "s3" and (not cfg.storage.S3_BUCKET_NAME or not cfg.storage.S3_ACCESS_KEY or not cfg.storage.S3_SECRET_KEY):
        _config_logger.error("Storage provider is S3, but required S3 settings (BUCKET_NAME, ACCESS_KEY, SECRET_KEY) are missing.")
    elif cfg.storage.PROVIDER == "azure" and (not cfg.storage.AZURE_ACCOUNT_NAME or not cfg.storage.AZURE_ACCOUNT_KEY or not cfg.storage.AZURE_CONTAINER_NAME):
        _config_logger.error("Storage provider is Azure, but required Azure settings (ACCOUNT_NAME, ACCOUNT_KEY, CONTAINER_NAME) are missing.")

    if cfg.redis.ENABLED and not (cfg.redis.HOST and cfg.redis.PORT is not None):
         _config_logger.error("Redis is ENABLED, but REDIS_HOST or REDIS_PORT is not configured.")

    # --- Path Checks ---
    if not os.path.exists(cfg.security.SOURCE_MATERIALS_PATH):
        _config_logger.warning(f"Source materials base path does not exist: {cfg.security.SOURCE_MATERIALS_PATH}. Creating it.")
        try:
            os.makedirs(cfg.security.SOURCE_MATERIALS_PATH, exist_ok=True)
        except OSError as e:
            _config_logger.error(f"Failed to create source materials directory: {e}")
            # Consider raising ConfigError if sources are essential

    if cfg.storage.PROVIDER == "local":
        local_base = cfg.storage.LOCAL_STORAGE_PATH
        paths_to_check = {
            "AUDIO_STORAGE_PATH": cfg.storage.AUDIO_STORAGE_PATH,
            "STORY_STORAGE_PATH": cfg.storage.STORY_STORAGE_PATH,
            "STORYBOARD_STORAGE_PATH": cfg.storage.STORYBOARD_STORAGE_PATH,
        }
        for name, path in paths_to_check.items():
            if not path.startswith(local_base):
                _config_logger.error(f"Configuration Error: {name} ({path}) is not inside LOCAL_STORAGE_PATH ({local_base}). This might indicate misconfiguration or a security risk.")
                # raise ConfigError(f"{name} path is outside the configured local storage base path.")

    _config_logger.info("Configuration validation complete.")


app_config = config()