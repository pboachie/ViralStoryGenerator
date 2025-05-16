# /viralStoryGenerator/util/config.py

from dotenv import load_dotenv
import os
import sys
from typing import List, Optional, Union, Tuple, Dict, Any
import logging
import torch
import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

load_dotenv()


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

    ENABLE_IMAGE_GENERATION: bool = os.environ.get("ENABLE_IMAGE_GENERATION", "True").lower() in ["true", "1", "yes"]
    ENABLE_AUDIO_GENERATION: bool = os.environ.get("ENABLE_AUDIO_GENERATION", "True").lower() in ["true", "1", "yes"]

    class elevenLabs:
        API_KEY: Optional[str] = os.environ.get("ELEVENLABS_API_KEY")
        VOICE_ID: Optional[str] = os.environ.get("ELEVENLABS_VOICE_ID")
        ENABLED: bool = os.environ.get("ENABLE_AUDIO_GENERATION", "True").lower() in ["true", "1", "yes"]
        DEFAULT_MODEL_ID: str = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
        DEFAULT_STABILITY: float = float(os.environ.get("ELEVENLABS_STABILITY", 0.5))
        DEFAULT_SIMILARITY_BOOST: float = float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", 0.75))

    class llm:
        CHUNK_SIZE: int = int(os.environ.get("LLM_CHUNK_SIZE", 5000))
        ENDPOINT: Optional[str] = os.environ.get("LLM_ENDPOINT")
        MODEL: Optional[str] = os.environ.get("LLM_MODEL") # Default model
        MODEL_MULTI: Optional[str] = os.environ.get("LLM_MODEL_MULTI", MODEL)
        MODEL_SMALL: Optional[str] = os.environ.get("LLM_MODEL_SMALL", MODEL)
        MODEL_LARGE: Optional[str] = os.environ.get("LLM_MODEL_LARGE", MODEL)
        SHOW_THINKING: bool = os.environ.get("LLM_SHOW_THINKING", "True").lower() in ["true", "1", "yes"]
        TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", 0.95))
        MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", 32768)) # 4096 is the default for GPT-3.5-turbo
        CLEANING_MAX_PROMPT_CHARS: int = int(os.environ.get("LLM_CLEANING_MAX_PROMPT_CHARS", 20000))
        CLEANING_MAX_OUTPUT_TOKENS: int = int(os.environ.get("LLM_CLEANING_MAX_OUTPUT_TOKENS", 32768))
        MIN_MARKDOWN_LENGTH_FOR_LLM_CLEANING: int = int(os.environ.get("LLM_MIN_CLEANING_LENGTH", 200))

    class openAI:
        API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY") # Used for DALL-E
        ENABLED: bool = os.environ.get("ENABLE_IMAGE_GENERATION", "True").lower() in ["true", "1", "yes"]

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
        METADATA_STORAGE_PATH: str = os.path.abspath(os.environ.get("METADATA_STORAGE_PATH", os.path.join(LOCAL_STORAGE_PATH, "metadata")))
        SCREENSHOT_STORAGE_PATH: str = os.path.abspath(os.environ.get("SCREENSHOT_STORAGE_PATH", os.path.join(LOCAL_STORAGE_PATH, "screenshots")))

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
        QUEUE_NAME: str = os.environ.get("REDIS_QUEUE_NAME", "api_jobs")
        RESULT_PREFIX: str = os.environ.get("REDIS_RESULT_PREFIX", "vs_result:")
        TTL: int = int(os.environ.get("REDIS_RESULT_TTL", 3600 * 24))
        SCRAPE_QUEUE_NAME: str = os.environ.get("REDIS_SCRAPE_QUEUE_NAME", "scraper_jobs")

        API_WORKER_GROUP_NAME: str = os.environ.get("REDIS_API_WORKER_GROUP_NAME", "api-workers")

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

    class rag:
        ENABLED: bool = os.environ.get("RAG_ENABLED", "True").lower() in ["true", "1", "yes"]
        EMBEDDING_MODEL: str = os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        VECTOR_DB_PATH: str = os.path.abspath(os.environ.get("RAG_VECTOR_DB_PATH", "./vector_db"))
        RELEVANT_CHUNKS_COUNT: int = int(os.environ.get("RAG_RELEVANT_CHUNKS_COUNT", 5))
        CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", 500)) # Chunks for RAG can often be smaller
        CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", 50))
        DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    class storyboard:
        ENABLE_STORYBOARD_GENERATION: bool = os.environ.get("ENABLE_STORYBOARD_GENERATION", "True").lower() in ["true", "1", "yes"]
        WPM: int = int(os.environ.get("STORYBOARD_WPM", 150))
        STRUCTURE_MAX_RETRIES: int = int(os.environ.get("STORYBOARD_STRUCTURE_MAX_RETRIES", 2))

    class scraper:
        # Content extraction
        BM25_THRESHOLD: float = float(os.environ.get("SCRAPER_BM25_THRESHOLD", 1.2))
        PRUNING_THRESHOLD: float = float(os.environ.get("SCRAPER_PRUNING_THRESHOLD", 0.48))
        PRUNING_THRESHOLD_TYPE: str = os.environ.get("SCRAPER_PRUNING_THRESHOLD_TYPE", "fixed") # "fixed" or "percentile"
        MD_IGNORE_LINKS: bool = os.environ.get("SCRAPER_MD_IGNORE_LINKS", "True").lower() in ["true", "1", "yes"]
        CHECK_ROBOTS_TXT: bool = os.environ.get("SCRAPER_CHECK_ROBOTS_TXT", "True").lower() in ["true", "1", "yes"]
        ARUN_MANY_TIMEOUT: float = float(os.environ.get("SCRAPER_ARUN_MANY_TIMEOUT", 60.0))

        # Rate Limiter for Dispatchers
        _rl_base_delay_str = os.environ.get("SCRAPER_RL_BASE_DELAY", "1.0,3.0")
        try:
            RL_BASE_DELAY: Tuple[float, float] = tuple(map(float, _rl_base_delay_str.split(','))) # type: ignore
            if len(RL_BASE_DELAY) != 2:
                RL_BASE_DELAY = (1.0, 3.0)
                _logger.warning(f"SCRAPER_RL_BASE_DELAY format error. Defaulting to {RL_BASE_DELAY}. Expected 'min,max'.")
        except ValueError:
            RL_BASE_DELAY = (1.0, 3.0)
            _logger.warning(f"SCRAPER_RL_BASE_DELAY value error. Defaulting to {RL_BASE_DELAY}. Expected numeric values.")

        RL_MAX_DELAY: float = float(os.environ.get("SCRAPER_RL_MAX_DELAY", 60.0))
        RL_MAX_RETRIES: int = int(os.environ.get("SCRAPER_RL_MAX_RETRIES", 3))
        _rl_codes_str = os.environ.get("SCRAPER_RL_CODES", "429,503")
        RL_CODES: List[int] = [int(code.strip()) for code in _rl_codes_str.split(',') if code.strip().isdigit()]

        # Crawler Monitor for Dispatchers
        MONITOR_ENABLED: bool = os.environ.get("SCRAPER_MONITOR_ENABLED", "False").lower() in ["true", "1", "yes"]
        MONITOR_MAX_ROWS: int = int(os.environ.get("SCRAPER_MONITOR_MAX_ROWS", 10))
        MONITOR_DISPLAY_MODE: str = os.environ.get("SCRAPER_MONITOR_DISPLAY_MODE", "AGGREGATED") # DETAILED or AGGREGATED

        # Semaphore Dispatcher
        DISP_SEMA_MAX_PERMIT: int = int(os.environ.get("SCRAPER_DISP_SEMA_MAX_PERMIT", 10))

        # MemoryAdaptive Dispatcher
        DISP_MEM_THRESHOLD: float = float(os.environ.get("SCRAPER_DISP_MEM_THRESHOLD", 85.0))
        DISP_MEM_INTERVAL: float = float(os.environ.get("SCRAPER_DISP_MEM_INTERVAL", 1.0))
        DISP_MEM_MAX_PERMIT: int = int(os.environ.get("SCRAPER_DISP_MEM_MAX_PERMIT", 5))
        DISP_MEM_WAIT_TIMEOUT: float = float(os.environ.get("SCRAPER_DISP_MEM_WAIT_TIMEOUT", 300.0))

        # Scraper Worker Specific (distinct from general Redis worker settings)
        WORKER_MAX_CONCURRENT_JOBS: int = int(os.environ.get("SCRAPER_WORKER_MAX_CONCURRENT_JOBS", 2))
        WORKER_SHUTDOWN_TIMEOUT: int = int(os.environ.get("SCRAPER_WORKER_SHUTDOWN_TIMEOUT", 30))

    @staticmethod
    def get_device() -> str:
        """Returns the appropriate device for PyTorch operations."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

# --- Configuration Validation ---
def validate_config_on_startup(cfg: config):
    """
    Performs basic checks on the loaded configuration, especially for production.
    Call this function early in the application startup sequence.
    """
    _logger.info("Validating application configuration...")
    is_production = cfg.ENVIRONMENT == "production"

    # --- Security Checks ---
    if is_production:
        if not cfg.http.API_KEY_ENABLED:
            _logger.warning("SECURITY WARNING: API Key authentication (API_KEY_ENABLED) is DISABLED in production environment!")
        elif not cfg.http.API_KEY:
            raise ConfigError("CRITICAL SECURITY ERROR: API Key authentication (API_KEY_ENABLED) is TRUE but no API_KEY is set in production environment!")
        elif len(cfg.http.API_KEY) < 32:
             _logger.warning(f"SECURITY WARNING: API_KEY seems short ({len(cfg.http.API_KEY)} characters). Consider a longer, randomly generated key.")


        if "*" in cfg.http.CORS_ORIGINS:
            _logger.critical("CRITICAL SECURITY WARNING: CORS_ORIGINS allows '*' in production. This MUST be restricted to your frontend domain(s).")
        elif not cfg.http.CORS_ORIGINS:
             _logger.warning("CORS_ORIGINS is empty in production. Frontend access might be blocked unless hosted on the same origin.")


        if not cfg.http.RATE_LIMIT_ENABLED:
            _logger.warning("Rate limiting (RATE_LIMIT_ENABLED) is DISABLED in production. Enabling is recommended.")

        if cfg.http.SSL_ENABLED and (not cfg.http.SSL_CERT_FILE or not cfg.http.SSL_KEY_FILE):
             _logger.warning("SSL_ENABLED is true, but SSL_CERT_FILE or SSL_KEY_FILE is missing. SSL might not function correctly (unless handled by proxy).")

    # --- Service Dependency Checks ---
    if cfg.elevenLabs.ENABLED and not cfg.elevenLabs.API_KEY: # Check if enabled AND key missing
         _logger.warning("Audio generation (ENABLE_AUDIO_GENERATION) is TRUE, but ELEVENLABS_API_KEY is missing. Audio generation will fail.")
    elif not cfg.elevenLabs.ENABLED:
         _logger.critical("Audio generation via ElevenLabs is globally disabled (ENABLE_AUDIO_GENERATION=False).")


    if cfg.openAI.ENABLED and not cfg.openAI.API_KEY: # Check if enabled AND key missing
        _logger.warning("Image generation (ENABLE_IMAGE_GENERATION) is TRUE, but OPENAI_API_KEY is missing. DALL-E image generation for storyboards will fail.")
    elif not cfg.openAI.ENABLED:
         _logger.critical("Image generation via DALL-E is globally disabled (ENABLE_IMAGE_GENERATION=False).")


    if not cfg.llm.ENDPOINT or not cfg.llm.MODEL:
        _logger.critical("LLM_ENDPOINT or the default LLM_MODEL is not configured. Core functionality might fail.")
    if not cfg.llm.MODEL_MULTI:
        _logger.warning("LLM_MODEL_MULTI is not explicitly configured. Falling back to default LLM_MODEL.")
    if not cfg.llm.MODEL_SMALL:
        _logger.warning("LLM_MODEL_SMALL is not explicitly configured. Falling back to default LLM_MODEL.")
    if not cfg.llm.MODEL_LARGE:
        _logger.warning("LLM_MODEL_LARGE is not explicitly configured. Falling back to default LLM_MODEL.")

    if cfg.openAI.API_KEY is None:
         _logger.critical("OPENAI_API_KEY is not configured. DALL-E image generation for storyboards will fail.")

    if cfg.storage.PROVIDER == "s3" and (not cfg.storage.S3_BUCKET_NAME or not cfg.storage.S3_ACCESS_KEY or not cfg.storage.S3_SECRET_KEY):
        _logger.error("Storage provider is S3, but required S3 settings (BUCKET_NAME, ACCESS_KEY, SECRET_KEY) are missing.")
    elif cfg.storage.PROVIDER == "azure" and (not cfg.storage.AZURE_ACCOUNT_NAME or not cfg.storage.AZURE_ACCOUNT_KEY or not cfg.storage.AZURE_CONTAINER_NAME):
        _logger.error("Storage provider is Azure, but required Azure settings (ACCOUNT_NAME, ACCOUNT_KEY, CONTAINER_NAME) are missing.")

    if cfg.redis.ENABLED and not (cfg.redis.HOST and cfg.redis.PORT is not None):
         _logger.error("Redis is ENABLED, but REDIS_HOST or REDIS_PORT is not configured.")

    # --- Path Checks ---
    if not os.path.exists(cfg.security.SOURCE_MATERIALS_PATH):
        _logger.warning(f"Source materials base path does not exist: {cfg.security.SOURCE_MATERIALS_PATH}. Creating it.")
        try:
            os.makedirs(cfg.security.SOURCE_MATERIALS_PATH, exist_ok=True)
        except OSError as e:
            _logger.error(f"Failed to create source materials directory: {e}")
            # Consider raising ConfigError if sources are essential

    if cfg.storage.PROVIDER == "local":
        local_base = cfg.storage.LOCAL_STORAGE_PATH
        paths_to_check = {
            "AUDIO_STORAGE_PATH": cfg.storage.AUDIO_STORAGE_PATH,
            "STORY_STORAGE_PATH": cfg.storage.STORY_STORAGE_PATH,
            "STORYBOARD_STORAGE_PATH": cfg.storage.STORYBOARD_STORAGE_PATH,
            "METADATA_STORAGE_PATH": cfg.storage.METADATA_STORAGE_PATH,
        }
        for name, path in paths_to_check.items():
            if not path.startswith(local_base):
                _logger.error(f"Configuration Error: {name} ({path}) is not inside LOCAL_STORAGE_PATH ({local_base}). This might indicate misconfiguration or a security risk.")
                # raise ConfigError(f"{name} path is outside the configured local storage base path.")

    # --- RAG Checks ---
    if cfg.rag.ENABLED:
        if not os.path.exists(cfg.rag.VECTOR_DB_PATH):
            _logger.warning(f"RAG Vector DB path does not exist: {cfg.rag.VECTOR_DB_PATH}. Attempting to create.")
            try:
                os.makedirs(cfg.rag.VECTOR_DB_PATH, exist_ok=True)
            except OSError as e:
                _logger.error(f"Failed to create RAG Vector DB directory: {e}")

    _logger.info("Configuration validation complete.")


app_config = config()
validate_config_on_startup(app_config)