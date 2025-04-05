# /viralStoryGenerator/util/config.py

from dotenv import load_dotenv
import os

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
    AUDIO_QUEUE_DIR = os.environ.get("AUDIO_QUEUE_DIR", "Output/AudioQueue")
    SOURCES_FOLDER = os.environ.get("SOURCES_FOLDER")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()

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
        TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", 90)) # Use longer timeout for Reasoning LLM Models
