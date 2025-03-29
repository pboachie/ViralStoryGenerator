#/viralStoryGenerator/util/config.py

from dotenv import load_dotenv
import os

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
