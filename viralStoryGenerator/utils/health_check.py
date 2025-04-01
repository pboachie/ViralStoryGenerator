import time
import redis
import os
from viralStoryGenerator.utils.config import config
from viralStoryGenerator.src.logger import logger as _logger
import requests

def get_service_status():
    """
    Check the status of various services and return detailed information.

    Returns:
        dict: A dictionary containing the status of each service.
    """
    service_status = {}

    # Check Redis connection
    try:
        redis_client = redis.Redis(
            host=config.redis.HOST,
            port=config.redis.PORT,
            db=config.redis.DB,
            password=config.redis.PASSWORD,
            decode_responses=True
        )
        redis_client.ping()
        service_status["redis"] = {
            "status": "up",
            "details": "Redis is connected",
            "uptime": redis_client.info().get("uptime_in_seconds", "unknown")
        }
    except Exception as e:
        _logger.error(f"Redis connection failed: {e}")
        service_status["redis"] = {
            "status": "down",
            "details": str(e)
        }

    # Check storage availability
    try:
        # Simulate a storage check (e.g., checking if a directory is writable)
        test_file_path = f"{config.storage.LOCAL_STORAGE_PATH}/health_check.tmp"
        with open(test_file_path, "w") as f:
            f.write("test")
        service_status["storage"] = {
            "status": "up",
            "details": "Storage is writable"
        }
        # Clean up test file
        os.remove(test_file_path)
    except Exception as e:
        _logger.error(f"Storage check failed: {e}")
        service_status["storage"] = {
            "status": "down",
            "details": str(e)
        }

    # Check ElevenLabs API availability
    try:
        url = "https://api.elevenlabs.io"
        response = requests.get(url, timeout=5)
        status = "up" if response.status_code < 500 else "down"
        details = "ElevenLabs API is available" if status == "up" else "ElevenLabs API is unavailable"
        service_status["elevenlabs"] = {
            "status": status,
            "details": details,
            "response_time": response.elapsed.total_seconds() if status == "up" else None
        }
    except Exception as e:
        _logger.error(f"ElevenLabs API check failed: {e}")
        service_status["elevenlabs"] = {
            "status": "down",
            "details": str(e)
        }

    return service_status