"""
Health check utilities for ViralStoryGenerator API.
"""
import shutil
import time
import os
import redis
import requests
from typing import Dict, Any, Optional
from datetime import datetime

from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.models import ServiceStatusDetail

# Store the application start time
APP_START_TIME = time.time()

async def check_redis_status() -> Dict[str, Any]:
    """Check Redis connection status."""
    if not app_config.redis.ENABLED:
        return {
            "status": "unavailable",
            "uptime": "N/A",
            "response_time": "N/A",
            "message": "Redis is disabled in configuration."
        }

    try:
        start_time = time.time()
        redis_client = redis.Redis(
            host=app_config.redis.HOST,
            port=app_config.redis.PORT,
            db=app_config.redis.DB,
            password=app_config.redis.PASSWORD,
            socket_timeout=2,
            decode_responses=True
        )

        # Check connection with ping
        redis_client.ping()

        # Get Redis uptime
        server_info = redis_client.info('server')
        uptime_in_seconds = server_info.get('uptime_in_seconds', 0)

        response_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "status": "healthy",
            "uptime": uptime_in_seconds,
            "response_time": response_time,
            "message": "Connected to Redis successfully."
        }
    except Exception as e:
        _logger.warning(f"Redis health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "uptime": "Unknown",
            "response_time": -1,
            "message": f"Failed to connect: {str(e)}"
        }

async def check_elevenlabs_status() -> Dict[str, Any]:
    """Check ElevenLabs API status."""
    try:
        start_time = time.time()

        headers = {}
        api_key = app_config.elevenLabs.API_KEY
        if api_key:
            headers["xi-api-key"] = api_key

        # Using a lightweight endpoint to check status
        response = requests.get(
            "https://api.elevenlabs.io/v1/user",
            headers=headers,
            timeout=5
        )

        response_time = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code == 200:
            return {
                "status": "healthy",
                "uptime": "N/A",  # ElevenLabs doesn't provide uptime info
                "response_time": response_time,
                "message": "ElevenLabs API is operational."
            }
        elif response.status_code == 401:
            return {
                "status": "degraded",
                "uptime": "N/A",
                "response_time": response_time,
                "message": "ElevenLabs API is operational but authentication failed."
            }
        else:
            return {
                "status": "unhealthy",
                "uptime": "N/A",
                "response_time": response_time,
                "message": f"ElevenLabs API returned status code {response.status_code}."
            }
    except Exception as e:
        _logger.warning(f"ElevenLabs health check failed: {str(e)}")
        return {
            "status": "unknown",
            "uptime": "N/A",
            "response_time": -1,
            "message": f"Failed to connect: {str(e)}"
        }

async def check_llm_status() -> Dict[str, Any]:
    """Check LLM endpoint status."""
    try:
        start_time = time.time()

        # Prepare headers based on LLM type
        headers = {"Content-Type": "application/json"}

        # Get endpoint from config
        endpoint = app_config.llm.ENDPOINT

        # If it's an OpenAI-compatible endpoint, just ping it with a GET request
        # to avoid unnecessary token usage
        # Extract base URL and create models endpoint
        v1_index = endpoint.find("/v1/")
        if v1_index != -1:
            models_endpoint = endpoint[:v1_index + 4] + "models"  # +4 to include "/v1/"
        else:
            # Fallback for unexpected endpoint format
            models_endpoint = endpoint.rsplit("/", 2)[0] + "/models"

        response = requests.get(
            models_endpoint,
            headers=headers,
            timeout=5
        )

        response_time = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code < 500:  # Accept any non-server error as "working"
            return {
                "status": "healthy",
                "uptime": "N/A",  # LLM doesn't provide uptime info
                "response_time": response_time,
                "message": "LLM service is responsive."
            }
        else:
            return {
                "status": "unhealthy",
                "uptime": "N/A",
                "response_time": response_time,
                "message": f"LLM service returned status code {response.status_code}."
            }
    except Exception as e:
        _logger.warning(f"LLM health check failed: {str(e)}")
        return {
            "status": "unknown",
            "uptime": "N/A",
            "response_time": -1,
            "message": f"Failed to connect: {str(e)}"
        }

async def check_disk_status() -> Dict[str, Any]:
    """Check disk space availability."""
    try:
        if app_config.storage.PROVIDER != "local":
            return {
                "status": "not_applicable",
                "uptime": "N/A",
                "response_time": "N/A",
                "message": f"Using non-local storage provider: {app_config.storage.PROVIDER}"
            }

        # Get disk usage stats for the storage path
        storage_path = app_config.storage.LOCAL_STORAGE_PATH
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)

        total, used, free = shutil.disk_usage(storage_path)

        # Convert to GB
        total_gb = total / (1024 ** 3)
        used_gb = used / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        percent_used = (used / total) * 100

        status = "healthy"
        message = f"Disk space: {free_gb:.2f} GB free of {total_gb:.2f} GB total ({percent_used:.1f}% used)"

        # Warning thresholds
        if percent_used > 90:
            status = "unhealthy"
            message = f"CRITICAL: Only {free_gb:.2f} GB free ({percent_used:.1f}% used)"
        elif percent_used > 80:
            status = "degraded"
            message = f"WARNING: Only {free_gb:.2f} GB free ({percent_used:.1f}% used)"

        return {
            "status": status,
            "uptime": "N/A",
            "response_time": "N/A",
            "message": message
        }
    except Exception as e:
        _logger.warning(f"Disk status check failed: {str(e)}")
        return {
            "status": "unknown",
            "uptime": "N/A",
            "response_time": "N/A",
            "message": f"Failed to check disk status: {str(e)}"
        }

async def get_service_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all services required by the API.

    Returns:
        Dict containing overall status and detailed service statuses.
    """
    _logger.debug("Health check endpoint called.")

    # Calculate API uptime
    uptime_seconds = time.time() - APP_START_TIME

    # Check each service
    redis_status = await check_redis_status()
    elevenlabs_status = await check_elevenlabs_status()
    llm_status = await check_llm_status()

    # Convert to ServiceStatusDetail models
    services = {
        "redis": ServiceStatusDetail(**redis_status),
        "elevenlabs": ServiceStatusDetail(**elevenlabs_status),
        "llm": ServiceStatusDetail(**llm_status)
    }

    # Get disk status if using local storage
    if app_config.storage.PROVIDER == "local":
        try:
            import shutil
            disk_status = await check_disk_status()
            services["disk"] = ServiceStatusDetail(**disk_status)
        except ImportError:
            # shutil might not be available in some environments
            services["disk"] = ServiceStatusDetail(
                status="unknown",
                uptime="N/A",
                response_time="N/A",
                message="Disk status check not available"
            )

    # Determine overall status based on individual service statuses
    if any(service.status == "unhealthy" for service in services.values()):
        overall_status = "unhealthy"
    elif any(service.status == "degraded" for service in services.values()):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    result = {
        "status": overall_status,
        "services": services,
        "version": app_config.VERSION,
        "environment": app_config.ENVIRONMENT,
        "uptime": uptime_seconds
    }

    _logger.debug("Health check response generated.")
    return result