"""
Health check utilities for ViralStoryGenerator API.
"""
import shutil
import time
import os
import redis
import requests
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.models import ServiceStatusDetail

# Store the application start time
APP_START_TIME = time.time()

# Format numbers to limit decimal places
def format_number(value: Union[int, float, None], decimal_places: int = 2) -> Union[float, int, str]:
    """Format a number with the specified number of decimal places"""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if value == int(value):  # If it's a whole number
            return int(value)
        return round(value, decimal_places)
    return value

# Dictionary to maintain service uptime tracking
class ServiceTracker:
    """Class to track and maintain service status history"""
    def __init__(self, storage_path=None):
        self.storage_path = storage_path or os.path.join(
            app_config.storage.LOCAL_STORAGE_PATH,
            "monitoring",
            "service_status.json"
        )
        self.services = {}
        self.load_status_history()

    def load_status_history(self):
        """Load service status history from file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.services = json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
                self.services = {}
        except Exception as e:
            _logger.error(f"Error loading service status history: {e}")
            self.services = {}

    def save_status_history(self):
        """Save service status history to file"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.services, f, indent=2)
        except Exception as e:
            _logger.error(f"Error saving service status history: {e}")

    def update_service_status(self, service_name: str, status: str, response_time=None):
        """Update the status of a service"""
        now = time.time()

        if service_name not in self.services:
            # First time seeing this service
            self.services[service_name] = {
                "first_seen": now,
                "last_status_change": now,
                "last_checked": now,
                "status": status,
                "uptime_start": now if status == "healthy" else None,
                "response_times": [],
                "status_history": [{"time": now, "status": status}],
                "total_uptime": 0  # Track cumulative uptime
            }
        else:
            service = self.services[service_name]

            # Update last checked time
            service["last_checked"] = now

            # If status changed, update last_status_change
            if service["status"] != status:
                # If service was previously healthy, add to total uptime
                if service["status"] == "healthy" and service["uptime_start"] is not None:
                    service["total_uptime"] += (now - service["uptime_start"])

                service["last_status_change"] = now
                service["status_history"].append({"time": now, "status": status})

                # Keep only last 100 status changes
                if len(service["status_history"]) > 100:
                    service["status_history"] = service["status_history"][-100:]

                # If service just came online, record uptime start
                if status == "healthy" and service["status"] != "healthy":
                    service["uptime_start"] = now

                # If service just went offline, clear uptime start
                if status != "healthy" and service["status"] == "healthy":
                    service["uptime_start"] = None

                service["status"] = status

            # Store response time history (keep last 100 values)
            if response_time is not None and response_time >= 0:
                service["response_times"].append({
                    "time": now,
                    "value": format_number(response_time)
                })
                # Keep only last 100 response times
                if len(service["response_times"]) > 100:
                    service["response_times"] = service["response_times"][-100:]

        # Save after each update
        self.save_status_history()

    def get_service_uptime(self, service_name):
        """Get the current uptime for a service in seconds"""
        if service_name not in self.services:
            return None

        service = self.services[service_name]

        # If service is currently healthy and we have an uptime_start
        if service["status"] == "healthy" and service["uptime_start"] is not None:
            current_uptime = time.time() - service["uptime_start"]
            total_uptime = service["total_uptime"] + current_uptime
            return total_uptime
        else:
            # Return the accumulated uptime if service is not currently healthy
            return service["total_uptime"] if service["total_uptime"] > 0 else None

    def get_average_response_time(self, service_name, samples=10):
        """Get the average response time for a service over recent samples"""
        if service_name not in self.services:
            return None

        service = self.services[service_name]

        # If we have response times to average
        if service["response_times"]:
            # Get the most recent samples (up to the number requested)
            recent_times = [r["value"] for r in service["response_times"][-samples:]]
            if recent_times:
                return format_number(sum(recent_times) / len(recent_times))

        return None

# Create a global service tracker
service_tracker = ServiceTracker()

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
        response_time = format_number(response_time)

        # Update service tracker with status
        service_tracker.update_service_status("redis", "healthy", response_time)

        return {
            "status": "healthy",
            "uptime": format_number(uptime_in_seconds),
            "response_time": response_time,
            "message": "Connected to Redis successfully."
        }
    except Exception as e:
        _logger.warning(f"Redis health check failed: {str(e)}")

        # Update service tracker with failed status
        service_tracker.update_service_status("redis", "unhealthy")

        return {
            "status": "unhealthy",
            "uptime": service_tracker.get_service_uptime("redis"),
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
        response_time = format_number(response_time)
        status = "healthy"

        if response.status_code == 200:
            message = "ElevenLabs API is operational."
            service_tracker.update_service_status("elevenlabs", "healthy", response_time)
        elif response.status_code == 401:
            status = "degraded"
            message = "ElevenLabs API is operational but authentication failed."
            service_tracker.update_service_status("elevenlabs", "degraded", response_time)
        else:
            status = "unhealthy"
            message = f"ElevenLabs API returned status code {response.status_code}."
            service_tracker.update_service_status("elevenlabs", "unhealthy", response_time)

        # Get tracked uptime for ElevenLabs
        uptime = service_tracker.get_service_uptime("elevenlabs")

        # If no uptime (service never healthy), use time since first seen
        if uptime is None and "elevenlabs" in service_tracker.services:
            now = time.time()
            first_seen = service_tracker.services["elevenlabs"].get("first_seen")
            if first_seen:
                uptime = now - first_seen

        return {
            "status": status,
            "uptime": format_number(uptime),
            "response_time": response_time,
            "message": message
        }
    except Exception as e:
        _logger.warning(f"ElevenLabs health check failed: {str(e)}")

        # Update service tracker with failed status
        service_tracker.update_service_status("elevenlabs", "unknown")

        # Try to get any accumulated uptime
        uptime = service_tracker.get_service_uptime("elevenlabs")

        # If no uptime (service never healthy), use time since first seen
        if uptime is None and "elevenlabs" in service_tracker.services:
            now = time.time()
            first_seen = service_tracker.services["elevenlabs"].get("first_seen")
            if first_seen:
                uptime = now - first_seen

        return {
            "status": "unknown",
            "uptime": format_number(uptime),
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
        response_time = format_number(response_time)

        if response.status_code < 500:  # Accept any non-server error as "working"
            # Update service tracker
            service_tracker.update_service_status("llm", "healthy", response_time)

            # Get tracked uptime for LLM
            uptime = service_tracker.get_service_uptime("llm")

            return {
                "status": "healthy",
                "uptime": format_number(uptime),
                "response_time": response_time,
                "message": "LLM service is responsive."
            }
        else:
            # Update service tracker with unhealthy status
            service_tracker.update_service_status("llm", "unhealthy", response_time)

            # Try to get any accumulated uptime
            uptime = service_tracker.get_service_uptime("llm")

            return {
                "status": "unhealthy",
                "uptime": format_number(uptime),
                "response_time": response_time,
                "message": f"LLM service returned status code {response.status_code}."
            }
    except Exception as e:
        _logger.warning(f"LLM health check failed: {str(e)}")

        # Update service tracker with failed status
        service_tracker.update_service_status("llm", "unknown")

        # Try to get any accumulated uptime
        uptime = service_tracker.get_service_uptime("llm")

        return {
            "status": "unknown",
            "uptime": format_number(uptime),
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

        # Format values with limited decimal places
        total_gb = format_number(total_gb)
        used_gb = format_number(used_gb)
        free_gb = format_number(free_gb)
        percent_used = format_number(percent_used, 1)  # One decimal for percentage

        status = "healthy"
        message = f"Disk space: {free_gb} GB free of {total_gb} GB total ({percent_used}% used)"

        # Warning thresholds
        if percent_used > 90:
            status = "unhealthy"
            message = f"CRITICAL: Only {free_gb} GB free ({percent_used}% used)"
        elif percent_used > 80:
            status = "degraded"
            message = f"WARNING: Only {free_gb} GB free ({percent_used}% used)"

        # Update service tracker
        service_tracker.update_service_status("disk", status)

        # For disk, we'll use time since first seen as uptime, even if not healthy
        disk_uptime = None
        if "disk" in service_tracker.services:
            now = time.time()
            first_seen = service_tracker.services["disk"].get("first_seen")
            if first_seen:
                disk_uptime = now - first_seen

        return {
            "status": status,
            "uptime": format_number(disk_uptime),
            "response_time": "N/A",
            "message": message
        }
    except Exception as e:
        _logger.warning(f"Disk status check failed: {str(e)}")

        # Update service tracker with failed status
        service_tracker.update_service_status("disk", "unknown")

        # For disk, we'll use time since first seen as uptime, even if not healthy
        disk_uptime = None
        if "disk" in service_tracker.services:
            now = time.time()
            first_seen = service_tracker.services["disk"].get("first_seen")
            if first_seen:
                disk_uptime = now - first_seen

        return {
            "status": "unknown",
            "uptime": format_number(disk_uptime),
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
    uptime_seconds = format_number(uptime_seconds)

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
            disk_status = await check_disk_status()
            services["disk"] = ServiceStatusDetail(**disk_status)
        except Exception as e:
            # Handle any unexpected errors
            services["disk"] = ServiceStatusDetail(
                status="unknown",
                uptime="N/A",
                response_time="N/A",
                message=f"Disk status check error: {str(e)}"
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