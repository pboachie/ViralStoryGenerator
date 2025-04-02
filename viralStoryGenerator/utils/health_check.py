# viralStoryGenerator/utils/health_check.py
"""Health check utilities for monitoring service dependencies."""
import shutil
import time
import os
import redis
import requests
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import asyncio

# Use dependencies from the application context
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.models import ServiceStatusDetail

#TODO:  Store the application start time (consider moving to a central app state if needed)
APP_START_TIME = time.time()
APP_USER_AGENT = f"{app_config.APP_TITLE}/{app_config.VERSION}"

# --- Helper Functions ---
def format_number(value: Union[int, float, None], decimal_places: int = 2) -> Union[float, int, str]:
    """Formats numbers, returns 'N/A' for None."""
    if value is None: return "N/A"
    if isinstance(value, (int, float)):
        return round(value, decimal_places)
    return str(value)

# --- Service Status Tracking ---
class ServiceTracker:
    """Tracks service status history with persistence."""
    def __init__(self, storage_path: Optional[str] = None):
        default_path = os.path.join(
            app_config.storage.LOCAL_STORAGE_PATH, "monitoring", "service_status.json"
        )
        self.storage_path = storage_path or default_path
        self.services: Dict[str, Any] = {}
        self.lock = asyncio.Lock()
        self._load_status_history()

    def _ensure_dir_exists(self):
        """Ensures the directory for the status file exists."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        except OSError as e:
            _logger.error(f"Failed to create monitoring directory {os.path.dirname(self.storage_path)}: {e}")

    def _load_status_history(self):
        """Loads status history from the JSON file."""
        self._ensure_dir_exists()
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.services = json.load(f)
            else: self.services = {}
        except (IOError, json.JSONDecodeError, TypeError) as e:
            _logger.error(f"Error loading service status history from {self.storage_path}: {e}")
            self.services = {}

    def _save_status_history(self):
        """Saves the current status history to the JSON file."""
        self._ensure_dir_exists()
        try:
            # Write atomically if possible (write to temp then rename)
            temp_path = self.storage_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.services, f, indent=2, default=str)
            os.replace(temp_path, self.storage_path)
        except (IOError, TypeError) as e:
            _logger.error(f"Error saving service status history to {self.storage_path}: {e}")

    async def update_service_status(self, service_name: str, status: str, response_time: Optional[float] = None):
        """Updates service status asynchronously and saves history."""
        async with self.lock:
            now = time.time()
            service_data = self.services.get(service_name)

            if not service_data:
                service_data = {
                    "first_seen": now, "last_status_change": now, "last_checked": now,
                    "status": status, "uptime_start": now if status == "healthy" else None,
                    "response_times": [], "status_history": [{"time": now, "status": status}],
                    "total_uptime": 0.0
                }
                self.services[service_name] = service_data
            else:
                service_data["last_checked"] = now
                if service_data["status"] != status:
                    if service_data["status"] == "healthy" and service_data.get("uptime_start"):
                        service_data["total_uptime"] += (now - service_data["uptime_start"])

                    service_data["last_status_change"] = now
                    service_data["status_history"].append({"time": now, "status": status})
                    service_data["status_history"] = service_data["status_history"][-100:] # Limit history

                    service_data["uptime_start"] = now if status == "healthy" else None
                    service_data["status"] = status

            if response_time is not None and isinstance(response_time, (int, float)) and response_time >= 0:
                 # Store raw float, format on retrieval
                 service_data["response_times"].append({"time": now, "value": response_time})
                 service_data["response_times"] = service_data["response_times"][-100:] # Limit history

            # Save history after update
            self._save_status_history()

    def get_service_uptime(self, service_name: str) -> Optional[float]:
        """Gets the total uptime for a service in seconds."""
        service_data = self.services.get(service_name)
        if not service_data: return None

        total_uptime = service_data.get("total_uptime", 0.0)
        if service_data["status"] == "healthy" and service_data.get("uptime_start"):
            total_uptime += (time.time() - service_data["uptime_start"])

        return total_uptime if total_uptime > 0 else None

    def get_average_response_time(self, service_name: str, samples: int = 10) -> Optional[float]:
        """Gets average response time over the last 'samples' checks."""
        service_data = self.services.get(service_name)
        if not service_data or not service_data.get("response_times"): return None

        recent_times = [
            rt["value"] for rt in service_data["response_times"][-samples:]
            if isinstance(rt.get("value"), (int, float))
        ]
        return sum(recent_times) / len(recent_times) if recent_times else None

# Initialize global service tracker
service_tracker = ServiceTracker()

# --- Individual Service Check Functions ---
async def check_redis_status() -> Dict[str, Any]:
    """Checks Redis connection and basic info."""
    service_name = "redis"
    if not app_config.redis.ENABLED:
        await service_tracker.update_service_status(service_name, "disabled")
        return {"status": "disabled", "message": "Redis is disabled in configuration."}

    start_time = time.time()
    try:
        # TODO: Consider using connection pool if checks are frequent
        redis_client = redis.Redis(
            host=app_config.redis.HOST, port=app_config.redis.PORT, db=app_config.redis.DB,
            password=app_config.redis.PASSWORD, socket_timeout=2, socket_connect_timeout=2,
            decode_responses=True
        )
        redis_client.ping()
        server_info = redis_client.info('server')
        uptime_seconds = server_info.get('uptime_in_seconds')
        response_time_ms = (time.time() - start_time) * 1000
        await service_tracker.update_service_status(service_name, "healthy", response_time_ms)
        return {
            "status": "healthy", "uptime": uptime_seconds, "response_time": response_time_ms,
            "message": "Connected successfully."
        }
    except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError) as e:
        _logger.warning(f"Redis health check failed: {e}")
        await service_tracker.update_service_status(service_name, "unhealthy")
        return {"status": "unhealthy", "message": f"Connection failed: {e}"}
    except Exception as e:
        _logger.exception(f"Unexpected error during Redis health check: {e}")
        await service_tracker.update_service_status(service_name, "unknown")
        return {"status": "unknown", "message": f"Unexpected error: {e}"}


async def check_elevenlabs_status() -> Dict[str, Any]:
    """Checks ElevenLabs API status via a lightweight endpoint."""
    service_name = "elevenlabs"
    api_key = app_config.elevenLabs.API_KEY
    if not api_key:
        await service_tracker.update_service_status(service_name, "disabled")
        return {"status": "disabled", "message": "ElevenLabs API key not configured."}

    url = "https://api.elevenlabs.io/v1/user"
    headers = {"xi-api-key": api_key, "User-Agent": APP_USER_AGENT}
    start_time = time.time()
    status, message = "unknown", "Check failed"
    response_time_ms = -1.0

    try:
        response = await asyncio.to_thread(
            requests.get, url, headers=headers, timeout=10
        )
        response_time_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            status, message = "healthy", "API is operational."
        elif response.status_code == 401:
            status, message = "degraded", "API operational but authentication failed (check API key)."
        else:
            status, message = "unhealthy", f"API returned status code {response.status_code}."
        await service_tracker.update_service_status(service_name, status, response_time_ms)

    except requests.exceptions.Timeout:
        status, message = "unhealthy", "Request timed out."
        await service_tracker.update_service_status(service_name, status)
    except requests.exceptions.RequestException as e:
        status, message = "unhealthy", f"Connection error: {e}"
        await service_tracker.update_service_status(service_name, status)
    except Exception as e:
        _logger.exception(f"Unexpected error during ElevenLabs health check: {e}")
        status, message = "unknown", f"Unexpected error: {e}"
        await service_tracker.update_service_status(service_name, status)

    return {"status": status, "response_time": response_time_ms, "message": message}


async def check_llm_status() -> Dict[str, Any]:
    """Checks the configured LLM endpoint status."""
    service_name = "llm"
    endpoint = app_config.llm.ENDPOINT
    if not endpoint:
        await service_tracker.update_service_status(service_name, "disabled")
        return {"status": "disabled", "message": "LLM endpoint not configured."}

    # Lightweight check, e.g., GET /v1/models for OpenAI-compatible APIs
    check_endpoint = endpoint
    if "/chat/completions" in endpoint:
        base_url = endpoint.split("/v1/")[0]
        check_endpoint = f"{base_url}/v1/models"
    else:
         _logger.warning(f"LLM check assumes OpenAI-compatible /v1/models endpoint based on {endpoint}.")


    headers = {"Content-Type": "application/json", "User-Agent": APP_USER_AGENT}
    # Add Auth header if needed based on LLM type (e.g., Bearer token)
    # if app_config.llm.API_KEY: headers["Authorization"] = f"Bearer {app_config.llm.API_KEY}" # Example

    start_time = time.time()
    status, message = "unknown", "Check failed"
    response_time_ms = -1.0

    try:
        # Using GET request for /models endpoint check
        response = await asyncio.to_thread(
            requests.get, check_endpoint, headers=headers, timeout=10
        )
        response_time_ms = (time.time() - start_time) * 1000

        # Consider non-server errors (4xx) as potentially 'healthy' if endpoint exists
        if response.status_code < 500:
            status, message = "healthy", "LLM service is responsive."
        else:
            status, message = "unhealthy", f"LLM service returned status code {response.status_code}."
        await service_tracker.update_service_status(service_name, status, response_time_ms)

    except requests.exceptions.Timeout:
        status, message = "unhealthy", "Request timed out."
        await service_tracker.update_service_status(service_name, status)
    except requests.exceptions.RequestException as e:
        status, message = "unhealthy", f"Connection error: {e}"
        await service_tracker.update_service_status(service_name, status)
    except Exception as e:
        _logger.exception(f"Unexpected error during LLM health check: {e}")
        status, message = "unknown", f"Unexpected error: {e}"
        await service_tracker.update_service_status(service_name, status)

    return {"status": status, "response_time": response_time_ms, "message": message}


async def check_disk_status() -> Dict[str, Any]:
    """Checks disk space for the local storage path."""
    service_name = "disk"
    if app_config.storage.PROVIDER != "local":
        await service_tracker.update_service_status(service_name, "not_applicable")
        return {"status": "not_applicable", "message": f"Disk check only applies to 'local' storage provider."}

    storage_path = app_config.storage.LOCAL_STORAGE_PATH
    status, message = "unknown", "Check failed"

    try:
        if not os.path.exists(storage_path):
             os.makedirs(storage_path, exist_ok=True)

        usage = await asyncio.to_thread(shutil.disk_usage, storage_path)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        percent_used = (usage.used / usage.total) * 100 if usage.total > 0 else 0

        # Determine status based on usage thresholds
        if percent_used > 95:
             status = "unhealthy"
             message = f"CRITICAL: Disk usage {percent_used:.1f}% ({free_gb:.2f} GB free)"
        elif percent_used > 85:
             status = "degraded"
             message = f"WARNING: Disk usage {percent_used:.1f}% ({free_gb:.2f} GB free)"
        else:
             status = "healthy"
             message = f"OK: Disk usage {percent_used:.1f}% ({free_gb:.2f} GB free)"

        await service_tracker.update_service_status(service_name, status)
        return {"status": status, "message": message, "total_gb": total_gb, "free_gb": free_gb, "percent_used": percent_used}

    except FileNotFoundError:
         status = "unhealthy"
         message = f"Storage path not found: {storage_path}"
         await service_tracker.update_service_status(service_name, status)
         return {"status": status, "message": message}
    except Exception as e:
        _logger.exception(f"Unexpected error during disk status check: {e}")
        status = "unknown"
        message = f"Unexpected error: {e}"
        await service_tracker.update_service_status(service_name, status)
        return {"status": status, "message": message}

# --- Main Status Aggregation ---
async def get_service_status() -> Dict[str, Any]:
    """Gets comprehensive status by checking all relevant services."""
    _logger.debug("Performing health checks for all services...")
    start_time = time.time()

    # Run checks concurrently
    check_tasks = {
        "redis": check_redis_status(),
        "elevenlabs": check_elevenlabs_status(),
        "llm": check_llm_status(),
        "disk": check_disk_status(),
    }
    results = await asyncio.gather(*check_tasks.values(), return_exceptions=True)
    service_results = dict(zip(check_tasks.keys(), results))

    # Process results and handle exceptions during checks
    services_final_status: Dict[str, ServiceStatusDetail] = {}
    for name, result in service_results.items():
        if isinstance(result, Exception):
            _logger.error(f"Health check for service '{name}' failed with exception: {result}")
            # Update tracker if possible, otherwise report unknown
            await service_tracker.update_service_status(name, "unknown")
            services_final_status[name] = ServiceStatusDetail(status="unknown", uptime="N/A", response_time="N/A", message=f"Check failed: {result}")
        elif isinstance(result, dict):
             # Add calculated uptime/response time from tracker
             uptime = service_tracker.get_service_uptime(name)
             avg_resp_time = service_tracker.get_average_response_time(name)

             services_final_status[name] = ServiceStatusDetail(
                 status=result.get("status", "unknown"),
                 uptime=format_number(uptime) if uptime is not None else "N/A",
                 response_time=format_number(avg_resp_time) if avg_resp_time is not None else result.get("response_time", "N/A"), # Use avg or direct check time
                 message=result.get("message", "No details")
             )
        else:
             # Should not happen with await asyncio.gather
             _logger.error(f"Unexpected result type for service '{name}' check: {type(result)}")
             services_final_status[name] = ServiceStatusDetail(status="unknown", uptime="N/A", response_time="N/A", message="Unexpected check result type")


    # Determine overall status
    overall_status = "healthy"
    has_unhealthy = any(s.status == "unhealthy" for s in services_final_status.values())
    has_degraded = any(s.status == "degraded" for s in services_final_status.values())
    has_unknown = any(s.status == "unknown" for s in services_final_status.values())

    if has_unhealthy or has_unknown:
        overall_status = "unhealthy"
    elif has_degraded:
        overall_status = "degraded"

    api_uptime = time.time() - APP_START_TIME
    duration = time.time() - start_time
    _logger.debug(f"Health check completed in {duration:.3f} seconds. Overall status: {overall_status}")

    return {
        "status": overall_status,
        "services": services_final_status,
        "version": app_config.VERSION,
        "environment": app_config.ENVIRONMENT,
        "uptime": format_number(api_uptime)
    }