"""
Client module for making API requests to workers via Redis.
This module provides lightweight functions for submitting jobs and checking results
without initializing worker resources.
"""
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple

from viralStoryGenerator.utils.redis_manager import RedisManager
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger

# Initialize Redis managers lazily
_api_redis_manager = None
_scrape_redis_manager = None

def get_api_redis_manager() -> Optional[RedisManager]:
    """Get or initialize the API Redis manager."""
    global _api_redis_manager
    if (_api_redis_manager is not None):
        return _api_redis_manager

    if not app_config.redis.ENABLED:
        return None

    try:
        _api_redis_manager = RedisManager(
            queue_name=app_config.redis.QUEUE_NAME,
            result_prefix=app_config.redis.RESULT_PREFIX,
            ttl=app_config.redis.TTL
        )
        if not _api_redis_manager.is_available():
            _logger.warning("API Redis manager initialized but Redis is unavailable.")
            _api_redis_manager = None
    except Exception as e:
        _logger.exception(f"Failed to initialize API Redis manager: {e}")
        _api_redis_manager = None

    return _api_redis_manager

def get_scrape_redis_manager() -> Optional[RedisManager]:
    """Get or initialize the Scraper Redis manager."""
    global _scrape_redis_manager
    if _scrape_redis_manager is not None:
        return _scrape_redis_manager

    if not app_config.redis.ENABLED:
        return None

    try:
        # Use scrape-specific config values
        scrape_queue_name = getattr(app_config.redis, 'SCRAPE_QUEUE_NAME', 'viralstory_scrape_queue')
        scrape_result_prefix = getattr(app_config.redis, 'SCRAPE_RESULT_PREFIX', 'viralstory_scrape_results:')
        scrape_ttl = getattr(app_config.redis, 'SCRAPE_TTL', app_config.redis.TTL)

        _scrape_redis_manager = RedisManager(
            queue_name=scrape_queue_name,
            result_prefix=scrape_result_prefix,
            ttl=scrape_ttl
        )
        if not _scrape_redis_manager.is_available():
            _logger.warning("Scraper Redis manager initialized but Redis is unavailable.")
            _scrape_redis_manager = None
    except Exception as e:
        _logger.exception(f"Failed to initialize Scraper Redis manager: {e}")
        _scrape_redis_manager = None

    return _scrape_redis_manager

async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    wait_for_result: bool = False,
    timeout: int = 300
) -> Optional[str]:
    """Queues a scraping request via Redis if manager is available."""
    manager = get_scrape_redis_manager()
    if not manager:
        _logger.warning("Redis manager for scraper not available, cannot queue scrape request.")
        return None

    job_id = str(uuid.uuid4())
    request_payload = {
        'id': job_id,
        'job_id': job_id,
        'urls': [urls] if isinstance(urls, str) else urls,
        'browser_config': browser_config_dict,
        'run_config': run_config_dict,
        'request_time': time.time()
    }

    _logger.debug(f"Client: Prepared request payload for job {job_id}")
    success = manager.add_request(request_payload)
    if not success:
        _logger.error("Failed to add scrape request to Redis queue.")
        return None

    _logger.info(f"Scrape request {job_id} queued successfully.")
    return job_id

async def get_scrape_result(request_id: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """Gets the result of a previously queued scraping request."""
    manager = get_scrape_redis_manager()
    if not manager:
        _logger.error("Redis manager for scraper not available, cannot get scrape result.")
        return None

    result_data = manager.get_result(request_id)
    _logger.debug(f"Client: get_scrape_result for {request_id}. Raw data from Redis: {result_data}")

    if not result_data:
        if manager.check_key_exists(request_id):
            _logger.debug(f"Scrape job {request_id} is pending/processing.")
        else:
            _logger.warning(f"Scrape job {request_id} not found.")
        return None

    status = result_data.get("status")
    _logger.debug(f"Client: Job {request_id} status from Redis: {status}")
    if status == "completed":
        scrape_output = result_data.get("data")
        if isinstance(scrape_output, list):
            return scrape_output
        else:
            _logger.error(f"Unexpected data format in completed scrape job {request_id}: {type(scrape_output)}")
            return None
    elif status == "failed":
        _logger.error(f"Scrape job {request_id} failed: {result_data.get('error')}")
        return None
    else:
        _logger.debug(f"Scrape job {request_id} status: {status}. Result not ready.")
        return None

def close_redis_connections():
    """Close all Redis connections."""
    global _api_redis_manager, _scrape_redis_manager

    if _api_redis_manager and hasattr(_api_redis_manager, 'close'):
        try:
            _logger.info("Closing API Redis manager connection pool...")
            _api_redis_manager.close()
        except Exception as e:
            _logger.exception(f"Error closing API Redis manager connection pool: {e}")
    _api_redis_manager = None

    if _scrape_redis_manager and hasattr(_scrape_redis_manager, 'close'):
        try:
            _logger.info("Closing Scraper Redis manager connection pool...")
            _scrape_redis_manager.close()
        except Exception as e:
            _logger.exception(f"Error closing Scraper Redis manager connection pool: {e}")
    _scrape_redis_manager = None
