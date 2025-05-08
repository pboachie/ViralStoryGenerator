"""
Client utilities for ViralStoryGenerator API services
"""
import os
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union

import redis
import asyncio

from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.src.logger import logger as _logger

# Redis message broker for communication
def get_api_message_broker() -> RedisMessageBroker:
    """Get Redis message broker for API client operations"""
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    return RedisMessageBroker(redis_url=redis_url, stream_name="api_jobs")

def get_scrape_message_broker() -> RedisMessageBroker:
    """Get Redis message broker for scraper client operations"""
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    return RedisMessageBroker(redis_url=redis_url, stream_name="scraper_jobs")

async def queue_api_request(request_data: Dict[str, Any], wait_for_result: bool = False, timeout: int = 300) -> Optional[str]:
    """Queues an API request via Redis Streams"""
    message_broker = get_api_message_broker()

    job_id = request_data.get("job_id", str(uuid.uuid4()))
    if "job_id" not in request_data:
        request_data["job_id"] = job_id

    try:
        # Ensure the stream exists
        message_broker.ensure_stream_exists("api_jobs")

        # Publish message to stream
        message_id = message_broker.publish_message(request_data)
        success = message_id is not None
    except Exception as e:
        _logger.error(f"Failed to publish to Redis Stream: {e}")
        success = False

    if not success:
        _logger.error("Failed to add API request to Redis Stream.")
        return None

    _logger.info(f"API request {job_id} queued successfully in Redis Stream.")

    if wait_for_result:
        result = await wait_for_job_result(job_id, timeout=timeout)
        _logger.debug(f"API: Wait result for {job_id}: {result}")

        if result and result.get("status") == "completed":
            _logger.info(f"Received result for API request {job_id}.")
            return job_id
        else:
            _logger.warning(f"Timed out or error waiting for API result {job_id}. Status: {result.get('status') if result else 'N/A'}")
            return None

    return job_id

async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    wait_for_result: bool = False,
    timeout: int = 300
) -> Optional[str]:
    """Queues a scraping request via Redis Streams"""
    message_broker = get_scrape_message_broker()

    job_id = str(uuid.uuid4())
    request_payload = {
        'job_id': job_id,
        'urls': [urls] if isinstance(urls, str) else urls,
        'browser_config': browser_config_dict,
        'run_config': run_config_dict,
        'request_time': time.time(),
        'message_type': 'scrape_request'
    }

    _logger.debug(f"Client: Prepared request payload for job {job_id}")

    try:
        # Ensure the stream exists
        message_broker.ensure_stream_exists("scraper_jobs")

        # Publish message to stream
        message_id = message_broker.publish_message(request_payload)
        success = message_id is not None
    except Exception as e:
        _logger.error(f"Failed to publish to Redis Stream: {e}")
        success = False

    if not success:
        _logger.error("Failed to add scrape request to Redis Stream.")
        return None

    _logger.info(f"Scrape request {job_id} queued successfully in Redis Stream.")

    if wait_for_result:
        result = await wait_for_job_result(job_id, timeout=timeout, stream_name="scraper_jobs")
        # _logger.debug(f"Scraper: Wait result for {job_id}: {result}")

        if result and result.get("status") == "completed":
            _logger.info(f"Received result for scrape request {job_id}.")
            return job_id
        else:
            _logger.warning(f"Timed out or error waiting for scrape result {job_id}. Status: {result.get('status') if result else 'N/A'}")
            return None

    return job_id

async def get_scrape_result(request_id: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """Gets the result of a previously queued scraping request from Redis Stream."""
    message_broker = get_scrape_message_broker()

    # Get job status from stream
    job_status = message_broker.get_job_status(request_id)

    if not job_status:
        _logger.warning(f"No status found for scrape request {request_id}")
        return None

    if job_status.get("status") != "completed":
        _logger.debug(f"Scrape request {request_id} is not yet completed. Status: {job_status.get('status')}")
        return None

    # For completed jobs, retrieve the results
    scraped_results = job_status.get("results", [])
    if not scraped_results:
        _logger.warning(f"No results data found for completed scrape request {request_id}")
        return []

    try:
        return scraped_results
    except Exception as e:
        _logger.error(f"Error parsing scrape results for {request_id}: {e}")
        return None

async def wait_for_job_result(job_id: str, timeout: int = 300, check_interval: float = 0.5, stream_name: str = "api_jobs") -> Optional[Dict[str, Any]]:
    """
    Wait for a job result from Redis Stream with polling.
    Checks the stream for job updates until it completes or times out.
    """
    redis_url = "redis://" + app_config.redis.HOST + ":" + str(app_config.redis.PORT)
    message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=stream_name)

    start_time = time.time()
    while (time.time() - start_time) < timeout:
        # Check for job status in the stream
        job_status = message_broker.get_job_status(job_id)

        if job_status:
            status = job_status.get("status")
            if status in ["completed", "failed"]:
                return job_status

        # Wait before checking again
        await asyncio.sleep(check_interval)

    _logger.warning(f"Timeout reached while waiting for job {job_id}")
    return {
        "status": "timeout",
        "error": f"Timeout reached waiting for job result after {timeout} seconds",
        "job_id": job_id
    }

def close_redis_connections():
    """
    Close all Redis connections.
    Note: Redis connections in MessageBroker are created as needed and closed automatically.
    This function is kept for compatibility with existing code that might call it.
    """
    _logger.debug("Redis connections are managed automatically by the MessageBroker.")
    pass
