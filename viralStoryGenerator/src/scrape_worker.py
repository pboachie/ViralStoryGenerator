from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Callable, Coroutine

from viralStoryGenerator.src.logger import base_app_logger as logger
from viralStoryGenerator.models.models import URLMetadata

if TYPE_CHECKING:
    from crawl4ai import AsyncWebCrawler as RealAsyncWebCrawler, BrowserConfig as RealBrowserConfig
    _ScrapeCallable = Callable[..., Coroutine[Any, Any, List[URLMetadata]]]
else:
    RealAsyncWebCrawler = Any
    RealBrowserConfig = Any
    _ScrapeCallable = Any

class MockBrowserConfig:
    def __init__(self, headless: bool = True, browser: Optional[str] = 'chromium', timeout: Optional[int] = 30, **kwargs):
        self.headless = headless
        self.browser = browser
        self.timeout = timeout
        self.kwargs = kwargs
        logger.debug(f"MockBrowserConfig initialized with headless={headless}, browser={browser}, timeout={timeout}, kwargs={kwargs}")

    def to_dict(self) -> Dict[str, Any]:
        return {"headless": self.headless, "browser": self.browser, "timeout": self.timeout, **self.kwargs}

class MockAsyncWebCrawler:
    def __init__(self, config: Optional[MockBrowserConfig] = None, playwright_context_manager: Optional[Any] = None):
        self.config = config if config else MockBrowserConfig()
        self.playwright_context_manager = playwright_context_manager
        logger.debug(f"MockAsyncWebCrawler initialized with config: {self.config.to_dict() if self.config else 'None'}")

    async def run(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        logger.debug(f"MockAsyncWebCrawler.run called with URLs: {urls}, kwargs: {kwargs}")
        mock_results = []
        for url_item in urls:
            mock_results.append({
                "url": url_item, "final_url": url_item, "status": "mock_success",
                "markdown": f"# Mock Markdown for {url_item}",
                "raw_html": f"<html><head><title>Mock Title for {url_item}</title></head><body>Mock HTML for {url_item}</body></html>",
                "metadata": {"title": f"Mock Title for {url_item}", "source": "mock_crawler_run"},
                "error_message": None,
            })
        return mock_results

    async def close(self):
        logger.debug("MockAsyncWebCrawler.close called.")
        pass

CRAWL4AI_AVAILABLE: bool
ActualAsyncWebCrawler: type
ActualBrowserConfig: type

try:
    from crawl4ai import AsyncWebCrawler as ImportedCrawl4aiCrawler, BrowserConfig as ImportedCrawl4aiConfig
    ActualAsyncWebCrawler = ImportedCrawl4aiCrawler
    ActualBrowserConfig = ImportedCrawl4aiConfig
    CRAWL4AI_AVAILABLE = True
    logger.debug("Successfully imported Crawl4AI. Using real components.")
except ImportError:
    ActualAsyncWebCrawler = MockAsyncWebCrawler # type: ignore
    ActualBrowserConfig = MockBrowserConfig   # type: ignore
    CRAWL4AI_AVAILABLE = False
    logger.debug("Crawl4AI library not found. Using mock objects for scraper components.")

_shared_crawler_instance: Optional[Union[RealAsyncWebCrawler, MockAsyncWebCrawler]] = None
_crawler_init_lock = asyncio.Lock()

from viralStoryGenerator.utils.redis_manager import RedisMessageBroker

async def _fallback_mock_scrape_urls_efficiently(
    urls_to_scrape: List[str],
    crawler_instance: Optional[Union[RealAsyncWebCrawler, MockAsyncWebCrawler]] = None,
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    dispatcher_config_dict: Optional[Dict[str, Any]] = None,
    user_query_for_bm25: Optional[str] = None,
    job_id_for_log: Optional[str] = None
) -> List[URLMetadata]:
    logger.warning(f"FALLBACK MOCKED scrape_urls_efficiently for job_id: {job_id_for_log}, URLs: {urls_to_scrape[:2]}")
    mock_metadata_list = []
    for u in urls_to_scrape:
        try:
            mock_data = {
                "url": u, "final_url": u, "status": "mock_success",
                "markdown_content": f"# Mock Markdown for {u}",
                "html_content": f"<html><head><title>Mock Title for {u}</title></head><body>Mock HTML for {u}</body></html>",
                "title": f"Mock Title for {u}",
                "metadata_payload": {"source": "fallback_mock_scrape_urls_efficiently"},
                "error": None, "language": "en", "favicon_url": "https://example.com/favicon.ico",
                "job_id": job_id_for_log
            }
            mock_metadata_list.append(URLMetadata(**mock_data))
        except Exception as ex:
            logger.error(f"Error in _fallback_mock_scrape_urls_efficiently for {u}: {ex}", exc_info=True)
    return mock_metadata_list

_scrape_callable_to_use: _ScrapeCallable

try:
    from viralStoryGenerator.utils.crawl4ai_scraper import scrape_urls_efficiently as imported_scrape_efficiently
    if not CRAWL4AI_AVAILABLE:
        logger.debug("Crawl4AI not available in worker. Using worker's fallback mock for scrape_urls_efficiently.")
        _scrape_callable_to_use = _fallback_mock_scrape_urls_efficiently # type: ignore
    else:
        _scrape_callable_to_use = imported_scrape_efficiently
        logger.debug("Using 'scrape_urls_efficiently' from viralStoryGenerator.utils.crawl4ai_scraper (worker has Crawl4AI).")
except ImportError as e:
    logger.error(f"Failed to import 'scrape_urls_efficiently' from viralStoryGenerator.utils.crawl4ai_scraper: {e}. Using worker's fallback mock function.")
    _scrape_callable_to_use = _fallback_mock_scrape_urls_efficiently # type: ignore

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
STREAM_NAME = os.environ.get("SCRAPER_STREAM_NAME", "scraper_jobs")
PROCESSING_STREAM_NAME = os.environ.get("PROCESSING_STREAM_NAME", "processing_stream")
WORKER_ID = f"scrape-worker-{os.getpid()}"

async def get_shared_crawler_instance(worker_id_log_prefix: str) -> Union[RealAsyncWebCrawler, MockAsyncWebCrawler]:
    global _shared_crawler_instance
    if _shared_crawler_instance is None:
        async with _crawler_init_lock:
            if _shared_crawler_instance is None:
                logger.debug(f"{worker_id_log_prefix}: Initializing shared crawler instance.")
                browser_timeout_seconds = int(os.environ.get("BROWSER_TIMEOUT_SECONDS", 120))

                common_config_params: Dict[str, Any] = {'headless': True}

                if CRAWL4AI_AVAILABLE:
                    real_config_params = {**common_config_params}
                    current_config_instance = ActualBrowserConfig(**real_config_params) # type: ignore[operator]
                    _shared_crawler_instance = ActualAsyncWebCrawler(config=current_config_instance) # type: ignore[operator]
                else:
                    mock_config_params = {**common_config_params, 'timeout': browser_timeout_seconds}
                    current_config_instance = ActualBrowserConfig(**mock_config_params) # type: ignore[operator]
                    _shared_crawler_instance = ActualAsyncWebCrawler(config=current_config_instance) # type: ignore[operator]

                logger.debug(f"{worker_id_log_prefix}: Shared crawler instance initialized ({'real' if CRAWL4AI_AVAILABLE else 'mock'}). Type: {type(_shared_crawler_instance).__name__}")

    if _shared_crawler_instance is None:
        raise RuntimeError("Crawler instance is unexpectedly None after initialization attempt.")
    return _shared_crawler_instance

async def close_shared_crawler_instance(worker_id_log_prefix: str):
    global _shared_crawler_instance
    if _shared_crawler_instance:
        async with _crawler_init_lock:
            if _shared_crawler_instance:
                logger.debug(f"{worker_id_log_prefix}: Closing shared crawler instance.")
                try:
                    await _shared_crawler_instance.close() # type: ignore
                except Exception as e:
                    logger.error(f"{worker_id_log_prefix}: Error closing shared crawler instance: {e}", exc_info=True)
                finally:
                    _shared_crawler_instance = None
                    logger.debug(f"{worker_id_log_prefix}: Shared crawler instance set to None.")

async def preload_components(worker_id_log_prefix: str, pre_init_crawler: bool = True):
    logger.debug(f"{worker_id_log_prefix}: Preloading components...")
    if pre_init_crawler:
        try:
            await get_shared_crawler_instance(worker_id_log_prefix)
            logger.debug(f"{worker_id_log_prefix}: Shared crawler pre-initialized successfully.")
        except Exception as e:
            logger.error(f"{worker_id_log_prefix}: Failed to pre-initialize shared crawler: {e}", exc_info=True)
    logger.debug(f"{worker_id_log_prefix}: Component preloading complete.")

async def process_single_scrape_job(
    job_data: Dict[str, Any],
    publisher_broker: RedisMessageBroker,
    worker_id_log_prefix: str
) -> bool:
    job_id = job_data.get("job_id")
    urls_to_scrape = job_data.get("urls")

    user_prompt_for_bm25 = job_data.get("user_prompt")
    browser_config_job: Optional[Dict[str, Any]] = job_data.get("browser_config_dict")
    run_config_job: Optional[Dict[str, Any]] = job_data.get("run_config_dict")
    dispatcher_config_job: Optional[Dict[str, Any]] = job_data.get("dispatcher_config_dict")

    if not job_id or not urls_to_scrape:
        logger.error(f"{worker_id_log_prefix}: Invalid job data for job_id '{job_id}': missing job_id or urls. Data: {job_data}")
        if job_id:
            await publisher_broker.track_job_progress(job_id, "failed", {"error": "Invalid job data received by worker"})
        return False

    logger.info(f"{worker_id_log_prefix}: Starting job_id: {job_id} for {len(urls_to_scrape)} URLs: {urls_to_scrape[:3]}")
    await publisher_broker.track_job_progress(job_id, "processing", {"message": f"Scrape worker picked up job {job_id}"})

    try:
        crawler = await get_shared_crawler_instance(worker_id_log_prefix)

        scraped_results_metadata: List[URLMetadata] = await _scrape_callable_to_use(
            urls_to_scrape=urls_to_scrape,
            crawler_instance=crawler,
            browser_config_dict=browser_config_job,
            run_config_dict=run_config_job,
            dispatcher_config_dict=dispatcher_config_job,
            user_query_for_bm25=user_prompt_for_bm25,
            job_id_for_log=job_id
        )

        logger.info(f"{worker_id_log_prefix}: Job {job_id} _scrape_callable_to_use completed. Results count: {len(scraped_results_metadata)}")

        results_as_dicts_for_tracking = []
        any_errors = False
        for item_meta in scraped_results_metadata:
            if not isinstance(item_meta, URLMetadata):
                logger.error(f"{worker_id_log_prefix}: Job {job_id}: Expected URLMetadata, got {type(item_meta)}. Data: {item_meta}")
                error_url = str(getattr(item_meta, 'url', 'unknown_url_type_error'))
                results_as_dicts_for_tracking.append({"url": error_url, "status": "type_error", "error": f"Expected URLMetadata, got {type(item_meta)}"})
                any_errors = True
                continue
            results_as_dicts_for_tracking.append(item_meta.model_dump(mode='json', by_alias=True))
            if item_meta.status not in ["success", "mock_success"] or item_meta.error:
                any_errors = True
                logger.warning(f"{worker_id_log_prefix}: Job {job_id}: URL {item_meta.url} processed with status: '{item_meta.status}', error: '{item_meta.error}'")

        processing_message_payload = {
            "job_id": job_id, "original_request": job_data,
            "scraped_data": results_as_dicts_for_tracking,
            "status": "completed_with_errors" if any_errors else "success"
        }

        final_status = "completed_with_errors" if any_errors else "success"
        await publisher_broker.track_job_progress(
            job_id,
            final_status,
            data={"message": f"Scraping finished by worker. Errors: {any_errors}",
                  "results_count": len(results_as_dicts_for_tracking),
                  "results": results_as_dicts_for_tracking}
        )

        await publisher_broker.publish_message(message_data=processing_message_payload, job_id=job_id)
        logger.info(f"{worker_id_log_prefix}: Job {job_id} results published to '{publisher_broker.stream_name}'. Overall status: {processing_message_payload['status']}")
        return True

    except Exception as e:
        logger.error(f"{worker_id_log_prefix}: Unhandled exception in process_single_scrape_job for job_id {job_id}: {e}", exc_info=True)
        error_payload = {"job_id": job_id, "original_request": job_data, "status": "failed", "error_message": str(e)}
        try:
            await publisher_broker.track_job_progress(job_id, "failed", {"error": str(e), "details": "Unhandled exception in process_single_scrape_job"})
            await publisher_broker.publish_message(message_data=error_payload, job_id=job_id)
            logger.info(f"{worker_id_log_prefix}: Job {job_id} failure notice published to '{publisher_broker.stream_name}'.")
        except Exception as pub_e:
            logger.error(f"{worker_id_log_prefix}: Job {job_id}: CRITICAL - Failed to publish error notice to '{publisher_broker.stream_name}': {pub_e}", exc_info=True)
        return False

async def consume_scrape_jobs(worker_id_log_prefix: str, stop_event: asyncio.Event):
    consumer_broker = RedisMessageBroker(redis_url=REDIS_URL, stream_name=STREAM_NAME)
    publisher_broker = RedisMessageBroker(redis_url=REDIS_URL, stream_name=PROCESSING_STREAM_NAME)

    consume_count = 1
    consume_block_ms = 1000
    sleep_if_no_messages_s = 0.1

    try:
        await consumer_broker.initialize()
        await publisher_broker.initialize(ensure_group=True)
        logger.debug(f"{worker_id_log_prefix}: Consumer from '{STREAM_NAME}' (group: {consumer_broker.consumer_group_name}) and Publisher to '{PROCESSING_STREAM_NAME}' initialized.")
        logger.debug(f"{worker_id_log_prefix}: Config - ConsumeCount: {consume_count}, BlockMs: {consume_block_ms}, SleepIfNoMessages: {sleep_if_no_messages_s}s")
        logger.debug(f"{worker_id_log_prefix}: Note: This worker processes scrape jobs sequentially from the stream.")

        await preload_components(worker_id_log_prefix, pre_init_crawler=True)

        while not stop_event.is_set():
            try:
                raw_messages_list = await consumer_broker.consume_messages(count=consume_count, block_ms=consume_block_ms)

                if not raw_messages_list:
                    await asyncio.sleep(sleep_if_no_messages_s)
                    continue

                for stream_id_str, message_fields_dict in raw_messages_list:
                    job_id_for_log = "unknown_job_id"
                    payload_json_str = None
                    try:
                        payload_json_str = message_fields_dict.get('payload')
                        if not payload_json_str:
                             payload_json_str = message_fields_dict.get('data')

                        if not payload_json_str:
                            logger.error(f"{worker_id_log_prefix}: Could not extract 'payload' or 'data' JSON string from message {stream_id_str}. Fields: {message_fields_dict}")
                            await consumer_broker.acknowledge_message(message_id=stream_id_str)
                            continue

                        full_payload_dict = json.loads(payload_json_str)
                        job_data = full_payload_dict.get("data")
                        job_id_from_payload = full_payload_dict.get("job_id")

                        if not job_data or not isinstance(job_data, dict) or not job_id_from_payload:
                            logger.error(f"{worker_id_log_prefix}: Malformed payload structure in message {stream_id_str}. Expected 'data' dict and 'job_id'. Payload: {full_payload_dict}")
                            await consumer_broker.acknowledge_message(message_id=stream_id_str)
                            continue

                        job_id_for_log = job_id_from_payload
                        logger.info(f"{worker_id_log_prefix}: Received job {job_id_for_log} (stream_id: {stream_id_str}). Processing...")

                        success = await process_single_scrape_job(job_data, publisher_broker, worker_id_log_prefix)

                        await consumer_broker.acknowledge_message(message_id=stream_id_str)
                        if success:
                            logger.info(f"{worker_id_log_prefix}: Job {job_id_for_log} (stream_id: {stream_id_str}) processed and ACKed.")
                        else:
                            logger.warning(f"{worker_id_log_prefix}: Job {job_id_for_log} (stream_id: {stream_id_str}) processing reported failure. Message ACKed.")

                    except json.JSONDecodeError as e:
                        logger.error(f"{worker_id_log_prefix}: Failed to decode JSON for job {job_id_for_log} (stream_id: {stream_id_str}): {e}. Raw payload string: '{payload_json_str}'", exc_info=True)
                        await consumer_broker.acknowledge_message(message_id=stream_id_str)
                    except Exception as e:
                        logger.error(f"{worker_id_log_prefix}: Error processing job {job_id_for_log} (stream_id: {stream_id_str}): {e}", exc_info=True)
                        try:
                            await consumer_broker.acknowledge_message(message_id=stream_id_str)
                        except Exception as ack_e:
                            logger.error(f"{worker_id_log_prefix}: Failed to ACK message {stream_id_str} after processing error: {ack_e}", exc_info=True)

            except (ConnectionError, TimeoutError, OSError) as e:
                is_redis_error = "redis" in str(e).lower() or isinstance(e, (ConnectionError, TimeoutError))
                if is_redis_error:
                    logger.error(f"{worker_id_log_prefix}: Redis connection/timeout error: {e}. Attempting to reconnect brokers...", exc_info=True)
                    if hasattr(consumer_broker, 'redis_manager') and hasattr(consumer_broker.redis_manager, 'close'): await consumer_broker.redis_manager.close() # type: ignore
                    if hasattr(publisher_broker, 'redis_manager') and hasattr(publisher_broker.redis_manager, 'close'): await publisher_broker.redis_manager.close() # type: ignore
                    await asyncio.sleep(5)
                    try:
                        await consumer_broker.initialize()
                        await publisher_broker.initialize()
                        logger.info(f"{worker_id_log_prefix}: Reconnected and re-initialized Redis brokers.")
                    except Exception as recon_e:
                        logger.error(f"{worker_id_log_prefix}: Failed to reconnect/re-initialize Redis brokers: {recon_e}", exc_info=True)
                        await asyncio.sleep(10)
                else:
                    logger.critical(f"{worker_id_log_prefix}: Unhandled OS error in consumer loop: {e}", exc_info=True)
                    await asyncio.sleep(5)

            except Exception as e:
                logger.critical(f"{worker_id_log_prefix}: Unhandled exception in consumer loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    except asyncio.CancelledError:
        logger.info(f"{worker_id_log_prefix}: Consumption task cancelled.")
    finally:
        logger.info(f"{worker_id_log_prefix}: Shutting down consumer and shared components...")
        await close_shared_crawler_instance(worker_id_log_prefix)
        if hasattr(consumer_broker, 'redis_manager') and hasattr(consumer_broker.redis_manager, 'close'): await consumer_broker.redis_manager.close() # type: ignore
        if hasattr(publisher_broker, 'redis_manager') and hasattr(publisher_broker.redis_manager, 'close'): await publisher_broker.redis_manager.close() # type: ignore
        logger.info(f"{worker_id_log_prefix}: Shutdown complete for {worker_id_log_prefix}.")

async def main_async():
    stop_event = asyncio.Event()
    logger.info(f"Starting {WORKER_ID}...")
    loop = asyncio.get_event_loop()

    def signal_handler_func():
        logger.info(f"{WORKER_ID}: Stop signal received. Initiating graceful shutdown...")
        stop_event.set()

    for sig_name in ('SIGINT', 'SIGTERM'):
        if hasattr(signal, sig_name):
            sig = getattr(signal, sig_name)
            try:
                loop.add_signal_handler(sig, signal_handler_func)
            except (AttributeError, NotImplementedError, ValueError, RuntimeError) as e:
                logger.warning(f"Could not set signal handler for {sig_name} on {WORKER_ID}: {e}")

    consumer_task = asyncio.create_task(consume_scrape_jobs(WORKER_ID, stop_event))

    try:
        await consumer_task
    except asyncio.CancelledError:
        logger.info(f"{WORKER_ID} main task explicitly cancelled.")
    finally:
        logger.info(f"{WORKER_ID} has shut down.")

def main():
    try:
        logger.info(f"Scrape worker ({WORKER_ID}) starting via main().")
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info(f"Scrape worker ({WORKER_ID}) process interrupted by user (KeyboardInterrupt) via main().")
    except Exception as e:
        logger.critical(f"Scrape worker ({WORKER_ID}) failed to run via main(): {e}", exc_info=True)
    finally:
        logger.info(f"Scrape worker ({WORKER_ID}) main() function finished.")

if __name__ == "__main__":
    main()