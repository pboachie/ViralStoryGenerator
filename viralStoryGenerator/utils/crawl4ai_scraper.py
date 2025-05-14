# viralStoryGenerator/utils/crawl4ai_scraper.py
"""Web scraping utilities using Crawl4AI with Redis Streams message broker."""
import asyncio
import os
from typing import List, Union, Optional, Tuple, Dict, Any
from pydantic import BaseModel, HttpUrl
from urllib.parse import urljoin
import time
import json
import uuid
import playwright
import logging
from viralStoryGenerator.main import shutdown_event
from viralStoryGenerator.src.logger import base_app_logger as _logger
_logger = logging.getLogger(__name__)

# Use Crawl4AI library
try:
    from crawl4ai import (
        AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
        RateLimiter, CrawlerMonitor, DisplayMode
    )
    from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, SemaphoreDispatcher
    from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _logger.error("Crawl4AI library not found. pip install crawl4ai")
    _CRAWL4AI_AVAILABLE = False
    # Mock classes for the code to run without Crawl4AI for illustration/testing
    class BrowserConfig:
        def __init__(self, headless=True, browser="chromium", **kwargs): pass
    class CrawlerRunConfig:
        def __init__(self, screenshot=False, cache_mode="default", raw_html=False, markdown_generator=None, content_filter=None, check_robots_txt=False, **kwargs): pass
    class CacheMode:
        BYPASS = "disabled"
        ENABLED = "enabled"
    class AsyncWebCrawler:
        def __init__(self, config=None): pass
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def arun_many(self, urls, config=None, dispatcher=None):
            results = []
            for u in urls:
                class MockResult:
                    def __init__(self, url_val):
                        self.success = False
                        self.url = url_val
                        self.final_url = url_val # Mock final_url
                        self.error_message = "Crawl4AI not installed (mocked response)"
                        self.metadata = {}
                        self.screenshot = None
                        self.raw_html = None
                        self.markdown = None
                        self.dispatch_result = None # Mock dispatch_result
                        if "example.com" in url_val:
                            self.success = True
                            self.error_message = None
                            self.url = "https://example.com/" # Final URL
                            self.metadata = {'title': 'Example Domain Title from Crawl4AI'}
                            self.raw_html = """
                                <html><head><title>Example Domain HTML Title</title>
                                <meta name="description" content="This is an example description.">
                                <meta property="og:title" content="Example OG Title">
                                </head><body><h1>Hello</h1></body></html>
                            """
                            self.screenshot = "mock_base64_screenshot_data"
                            class MockMarkdown:
                                fit_markdown = "Mocked markdown content for example.com"
                                raw_markdown = "Mocked raw markdown for example.com"
                                fit_html = self.raw_html # for simplicity
                            self.markdown = MockMarkdown()
                results.append(MockResult(u))
            return results


    class PruningContentFilter:
        def __init__(self, threshold=0.48, threshold_type="fixed"): pass
    class BM25ContentFilter:
        def __init__(self, user_query=None, bm25_threshold=1.2): pass
    class DefaultMarkdownGenerator:
        def __init__(self, content_filter=None, options=None): pass # Added options
    class RateLimiter:
        def __init__(self, base_delay=(1.0,3.0), max_delay=60.0, max_retries=3, rate_limit_codes=None): pass
    class CrawlerMonitor:
        def __init__(self, max_visible_rows=15, display_mode=None): pass
    class DisplayMode:
        DETAILED = "DETAILED"
        AGGREGATED = "AGGREGATED"
    class MemoryAdaptiveDispatcher:
        def __init__(self, memory_threshold_percent=90.0, check_interval=1.0, max_session_permit=10, rate_limiter=None, monitor=None, memory_wait_timeout=300.0): pass
    class SemaphoreDispatcher:
        def __init__(self, max_session_permit=20, rate_limiter=None, monitor=None, semaphore_count=5): pass


from .redis_manager import RedisMessageBroker
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.models.models import URLMetadata
from viralStoryGenerator.utils.storage_manager import storage_manager
# Check if BeautifulSoup4 is available
try:
    from bs4 import BeautifulSoup
except ImportError:
    _logger.error("BeautifulSoup4 (bs4) is not installed. Rich HTML metadata extraction will be unavailable.")
    BeautifulSoup = None

_message_broker = None

# Function to get or initialize Redis message broker
def get_message_broker() -> Optional[RedisMessageBroker]:
    """Get or initialize Redis message broker for scraping."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    if not app_config.redis.ENABLED:
        _logger.info("Redis is disabled in config. Scraper will not use Redis Message Broker.")
        return None

    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        scrape_queue_name = getattr(app_config.redis, 'SCRAPE_QUEUE_NAME', 'vsg_scrape_jobs')
        _message_broker = RedisMessageBroker(redis_url=redis_url, stream_name=scrape_queue_name)
        _logger.info(f"Initialized Scraper RedisMessageBroker with stream: '{scrape_queue_name}'")

        try:
            consumer_group = getattr(app_config.redis, 'SCRAPER_CONSUMER_GROUP', 'scraper-workers')
            _message_broker.create_consumer_group(consumer_group)
        except Exception as e:
            _logger.warning(f"Error creating consumer group (may already exist): {e}")

        return _message_broker
    except Exception as e:
         _logger.exception(f"Failed to initialize RedisMessageBroker for scraper: {e}")
         return None

# --- Main Efficient Scraping Function ---
async def scrape_urls_efficiently(
    urls_to_scrape: List[str],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    dispatcher_config_dict: Optional[Dict[str, Any]] = None,
    user_query_for_bm25: Optional[str] = None,
    job_id_for_log: Optional[str] = None
) -> List[URLMetadata]:
    """
    Scrapes multiple URLs efficiently using Crawl4AI's arun_many with a dispatcher.
    Aims to get Markdown, raw HTML (for metadata), and screenshot in a single operation.
    """
    current_job_id_for_log = job_id_for_log if job_id_for_log else "efficient_scrape_unknown_job"
    _logger.info(f"[SCRAPE_EFFICIENTLY_ENTRY] Job {current_job_id_for_log} for {len(urls_to_scrape)} URLs. First few: {urls_to_scrape[:3]}")
    if not _CRAWL4AI_AVAILABLE:
        _logger.error(f"Job {current_job_id_for_log}: Cannot scrape URLs: Crawl4AI library is not available.")
        return [URLMetadata(url=u, error="Crawl4AI library not available.", final_url=u) for u in urls_to_scrape]

    if not urls_to_scrape:
        return []

    _logger.info(f"Starting efficient Crawl4AI scraping for {len(urls_to_scrape)} URL(s)...")

    # --- Configure Browser ---
    effective_browser_config_dict = browser_config_dict or {}
    effective_browser_config_dict.setdefault('headless', getattr(app_config.scraper, 'HEADLESS', True))
    effective_browser_config_dict.setdefault('verbose', getattr(app_config.scraper, 'VERBOSE_BROWSER', False))
    browser_cfg = BrowserConfig(**effective_browser_config_dict)

    # --- Configure Markdown Generation ---
    content_filter_for_md: Union[BM25ContentFilter, PruningContentFilter]
    if user_query_for_bm25 and user_query_for_bm25.strip():
        content_filter_for_md = BM25ContentFilter(
            user_query=user_query_for_bm25,
            bm25_threshold=getattr(app_config.scraper, 'BM25_THRESHOLD', 1.0)
        )
        _logger.debug(f"Using BM25ContentFilter for Markdown with query: {user_query_for_bm25}")
    else:
        content_filter_for_md = PruningContentFilter(
            threshold=getattr(app_config.scraper, 'PRUNING_THRESHOLD', 0.48),
            threshold_type=getattr(app_config.scraper, 'PRUNING_THRESHOLD_TYPE', "fixed")
        )
        _logger.debug("Using PruningContentFilter for Markdown.")

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter_for_md,
        options={
            "ignore_links": getattr(app_config.scraper, 'MD_IGNORE_LINKS', False),
            "base_url_for_links": None
        }
    )

    # --- Configure CrawlerRun ---
    effective_run_config_dict = run_config_dict or {}
    effective_run_config_dict.setdefault('cache_mode', getattr(CacheMode, getattr(app_config.scraper, 'CACHE_MODE', 'BYPASS').upper(), CacheMode.BYPASS))
    effective_run_config_dict.setdefault('screenshot', getattr(app_config.scraper, 'TAKE_SCREENSHOT', True))
    effective_run_config_dict.setdefault('raw_html', getattr(app_config.scraper, 'FETCH_RAW_HTML', True))
    effective_run_config_dict.setdefault('markdown_generator', markdown_generator)
    effective_run_config_dict.setdefault('check_robots_txt', getattr(app_config.scraper, 'CHECK_ROBOTS_TXT', True))
    effective_run_config_dict.setdefault('stream', False) # Stream mode is handled differently by arun_many

    run_config_for_constructor = {
        key: value for key, value in effective_run_config_dict.items() if key != 'raw_html'
    }

    run_cfg = CrawlerRunConfig(**run_config_for_constructor)

    # --- Configure Dispatcher ---
    dispatcher_type = getattr(app_config.scraper, 'DISPATCHER_TYPE', 'memory_adaptive')
    if dispatcher_config_dict and "type" in dispatcher_config_dict:
        dispatcher_type = dispatcher_config_dict["type"]

    base_dispatcher_params = {}
    if dispatcher_config_dict and "params" in dispatcher_config_dict:
        base_dispatcher_params = dispatcher_config_dict.get("params", {})


    rate_limiter_params = base_dispatcher_params.pop("rate_limiter", {})
    rate_limiter_params.setdefault('base_delay', tuple(getattr(app_config.scraper, 'RL_BASE_DELAY', [1.0, 3.0])))
    rate_limiter_params.setdefault('max_delay', getattr(app_config.scraper, 'RL_MAX_DELAY', 60.0))
    rate_limiter_params.setdefault('max_retries', getattr(app_config.scraper, 'RL_MAX_RETRIES', 3))
    rate_limiter_params.setdefault('rate_limit_codes', getattr(app_config.scraper, 'RL_CODES', [429, 503]))
    rate_limiter = RateLimiter(**rate_limiter_params)

    monitor_params = base_dispatcher_params.pop("monitor", {})
    monitor_params.setdefault('max_visible_rows', getattr(app_config.scraper, 'MONITOR_MAX_ROWS', 10))
    monitor_params.setdefault('display_mode', getattr(DisplayMode, getattr(app_config.scraper, 'MONITOR_DISPLAY_MODE', 'AGGREGATED').upper()))
    monitor = CrawlerMonitor(**monitor_params) if getattr(app_config.scraper, 'MONITOR_ENABLED', False) else None

    dispatcher: Union[MemoryAdaptiveDispatcher, SemaphoreDispatcher]
    if dispatcher_type.lower() == "semaphore":
        final_dispatcher_params = {
            'max_session_permit': getattr(app_config.scraper, 'DISP_SEMA_MAX_PERMIT', 10),
            **base_dispatcher_params
        }
        if 'semaphore_count' not in final_dispatcher_params:
             final_dispatcher_params['semaphore_count'] = final_dispatcher_params['max_session_permit']

        dispatcher = SemaphoreDispatcher(
            rate_limiter=rate_limiter,
            monitor=monitor,
            **final_dispatcher_params
        )
        _logger.info(f"Using SemaphoreDispatcher with params: {final_dispatcher_params}")
    else: # Default to MemoryAdaptiveDispatcher
        raw_mem_adaptive_params = {
            'memory_threshold_percent': getattr(app_config.scraper, 'DISP_MEM_THRESHOLD', 85.0),
            'check_interval': getattr(app_config.scraper, 'DISP_MEM_INTERVAL', 1.0),
            'max_session_permit': getattr(app_config.scraper, 'DISP_MEM_MAX_PERMIT', 5),
            **base_dispatcher_params
        }

        final_dispatcher_params = {
            key: value for key, value in raw_mem_adaptive_params.items() if key != 'memory_wait_timeout'
        }

        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=rate_limiter,
            monitor=monitor,
            **final_dispatcher_params
        )
        _logger.info(f"Using MemoryAdaptiveDispatcher with params: {final_dispatcher_params}")

    results_metadata: List[URLMetadata] = []

    try:
        _logger.info(f"[SCRAPE_EFFICIENTLY_PRE_CRAWLER] Initializing AsyncWebCrawler.")
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            _logger.info(f"[SCRAPE_EFFICIENTLY_POST_CRAWLER] AsyncWebCrawler initialized. Launching arun_many for {len(urls_to_scrape)} URLs...")

            arun_many_timeout = getattr(app_config.scraper, 'ARUN_MANY_TIMEOUT', 60.0)
            crawl_results = []

            try:
                _logger.debug(f"Calling crawler.arun_many with a timeout of {arun_many_timeout}s for URLs (first 5): {urls_to_scrape[:5]}")
                if urls_to_scrape: # Only run if there are URLs
                    crawl_results = await asyncio.wait_for(
                        crawler.arun_many(urls_to_scrape, config=run_cfg, dispatcher=dispatcher),
                        timeout=arun_many_timeout
                    )
                else:
                    _logger.info("No URLs provided to arun_many, skipping call.")
            except asyncio.TimeoutError:
                _logger.warning(f"Crawl4AI arun_many timed out after {arun_many_timeout}s for {len(urls_to_scrape)} URLs.")
                for u_original in urls_to_scrape:
                    if not any(item.url == u_original for item in results_metadata):
                        results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Scraping timed out after {arun_many_timeout}s"))
                crawl_results = []
            except playwright._impl._errors.TargetClosedError as e:
                _logger.error(f"A Playwright TargetClosedError occurred during arun_many: {e}. This might indicate premature browser closure or resource issue.")
                for u_original in urls_to_scrape:
                    if not any(item.url == u_original for item in results_metadata):
                        results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Playwright TargetClosedError: {str(e)}"))
                crawl_results = []
            except Exception as e_arun: # Catch other errors from arun_many or wait_for
                _logger.exception(f"Error during crawler.arun_many or asyncio.wait_for: {e_arun}")
                for u_original in urls_to_scrape:
                    if not any(item.url == u_original for item in results_metadata):
                        results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Scraping execution error: {str(e_arun)}"))

        log_message_arun_many = f"Crawl4AI arun_many processing finished. "
        if any(r.error and "timed out" in r.error for r in results_metadata):
            log_message_arun_many += f"Timeout occurred. "
        log_message_arun_many += f"Processing {len(crawl_results)} direct results from arun_many. Total metadata items (including prior errors/timeouts): {len(results_metadata)}."
        _logger.info(log_message_arun_many)

        for idx, result in enumerate(crawl_results):
            original_url = getattr(result, 'url', None)
            final_url_candidate = getattr(result, 'final_url', None)

            # --- Save screenshot for troubleshooting ---
            if hasattr(result, 'screenshot') and result.screenshot:
                import base64, os
                try:
                    screenshot_bytes = base64.b64decode(result.screenshot)
                    safe_job_id = str(current_job_id_for_log or 'unknown').replace('/', '_')
                    screenshot_filename = f"scrape_img_{safe_job_id}_{idx}.png"
                    store_result = storage_manager.store_file(
                        file_data=screenshot_bytes,
                        file_type="screenshot",
                        filename=screenshot_filename,
                        content_type="image/png"
                    )
                    if "error" not in store_result:
                        _logger.info(f"Saved debug screenshot for job {safe_job_id}, url idx {idx} to {store_result.get('file_path')} via {store_result.get('provider')}")
                    else:
                        _logger.error(f"Failed to save screenshot via storage_manager for job {job_id_for_log}, url idx {idx}: {store_result.get('error')}")

                except Exception as e:
                    _logger.error(f"Failed to save screenshot for job {job_id_for_log}, url idx {idx}: {e}")

            if not original_url:
                _logger.error(f"CrawlResult is missing 'url' attribute. Result: {vars(result) if hasattr(result, '__dict__') else result}. Skipping this result.")
                continue

            effective_final_url = final_url_candidate if final_url_candidate else original_url

            metadata_item = URLMetadata(url=original_url, final_url=effective_final_url)

            if result.success:
                metadata_item.screenshot_base64 = result.screenshot

                # Extract Markdown
                if result.markdown:
                    if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                        metadata_item.markdown_content = result.markdown.fit_markdown
                    elif hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
                        metadata_item.markdown_content = result.markdown.raw_markdown
                    elif isinstance(result.markdown, str):
                        metadata_item.markdown_content = result.markdown
                    else:
                        _logger.warning(f"Markdown object present for {result.url} but no known content field (fit_markdown, raw_markdown).")
                else:
                    _logger.warning(f"Crawl for {result.url} was successful but no 'markdown' attribute in result.")

                # Extract Metadata from HTML
                html_to_parse = None
                if hasattr(result, 'html') and result.html:
                    html_to_parse = result.html
                elif hasattr(result, 'raw_html') and result.raw_html:
                    html_to_parse = result.raw_html
                elif hasattr(result, 'fit_html') and result.fit_html:
                    html_to_parse = result.fit_html

                if html_to_parse:
                    metadata_item.raw_html_snippet = html_to_parse[:500] + "..."
                    if BeautifulSoup is None:
                        _logger.error("BeautifulSoup4 (bs4) is not installed. Cannot parse HTML for rich metadata.")
                        metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + "BeautifulSoup4 not installed; HTML parsing skipped."
                        if result.metadata and result.metadata.get('title') and not metadata_item.title:
                            metadata_item.title = result.metadata.get('title')
                    else:
                        try:
                            soup_obj = BeautifulSoup(html_to_parse, "html.parser")
                            bs_base_url = str(metadata_item.final_url) if metadata_item.final_url else str(result.url)
                            extracted_bs_meta = _extract_rich_metadata_from_html(soup_obj, bs_base_url)

                            if "error" in extracted_bs_meta and extracted_bs_meta["error"]:
                                 metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + extracted_bs_meta["error"]
                            else:
                                metadata_item.title = extracted_bs_meta.get("title") or metadata_item.title # Prioritize BS4
                                metadata_item.description = extracted_bs_meta.get("description")
                                metadata_item.image_url = extracted_bs_meta.get("image_url")
                                if extracted_bs_meta.get("canonical_url"):
                                    try:
                                        metadata_item.final_url = HttpUrl(extracted_bs_meta.get("canonical_url"))
                                    except Exception:
                                        _logger.warning(f"Invalid canonical URL '{extracted_bs_meta.get('canonical_url')}' for {result.url}, using Crawl4AI's final_url.")
                                metadata_item.all_meta_tags = extracted_bs_meta.get("all_meta_tags", {})
                        except Exception as e_parse:
                            _logger.error(f"Error parsing HTML for {result.url}: {e_parse}")
                            metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + f"HTML parsing error: {str(e_parse)}"
                            if result.metadata and result.metadata.get('title') and not metadata_item.title: # Fallback title
                                metadata_item.title = result.metadata.get('title')
                else:
                    _logger.warning(f"No raw_html or fit_html returned by Crawl4AI for {result.url}. Metadata extraction will be limited.")
                    current_error = metadata_item.error or ""
                    metadata_item.error = (current_error + "; " if current_error else "") + "No HTML content from Crawl4AI for metadata."
                    if result.metadata and result.metadata.get('title') and not metadata_item.title: # Fallback title
                        metadata_item.title = result.metadata.get('title')
            else:
                err_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawl failure"
                if result.status_code == 403 and run_cfg.check_robots_txt and "robots.txt" in err_msg.lower():
                     _logger.warning(f"Skipped {result.url} - blocked by robots.txt")
                     metadata_item.error = f"Skipped: Blocked by robots.txt (status: {result.status_code})"
                elif "playwright" in err_msg.lower() and ("executable doesn't exist" in err_msg.lower() or "browser server" in err_msg.lower()):
                    _logger.critical(f"Playwright browser not installed or found for {result.url}. Run 'playwright install'. Error: {err_msg}")
                    metadata_item.error = f"Playwright setup issue: {err_msg}"
                elif "Host system is missing dependencies" in err_msg:
                     _logger.critical(f"Playwright browser dependencies missing for {result.url}. Run 'playwright install-deps'. Error: {err_msg}")
                     metadata_item.error = f"Playwright dependency issue: {err_msg}"
                else:
                    _logger.warning(f"Crawl failed for {result.url}: {err_msg} (status: {result.status_code if hasattr(result, 'status_code') else 'N/A'})")
                    metadata_item.error = err_msg

            # Store dispatch result if available (for testing/debugging)
            if result.dispatch_result:
                metadata_item.dispatch_task_id = result.dispatch_result.task_id
                metadata_item.dispatch_memory_usage = result.dispatch_result.memory_usage
                if result.dispatch_result.start_time and result.dispatch_result.end_time:
                    try:
                        start_t = result.dispatch_result.start_time
                        end_t = result.dispatch_result.end_time

                        if isinstance(start_t, (float, int)) and isinstance(end_t, (float, int)):
                            metadata_item.dispatch_duration_seconds = end_t - start_t
                        elif hasattr(start_t, 'timestamp') and hasattr(end_t, 'timestamp') and callable(getattr(start_t, 'timestamp')) and callable(getattr(end_t, 'timestamp')):
                            start_ts = start_t.timestamp()
                            end_ts = end_t.timestamp()
                            metadata_item.dispatch_duration_seconds = end_ts - start_ts
                        elif hasattr(end_t - start_t, 'total_seconds'):
                            metadata_item.dispatch_duration_seconds = (end_t - start_t).total_seconds()
                        else:
                            _logger.warning(f"Job {current_job_id_for_log}: Dispatch start/end times are of unexpected types ({type(start_t)}, {type(end_t)}) or their difference is not a timedelta. Cannot calculate duration accurately.")
                            metadata_item.dispatch_duration_seconds = None
                    except Exception as e_dur:
                        _logger.error(f"Job {current_job_id_for_log}: Error calculating dispatch_duration_seconds: {e_dur}")
                        metadata_item.dispatch_duration_seconds = None

            results_metadata.append(metadata_item)

    except playwright._impl._errors.TargetClosedError as e:
        _logger.error(f"Job {current_job_id_for_log}: A Playwright TargetClosedError occurred during arun_many: {e}. This might indicate premature browser closure.")
        processed_urls = {str(item.url) for item in results_metadata}
        for u_original in urls_to_scrape:
            if u_original not in processed_urls:
                results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Scraping aborted due to TargetClosedError: {e}"))
    except Exception as e:
        error_msg = str(e)
        if "Host system is missing dependencies" in error_msg or \
           ("Executable doesn't exist" in error_msg and "playwright" in error_msg.lower()):
            _logger.critical(f"Job {current_job_id_for_log}: Critical Playwright setup issue during arun_many: {e}. Run 'playwright install' and/or 'playwright install-deps'.")
            results_metadata = [URLMetadata(url=u, final_url=u, error=f"Critical Playwright setup issue: {e}") for u in urls_to_scrape]
        else:
            _logger.exception(f"Job {current_job_id_for_log}: Unexpected error during efficient Crawl4AI execution: {e}")
            processed_urls = {str(item.url) for item in results_metadata}
            for u_original in urls_to_scrape:
                if u_original not in processed_urls:
                    results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Unexpected error during crawl: {e}"))
    finally:
        _logger.info(f"[SCRAPE_EFFICIENTLY_FINALLY] Reached finally block for scrape attempt for job {current_job_id_for_log}.")

    _logger.info(f"[SCRAPE_EFFICIENTLY_RETURN] Efficient Crawl4AI scraping finished for job {current_job_id_for_log}. Returning {len(results_metadata)} URLMetadata objects.")
    return results_metadata


# --- Extract metadata using BeautifulSoup ---
def _extract_rich_metadata_from_html(soup: Any, base_url: str) -> Dict:
    """
    Extracts rich metadata from a BeautifulSoup parsed HTML document.
    """
    if BeautifulSoup is None:
        _logger.error("BeautifulSoup4 (bs4) is not available. Cannot perform rich HTML parsing.")
        return {
            "title": None, "description": None, "image_url": None,
            "canonical_url": None, "all_meta_tags": {},
            "error": "BeautifulSoup4 (bs4) is not installed."
        }

    metadata = {
        "title": None,
        "description": None,
        "image_url": None,
        "canonical_url": None,
        "all_meta_tags": {}
    }

    # Title
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        metadata["title"] = og_title["content"].strip()
    else:
        twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
        if twitter_title and twitter_title.get("content"):
            metadata["title"] = twitter_title["content"].strip()
        elif soup.title and soup.title.string: # Fallback to HTML title tag
            metadata["title"] = soup.title.string.strip()

    # Description
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        metadata["description"] = og_desc["content"].strip()
    else:
        twitter_desc = soup.find("meta", attrs={"name": "twitter:description"})
        if twitter_desc and twitter_desc.get("content"):
            metadata["description"] = twitter_desc["content"].strip()
        else: # Fallback to meta description tag
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                metadata["description"] = meta_desc["content"].strip()

    # Image URL
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        metadata["image_url"] = urljoin(base_url, og_image["content"].strip())
    if not metadata["image_url"]:
        twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
        if twitter_image and twitter_image.get("content"):
            metadata["image_url"] = urljoin(base_url, twitter_image["content"].strip())
    if not metadata["image_url"]:
        link_image_src = soup.find("link", rel="image_src")
        if link_image_src and link_image_src.get("href"):
            metadata["image_url"] = urljoin(base_url, link_image_src["href"].strip())

    # Canonical URL
    canonical_link = soup.find("link", rel="canonical")
    if canonical_link and canonical_link.get("href"):
        metadata["canonical_url"] = urljoin(base_url, canonical_link["href"].strip())
    if not metadata["canonical_url"]:
        og_url = soup.find("meta", property="og:url")
        if og_url and og_url.get("content"):
            metadata["canonical_url"] = urljoin(base_url, og_url["content"].strip())

    # All meta tags
    for tag in soup.find_all("meta"):
        key = tag.get("name") or tag.get("property")
        value = tag.get("content")
        if key and value:
            key_str = str(key).strip()
            metadata["all_meta_tags"][key_str] = value.strip()

    # Fallback for title: try first <h1>
    if not metadata["title"]:
        h1_tag = soup.find("h1")
        if h1_tag:
            h1_text = h1_tag.string if h1_tag.string is not None else h1_tag.get_text(separator=" ", strip=True)
            if h1_text:
                 metadata["title"] = h1_text.strip()
    return metadata


async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    dispatcher_config_dict: Optional[Dict[str, Any]] = None,
    user_query_for_bm25: Optional[str] = None,
    wait_for_result: bool = False,
    timeout: int = 300 # Default timeout of 5 minutes
) -> Optional[str]:
    """
    Queues a scraping request via Redis Streams.
    The worker will use the new 'scrape_urls_efficiently' method.
    The 'job_type' parameter is removed as the efficient method gets all data.
    """
    message_broker = get_message_broker()
    if not message_broker:
        _logger.warning("Redis message broker for scraper not available, cannot queue request.")
        # todo: could fall back to direct scraping if Redis is down and not wait_for_result
        return None

    job_id = str(uuid.uuid4())

    url_list = [urls] if isinstance(urls, str) else urls
    if not url_list or not all(isinstance(u, str) for u in url_list):
        _logger.error(f"Invalid URLs provided for job {job_id}: {urls}")
        return None

    request_payload = {
        'job_id': job_id,
        'urls': url_list,
        'browser_config': browser_config_dict, # Will be serialized to JSON string by RedisMessageBroker
        'run_config': run_config_dict,         # Will be serialized to JSON string
        'dispatcher_config': dispatcher_config_dict, # Will be serialized to JSON string
        'user_query_for_bm25': user_query_for_bm25,
        'request_time': time.time(),
        'status': 'pending',
        'message_type': 'scrape_request_efficient'
    }

    _logger.debug(f"Scraper: Prepared efficient scrape request payload for job {job_id}")

    try:
        message_id = message_broker.publish_message(request_payload)
        success = message_id is not None
    except Exception as e:
        _logger.error(f"Failed to publish efficient scrape request to Redis Stream: {e}")
        success = False

    if not success:
        _logger.error(f"Failed to add efficient scrape request {job_id} to Redis Stream.")
        return None

    _logger.info(f"Efficient scrape request {job_id} queued successfully in Redis Stream.")

    if wait_for_result:
        _logger.info(f"Waiting for efficient scrape result {job_id} (timeout: {timeout}s) - Blocking operation.")
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < timeout:
            result_status_dict = message_broker.get_job_status(job_id)
            if result_status_dict and result_status_dict.get("status") in ["completed", "failed"]:
                _logger.info(f"Received final status '{result_status_dict.get('status')}' for efficient scrape request {job_id}.")
                return job_id if result_status_dict.get("status") == "completed" else None
            await asyncio.sleep(0.5)

        _logger.warning(f"Timed out waiting for efficient scrape result {job_id} after {timeout}s.")
        return None

    return job_id

async def get_scrape_result(request_id: str) -> Optional[List[URLMetadata]]:
    """
    Gets the result of a previously queued efficient scraping request.
    The result data will be a list of URLMetadata objects.
    """
    message_broker = get_message_broker()
    if not message_broker:
        _logger.error("Redis message broker for scraper not available, cannot get scrape result.")
        return None

    job_info = message_broker.get_job_status(request_id)

    if not job_info:
        _logger.warning(f"Scrape job {request_id} not found in the stream or cache.")
        return None

    status = job_info.get("status")
    _logger.debug(f"Scraper: Job {request_id} status: {status}")

    if status == "completed":
        try:
            scraped_data_raw = job_info.get("data")

            if scraped_data_raw is None:
                _logger.warning(f"Job {request_id} is marked completed but has no 'data' field. Returning empty list.")
                return []

            if isinstance(scraped_data_raw, str):
                try:
                    scraped_data_list = json.loads(scraped_data_raw)
                except json.JSONDecodeError:
                    _logger.error(f"Failed to JSON decode 'data' for job {request_id}. Raw: {scraped_data_raw[:200]}")
                    return None
            elif isinstance(scraped_data_raw, list):
                 scraped_data_list = scraped_data_raw
            else:
                _logger.error(f"Unexpected data format for completed job {request_id}. Expected list or JSON string of list, got {type(scraped_data_raw)}")
                return None

            if isinstance(scraped_data_list, list):
                if not all(isinstance(item, dict) for item in scraped_data_list):
                    _logger.error(f"Invalid item type in data list for job {request_id}. Expected all dicts.")
                    return None
                return [URLMetadata(**item) for item in scraped_data_list]
            else:
                _logger.error(f"Parsed data for job {request_id} is not a list: {type(scraped_data_list)}")
                return None

        except Exception as e:
            _logger.exception(f"Error parsing result data for job {request_id}: {e}")
            return None
    elif status == "failed":
        _logger.error(f"Scrape job {request_id} failed: {job_info.get('error')}")
        return None
    else: # pending, processing
        _logger.debug(f"Scrape job {request_id} status: {status}. Result not ready.")
        return None

async def process_scrape_queue_worker(
    batch_size: int = 1,
    sleep_interval: float = 1.0,
    max_concurrent: int = 2
) -> None:
    """Worker function to process queued scraping requests from Redis Streams using the efficient method."""
    batch_size = getattr(app_config.redis, 'WORKER_BATCH_SIZE', batch_size)
    sleep_interval = getattr(app_config.redis, 'WORKER_SLEEP_INTERVAL', sleep_interval)
    max_concurrent = getattr(app_config.scraper, 'WORKER_MAX_CONCURRENT_JOBS', getattr(app_config.redis, 'WORKER_MAX_CONCURRENT', max_concurrent))


    message_broker = get_message_broker()
    if not message_broker:
        _logger.error("Redis message broker not available. Scrape worker cannot start.")
        return
    if not _CRAWL4AI_AVAILABLE:
        _logger.error("Crawl4AI library not available. Scrape worker cannot start.")
        return

    group_name = getattr(app_config.redis, 'SCRAPER_CONSUMER_GROUP', 'scraper-workers')
    consumer_name = f"scraper-worker-eff-{uuid.uuid4().hex[:8]}"
    try:
        message_broker.create_consumer_group(group_name)
        _logger.info(f"Scraper worker {consumer_name} initialized for group: {group_name}")
    except Exception as e:
        _logger.warning(f"Error creating/ensuring consumer group '{group_name}' (may already exist): {e}")

    active_tasks = set()

    _logger.info(f"Scraper worker {consumer_name} starting main loop for group: {group_name}.")

    while True: # Main worker loop
        try:
            available_slots = max_concurrent - len(active_tasks)
            if (available_slots <= 0):
                _logger.debug(f"Max concurrency ({max_concurrent}) reached. Waiting for a task to finish.")
                done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=max(0.1, sleep_interval / 2))
                active_tasks = pending
                if done:
                    _logger.debug(f"{len(done)} task(s) completed, {len(pending)} still active.")
                if shutdown_event.is_set():
                    _logger.info(f"Scraper worker {consumer_name} received shutdown signal during task wait.")
                    break
                continue

            available_slots = max_concurrent - len(active_tasks)
            if available_slots > 0:
                messages_to_fetch = min(available_slots, batch_size)
                _logger.debug(f"Attempting to consume up to {messages_to_fetch} messages.")
                messages = await asyncio.get_running_loop().run_in_executor(
                    None,
                    message_broker.consume_messages,
                    group_name,
                    consumer_name,
                    messages_to_fetch,
                    1000
                )

                if messages:
                    _logger.info(f"{consumer_name} received {sum(len(s_msgs) for _, s_msgs in messages)} new message(s).")
                    for stream_name, stream_messages in messages:
                        for message_id_bytes, message_data_raw_bytes in stream_messages:
                            if len(active_tasks) >= max_concurrent:
                                _logger.warning(f"Max concurrency ({max_concurrent}) reached unexpectedly before processing message {message_id_bytes}. Will process in next cycle.")
                                # todo: Ideally, this message should be nack'd or left for re-delivery if possible,
                                # but current RedisMessageBroker might not support that directly.
                                break

                            message_id = message_id_bytes.decode('utf-8') if isinstance(message_id_bytes, bytes) else message_id

                            job_data = {}
                            try:
                                for k_bytes, v_bytes in message_data_raw_bytes.items():
                                    key = k_bytes.decode('utf-8') if isinstance(k_bytes, bytes) else k_bytes
                                    value = v_bytes.decode('utf-8') if isinstance(v_bytes, bytes) else v_bytes
                                    job_data[key] = value
                            except Exception as e_decode:
                                _logger.error(f"Error decoding message {message_id} data: {e_decode}. Acknowledging and skipping.")
                                message_broker.acknowledge_message(group_name, message_id)
                                continue

                            job_id = job_data.get('job_id')
                            message_id_str = message_id.decode('utf-8', 'replace') if isinstance(message_id, bytes) else str(message_id)

                            if not job_id:
                                _logger.error(f"SCRAPER_WORKER: Message {message_id_str} is missing 'job_id'. Skipping.")
                                message_broker.acknowledge_message(group_name, message_id)
                                continue

                            if job_data.get('message_type') != 'scrape_request_efficient':
                                _logger.debug(f"SCRAPER_WORKER: Skipping message {message_id_str} with type '{job_data.get('message_type')}' (expected 'scrape_request_efficient')...")
                                message_broker.acknowledge_message(group_name, message_id)
                                continue

                            _logger.debug(f"SCRAPER_WORKER: Preparing task for job {job_id}, msg {message_id_str}.")
                            task = None
                            try:
                                _logger.info(f"SCRAPER_WORKER_CREATE_TASK_ATTEMPT: Job {job_id}, Msg {message_id_str}") # Added Msg ID for clarity

                                task = asyncio.create_task(
                                    _process_single_efficient_scrape_job(
                                        job_id=job_id,
                                        message_id=message_id_str,
                                        job_data_dict=job_data,
                                        message_broker=message_broker,
                                        group_name=group_name
                                    )
                                )

                                _logger.info(f"SCRAPER_WORKER_CREATE_TASK_SUCCESS: Job {job_id}, Msg {message_id_str}. Task created: {task is not None}") # Added Msg ID

                                active_tasks.add(task)
                                task.add_done_callback(active_tasks.discard)
                                _logger.debug(f"SCRAPER_WORKER: Task added for job {job_id}, Msg {message_id_str}. Active tasks: {len(active_tasks)}")
                            except Exception as e_task_create:
                                task_repr_on_error = "ERROR_GETTING_REPR_IN_OUTER_EXCEPT"
                                try:
                                    if task:
                                        task_repr_on_error = repr(task)
                                except:
                                    pass
                                _logger.exception(f"SCRAPER_WORKER_CRITICAL_ERROR_TASK_HANDLING: Job {job_id}, Msg {message_id_str}: {e_task_create}. Task object was: {task_repr_on_error}")
                                message_broker.acknowledge_message(group_name, message_id) # Acknowledge on error here
                                continue

                        else:
                            continue
                        break
                elif not active_tasks:
                    _logger.debug(f"No messages and no active tasks. Worker {consumer_name} sleeping for {sleep_interval}s.")
                    await asyncio.sleep(sleep_interval)

            if active_tasks:
                done_proactive, pending_proactive = await asyncio.wait(active_tasks, timeout=0.001, return_when=asyncio.ALL_COMPLETED)
                if done_proactive:
                     _logger.debug(f"{len(done_proactive)} task(s) proactively cleaned up. {len(pending_proactive)} still active.")
                active_tasks = pending_proactive

        except asyncio.CancelledError:
            _logger.info(f"Scrape worker {consumer_name} main loop cancelled.")
            break
        except Exception as e:
            _logger.exception(f"Error in scrape worker {consumer_name} main loop: {e}")
            await asyncio.sleep(sleep_interval * 2)

    _logger.info(f"Scrape worker {consumer_name} loop finished. Waiting for {len(active_tasks)} active tasks to complete...")
    if active_tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=getattr(app_config.scraper, 'WORKER_SHUTDOWN_TIMEOUT', 30.0))
            _logger.info(f"All remaining scrape tasks for {consumer_name} finished or timed out.")
        except asyncio.TimeoutError:
            _logger.warning(f"Timeout waiting for {len(active_tasks)} remaining scrape tasks for {consumer_name} during shutdown.")
        except Exception as e_gather:
            _logger.exception(f"Error waiting for remaining scrape tasks for {consumer_name}: {e_gather}")
    _logger.info(f"Scrape worker {consumer_name} task cleanup complete. Exiting.")


async def _process_single_efficient_scrape_job(job_id: str, message_id: str, job_data_dict: Dict[str, Any],
                                              message_broker: RedisMessageBroker, group_name: str):
    try:
        job_data_summary = {k: v for k, v in job_data_dict.items() if k in ['urls', 'message_type', 'status', 'user_query_for_bm25']}
        _logger.debug(f"ENTERED_SCRAPE_JOB_CORO: JobID={job_id}, MsgID={message_id}, DataSummary={job_data_summary}")
    except Exception as e_early_log:
        _logger.debug(f"ENTERED_SCRAPE_JOB_CORO_EARLY_LOG_ERROR: JobID={job_id}, MsgID={message_id}, Error: {e_early_log}")

    current_task_name = "N/A"
    try:
        current_task = asyncio.current_task()
        if current_task is not None and hasattr(current_task, 'get_name'):
            current_task_name = current_task.get_name()
    except Exception as e_get_task_name:
        _logger.error(f"Error getting task name for job {job_id}, msg {message_id}: {e_get_task_name}")
        pass
    _logger.debug(f"SINGLE_JOB_CORO_STARTED: JobID={job_id}, MsgID={message_id}. Current Task Name: {current_task_name}. Task Obj: {repr(asyncio.current_task()) if asyncio.current_task() else 'N/A'}")

    try:
        _logger.debug(f"Entered _process_single_efficient_scrape_job for job_id={job_id}, message_id={message_id}")
        urls_to_scrape = job_data_dict.get('urls')
        _logger.debug(f"Job {job_id}: URLs to scrape: {urls_to_scrape}")
        def _parse_json_config(config_val: Optional[Union[str, Dict]]) -> Optional[Dict]:
            if isinstance(config_val, dict):
                return config_val
            if isinstance(config_val, str):
                try:
                    return json.loads(config_val)
                except json.JSONDecodeError:
                    _logger.warning(f"Job {job_id}: Failed to parse config string: '{config_val[:100]}...'")
                    return None
            return None

        browser_config_dict = _parse_json_config(job_data_dict.get('browser_config'))
        run_config_dict = _parse_json_config(job_data_dict.get('run_config'))
        dispatcher_config_dict = _parse_json_config(job_data_dict.get('dispatcher_config'))
        user_query_for_bm25 = job_data_dict.get('user_query_for_bm25')

        if not isinstance(urls_to_scrape, list) or not all(isinstance(u, str) for u in urls_to_scrape):
            _logger.error(f"Job {job_id} (MsgID: {message_id}): Invalid 'urls' format. Expected list of strings. Got: {type(urls_to_scrape)}. Skipping.")
            message_broker.track_job_progress(job_id, "failed", {"error": "Invalid URLs format in job data."})
            message_broker.acknowledge_message(group_name, message_id)
            return

        message_broker.track_job_progress(job_id, "processing", {
            "message": f"Starting efficient scrape for {len(urls_to_scrape)} URL(s)..."
        })
        _logger.debug(f"Job {job_id}: Starting scrape_urls_efficiently...")
        scraped_results_metadata: List[URLMetadata] = []
        final_status = "failed"
        error_message_summary = "Unknown processing error during efficient scrape execution."

        try:
            _logger.info(f"[PRE_SCRAPE_URLS_EFFICIENTLY] Job {job_id}: About to call scrape_urls_efficiently.")
            scraped_results_metadata = await scrape_urls_efficiently(
                urls_to_scrape=urls_to_scrape,
                browser_config_dict=browser_config_dict,
                run_config_dict=run_config_dict,
                dispatcher_config_dict=dispatcher_config_dict,
                user_query_for_bm25=user_query_for_bm25,
                job_id_for_log=job_id
            )
            _logger.debug(f"Job {job_id}: scrape_urls_efficiently returned {len(scraped_results_metadata)} results.")
            successful_scrapes = sum(1 for item in scraped_results_metadata if not item.error and (item.markdown_content or item.title))
            if successful_scrapes > 0:
                final_status = "completed"
                error_message_summary = f"Efficient scrape completed. {successful_scrapes}/{len(urls_to_scrape)} URLs processed successfully with content."
                _logger.info(f"Job {job_id}: {error_message_summary}")
            elif scraped_results_metadata:
                final_status = "completed"
                error_message_summary = f"Efficient scrape processed {len(urls_to_scrape)} URLs, but none yielded significant content or all had errors."
                _logger.warning(f"Job {job_id}: {error_message_summary}")
            else:
                error_message_summary = "Efficient scrape attempt returned no results. All URLs failed or critical error."
                _logger.error(f"Job {job_id}: {error_message_summary}")
        except Exception as e:
            _logger.exception(f"Critical error processing efficient scrape job {job_id}: {e}")
            error_message_summary = f"Core processing error for efficient scrape: {str(e)}"
            final_status = "failed"
            _logger.info(f"[ERROR_IN_SCRAPE_URLS_EFFICIENTLY] Job {job_id}: Error was: {error_message_summary}") # Added log
            if not scraped_results_metadata or len(scraped_results_metadata) < len(urls_to_scrape):
                processed_urls_in_results = {str(item.url) for item in scraped_results_metadata}
                updated_results_metadata = list(scraped_results_metadata)
                for u_original in urls_to_scrape:
                    if u_original not in processed_urls_in_results:
                        updated_results_metadata.append(URLMetadata(url=u_original, error=f"Job failed due to: {error_message_summary}"))
                scraped_results_metadata = updated_results_metadata
        finally:
            results_as_dicts = [item.model_dump(mode='json') for item in scraped_results_metadata]
            _logger.debug(f"Job {job_id}: Final status: {final_status}, error_message_summary: {error_message_summary}")
            message_broker.track_job_progress(job_id, final_status, {
                "message": error_message_summary,
                "error": error_message_summary if final_status == "failed" else None,
                "data": results_as_dicts
            })
            message_broker.acknowledge_message(group_name, message_id)
            _logger.info(f"Job {job_id}: Acknowledged message {message_id} (status: {final_status}).")

    except Exception as e_outer:
        _logger.exception(f"[OUTER_EXCEPTION] Outer critical error in _process_single_efficient_scrape_job for job_id={job_id}, message_id={message_id}: {e_outer}")
        try:
            if message_broker and hasattr(message_broker, 'track_job_progress') and hasattr(message_broker, 'acknowledge_message'):
                 message_broker.track_job_progress(job_id, "failed", {
                    "error": f"Outer critical error: {str(e_outer)}",
                    "message": f"Outer critical error: {str(e_outer)}"
                })
                 message_broker.acknowledge_message(group_name, message_id)
            else:
                _logger.error(f"Message broker not available to finalize/acknowledge job {job_id} after outer error.")

        except Exception as e_final_ack:
            _logger.exception(f"Error during final acknowledgment for job {job_id} after outer error: {e_final_ack}")


# --- Cleanup Function ---
def close_redis_connections():
    """Closes Redis connections managed by this module's broker instance."""
    global _message_broker
    if (_message_broker and hasattr(_message_broker, 'close')):
        try:
            _message_broker.close()
            _logger.info("Redis connection for scraper message broker closed.")
        except Exception as e:
            _logger.error(f"Error closing Redis connection for scraper: {e}")
    _message_broker = None

# Export functions
__all__ = [
    'scrape_urls_efficiently',
    'queue_scrape_request',
    'get_scrape_result',
    'get_message_broker',
    'process_scrape_queue_worker',
    'close_redis_connections',
    'URLMetadata'
]