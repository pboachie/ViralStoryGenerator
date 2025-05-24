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
import signal

shutdown_event = asyncio.Event()

from viralStoryGenerator.src.logger import base_app_logger as _logger
_logger = logging.getLogger(__name__)

# Use Crawl4AI library
try:
    from crawl4ai import ( # type: ignore
        AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, # type: ignore
        RateLimiter, CrawlerMonitor, DisplayMode # type: ignore
    ) # type: ignore
    from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, SemaphoreDispatcher # type: ignore
    from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter # type: ignore
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator # type: ignore
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
                        self.dispatch_result = None
                        if "example.com" in url_val:
                            self.success = True
                            self.error_message = None
                            self.url = "https://example.com/"
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
                                fit_html = self.raw_html
                            self.markdown = MockMarkdown()
                results.append(MockResult(u))
            return results


    class PruningContentFilter:
        def __init__(self, threshold=0.48, threshold_type="fixed"): pass
    class BM25ContentFilter:
        def __init__(self, user_query=None, bm25_threshold=1.2): pass
    class DefaultMarkdownGenerator:
        def __init__(self, content_filter=None, options=None): pass
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

class _GenericPlaywrightErrorPlaceholder(Exception):
    """Placeholder for Playwright errors when specific types cannot be imported."""
    pass

try:
    from playwright.async_api import Error as AsyncPlaywrightError
    from playwright.sync_api import Error as SyncPlaywrightError
    _PLAYWRIGHT_ERRORS_IMPORTED = True
    _logger.debug("Imported specific Playwright Error classes")
except ImportError:
    _logger.warning("Could not import specific Playwright Error classes. Falling back to a generic placeholder for Playwright errors.")
    AsyncPlaywrightError = _GenericPlaywrightErrorPlaceholder
    SyncPlaywrightError = _GenericPlaywrightErrorPlaceholder
    _PLAYWRIGHT_ERRORS_IMPORTED = False

_message_broker = None

# Function to get or initialize Redis message broker
def get_message_broker() -> Optional[RedisMessageBroker]:
    """Get or initialize Redis message broker for scraping."""
    global _message_broker
    if (_message_broker is not None):
        return _message_broker

    if not app_config.redis.ENABLED:
        _logger.debug("Redis disabled in config, skipping message broker init")
        return None

    try:
        redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
        scrape_queue_name = getattr(app_config.redis, 'SCRAPE_QUEUE_NAME', 'vsg_scrape_jobs')
        _message_broker = RedisMessageBroker(
            redis_url=redis_url,
            stream_name=scrape_queue_name
        )
        _logger.debug(f"Initialized RedisMessageBroker: stream='{_message_broker.stream_name}' group='{_message_broker.consumer_group_name}'")
        return _message_broker
    except Exception as e:
         _logger.exception(f"Failed to initialize RedisMessageBroker for scraper: {e}")
         return None

# --- Main Efficient Scraping Function ---
async def scrape_urls_efficiently(
    urls_to_scrape: List[str],
    crawler_instance: Optional[AsyncWebCrawler] = None, # ADDED: Allow passing an existing crawler
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    dispatcher_config_dict: Optional[Dict[str, Any]] = None,
    user_query_for_bm25: Optional[str] = None,
    job_id_for_log: Optional[str] = None
) -> List[URLMetadata]:
    """
    Scrapes multiple URLs efficiently using Crawl4AI's arun_many with a dispatcher.
    Aims to get Markdown, raw HTML (for metadata), and screenshot in a single operation.
    Can use a pre-initialized crawler_instance or create one locally.
    """
    current_job_id_for_log = job_id_for_log if job_id_for_log else "efficient_scrape_unknown_job"
    _logger.debug(f"SCRAPE ENTRY: Job {current_job_id_for_log} scraping {len(urls_to_scrape)} URLs, existing crawler={bool(crawler_instance)}")
    if not _CRAWL4AI_AVAILABLE:
        _logger.error(f"Job {current_job_id_for_log}: Cannot scrape URLs: Crawl4AI library is not available.")
        return [URLMetadata(url=u, error="Crawl4AI library not available.", final_url=u) for u in urls_to_scrape]

    if not urls_to_scrape:
        return []

    # --- Configure Browser (used if crawler_instance is None) ---
    effective_browser_config_dict = browser_config_dict or {}
    effective_browser_config_dict.setdefault('headless', getattr(app_config.scraper, 'HEADLESS', True))
    effective_browser_config_dict.setdefault('verbose', getattr(app_config.scraper, 'VERBOSE_BROWSER', False))
    # Consider adding 'browser': getattr(app_config.scraper, 'BROWSER_TYPE', 'chromium') if applicable
    browser_cfg = BrowserConfig(**effective_browser_config_dict)

    # --- Configure Markdown Generation (remains the same) ---
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

    # --- Configure CrawlerRun (remains the same) ---
    effective_run_config_dict = run_config_dict or {}
    effective_run_config_dict.setdefault('cache_mode', getattr(CacheMode, getattr(app_config.scraper, 'CACHE_MODE', 'BYPASS').upper(), CacheMode.BYPASS))
    effective_run_config_dict.setdefault('screenshot', getattr(app_config.scraper, 'TAKE_SCREENSHOT', True))
    # raw_html is handled by crawl4ai's arun_many if the result object has it, not a direct run_cfg param for arun_many itself.
    # The original code had a special handling for 'raw_html' in run_config_for_constructor.
    # Let's ensure CrawlerRunConfig is created correctly.
    # Crawl4AI arun_many takes 'config' which is CrawlerRunConfig.
    # The 'raw_html' parameter for CrawlerRunConfig constructor was previously excluded.
    # This seems to be a detail of how CrawlerRunConfig is initialized vs. what arun_many expects or returns.
    # For now, mirroring the original logic for run_cfg creation.
    # effective_run_config_dict.setdefault('raw_html', getattr(app_config.scraper, 'FETCH_RAW_HTML', True)) # If needed
    effective_run_config_dict.setdefault('markdown_generator', markdown_generator)
    effective_run_config_dict.setdefault('check_robots_txt', getattr(app_config.scraper, 'CHECK_ROBOTS_TXT', True))
    effective_run_config_dict.setdefault('stream', False)

    # Original code created run_config_for_constructor by excluding 'raw_html'.
    # Let's assume CrawlerRunConfig doesn't take 'raw_html' in constructor but it's a conceptual flag for the operation.
    # The mock CrawlerRunConfig doesn't show 'raw_html'.
    # If 'raw_html' is a direct parameter to CrawlerRunConfig, this should be adjusted.
    # For safety, let's stick to the original filtering if it was intentional.
    run_config_params_for_constructor = {
        k: v for k, v in effective_run_config_dict.items() if k != 'raw_html' # Keep this logic if it was specific to Crawl4AI
    }
    run_cfg = CrawlerRunConfig(**run_config_params_for_constructor)


    # --- Configure Dispatcher (remains the same) ---
    dispatcher_type = getattr(app_config.scraper, 'DISPATCHER_TYPE', 'memory_adaptive')
    if dispatcher_config_dict and "type" in dispatcher_config_dict:
        dispatcher_type = dispatcher_config_dict["type"]

    base_dispatcher_params = {}
    if dispatcher_config_dict and "params" in dispatcher_config_dict:
        base_dispatcher_params = dispatcher_config_dict.get("params", {})

    rate_limiter_params = base_dispatcher_params.pop("rate_limiter", {})
    rate_limiter_params.setdefault('base_delay', tuple(getattr(app_config.scraper, 'RL_BASE_DELAY', [1.0, 3.0])) )
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
        dispatcher = SemaphoreDispatcher(rate_limiter=rate_limiter, monitor=monitor, **final_dispatcher_params)
        _logger.debug(f"SemaphoreDispatcher params: {final_dispatcher_params}")
    else: # Default to MemoryAdaptiveDispatcher
        raw_mem_adaptive_params = {
            'memory_threshold_percent': getattr(app_config.scraper, 'DISP_MEM_THRESHOLD', 85.0),
            'check_interval': getattr(app_config.scraper, 'DISP_MEM_INTERVAL', 1.0),
            'max_session_permit': getattr(app_config.scraper, 'DISP_MEM_MAX_PERMIT', 5),
            **base_dispatcher_params
        }
        final_dispatcher_params = { k: v for k, v in raw_mem_adaptive_params.items() if k != 'memory_wait_timeout'}
        dispatcher = MemoryAdaptiveDispatcher(rate_limiter=rate_limiter, monitor=monitor, **final_dispatcher_params)
        _logger.debug(f"MemoryAdaptiveDispatcher params: {final_dispatcher_params}")

    results_metadata: List[URLMetadata] = []
    crawl_results: List[Any] = []

    active_crawler: Optional[AsyncWebCrawler] = None # To hold the crawler being used

    try:
        arun_many_timeout = getattr(app_config.scraper, 'ARUN_MANY_TIMEOUT', 60.0)

        if crawler_instance:
            _logger.info(f"[SCRAPE_EFFICIENTLY_USE_EXISTING_CRAWLER] Job {current_job_id_for_log}: Using provided AsyncWebCrawler instance.")
            active_crawler = crawler_instance
            # No async with block here, lifecycle managed externally
            try:
                if urls_to_scrape:
                    _logger.debug(f"Job {current_job_id_for_log}: Calling existing_crawler.arun_many with timeout {arun_many_timeout}s for URLs: {urls_to_scrape[:5]}")
                    crawl_results = await asyncio.wait_for(
                        active_crawler.arun_many(urls_to_scrape, config=run_cfg, dispatcher=dispatcher),
                        timeout=arun_many_timeout
                    )
                else:
                    _logger.info(f"Job {current_job_id_for_log}: No URLs provided to arun_many, skipping call.")
                    crawl_results = []
            except asyncio.TimeoutError:
                _logger.warning(f"Job {current_job_id_for_log}: Crawl4AI arun_many (using existing crawler) timed out after {arun_many_timeout}s for {len(urls_to_scrape)} URLs.")
                # ... (error handling as below)
                for u_original in urls_to_scrape:
                    if not any(item.url == u_original for item in results_metadata): # Avoid duplicates
                        results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Scraping timed out after {arun_many_timeout}s", status="error"))
                crawl_results = [] # Ensure crawl_results is empty on timeout
            except (AsyncPlaywrightError, SyncPlaywrightError) as e_playwright:
                _logger.error(f"Job {current_job_id_for_log}: A Playwright error occurred during arun_many (using existing crawler): {e_playwright}.")
                for u_original in urls_to_scrape:
                    if not any(item.url == u_original for item in results_metadata):
                        results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Playwright error: {str(e_playwright)}", status="error"))
                crawl_results = []
            except Exception as e_arun:
                _logger.error(f"Job {current_job_id_for_log}: An unexpected error occurred during arun_many (using existing crawler): {e_arun} (Type: {type(e_arun)}).")
                for u_original in urls_to_scrape:
                    if not any(item.url == u_original for item in results_metadata):
                         results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"arun_many error: {str(e_arun)}", status="error"))
                crawl_results = []

        else: # Original path: create crawler locally
            _logger.info(f"[SCRAPE_EFFICIENTLY_CREATE_CRAWLER] Job {current_job_id_for_log}: Initializing new AsyncWebCrawler locally.")
            async with AsyncWebCrawler(config=browser_cfg) as local_crawler:
                active_crawler = local_crawler
                _logger.info(f"[SCRAPE_EFFICIENTLY_POST_CRAWLER] Job {current_job_id_for_log}: New AsyncWebCrawler initialized. Launching arun_many for {len(urls_to_scrape)} URLs...")
                try:
                    if urls_to_scrape:
                        _logger.debug(f"Job {current_job_id_for_log}: Calling new_crawler.arun_many with timeout {arun_many_timeout}s for URLs: {urls_to_scrape[:5]}")
                        crawl_results = await asyncio.wait_for(
                            active_crawler.arun_many(urls_to_scrape, config=run_cfg, dispatcher=dispatcher),
                            timeout=arun_many_timeout
                        )
                    else:
                        _logger.info(f"Job {current_job_id_for_log}: No URLs provided to arun_many, skipping call.")
                        crawl_results = []
                except asyncio.TimeoutError:
                    _logger.warning(f"Job {current_job_id_for_log}: Crawl4AI arun_many (new crawler) timed out after {arun_many_timeout}s for {len(urls_to_scrape)} URLs.")
                    for u_original in urls_to_scrape:
                        if not any(item.url == u_original for item in results_metadata):
                            results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Scraping timed out after {arun_many_timeout}s", status="error"))
                    crawl_results = []
                except (AsyncPlaywrightError, SyncPlaywrightError) as e_playwright:
                    _logger.error(f"Job {current_job_id_for_log}: A Playwright error occurred during arun_many (new crawler): {e_playwright}.")
                    for u_original in urls_to_scrape:
                        if not any(item.url == u_original for item in results_metadata):
                            results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Playwright error: {str(e_playwright)}", status="error"))
                    crawl_results = []
                except Exception as e_arun:
                    _logger.error(f"Job {current_job_id_for_log}: An unexpected error occurred during arun_many (new crawler): {e_arun} (Type: {type(e_arun)}).")
                    for u_original in urls_to_scrape:
                        if not any(item.url == u_original for item in results_metadata):
                             results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"arun_many error: {str(e_arun)}", status="error"))
                    crawl_results = []

        # Log after arun_many attempt, regardless of which path was taken
        log_message_arun_many = f"Job {current_job_id_for_log}: Crawl4AI arun_many processing finished. "
        if any(r.error and "timed out" in r.error for r in results_metadata): # Check existing metadata for timeouts
            log_message_arun_many += f"Timeout occurred. "
        log_message_arun_many += f"Processing {len(crawl_results)} direct results from arun_many. Total metadata items (including prior errors/timeouts): {len(results_metadata)}."
        _logger.debug(log_message_arun_many)

        # --- Process results (remains largely the same, ensure it uses original_url correctly) ---
        processed_urls_from_crawl_results = set()

        for idx, result in enumerate(crawl_results):
            original_url = getattr(result, 'url', None)
            if original_url:
                processed_urls_from_crawl_results.add(original_url)
            else: # If result.url is None, try to find a match in urls_to_scrape if count is 1
                if len(urls_to_scrape) == 1 and len(crawl_results) == 1:
                    original_url = urls_to_scrape[0]
                    _logger.warning(f"Job {current_job_id_for_log}: CrawlResult missing 'url', but only one URL was scraped. Assuming result is for: {original_url}")
                else:
                    _logger.error(f"Job {current_job_id_for_log}: CrawlResult is missing 'url' attribute. Result: {vars(result) if hasattr(result, '__dict__') else result}. Skipping this result.")
                    # Add a placeholder error if we can't associate it with an original URL
                    # This case should be rare if Crawl4AI behaves well.
                    continue # Skip if no original_url can be determined

            # Check if this URL already has an error entry (e.g. from timeout)
            existing_meta_for_url = next((m for m in results_metadata if m.url == original_url), None)
            if existing_meta_for_url and existing_meta_for_url.status == "error":
                _logger.warning(f"Job {current_job_id_for_log}: URL {original_url} already has an error status. Skipping result processing for it.")
                continue


            final_url_candidate = getattr(result, 'final_url', None)
            effective_final_url = final_url_candidate if final_url_candidate else original_url

            # Create or update metadata item
            if existing_meta_for_url:
                metadata_item = existing_meta_for_url # Update existing item
                metadata_item.final_url = effective_final_url # Update final_url if needed
                metadata_item.status = "processing_result" # Reset status if it was e.g. pending
            else:
                metadata_item = URLMetadata(url=original_url, final_url=effective_final_url, status="processing_result")


            # ... (rest of the result processing logic: screenshot, markdown, HTML metadata)
            # Ensure job_id_for_log is used in screenshot saving
            if hasattr(result, 'screenshot') and result.screenshot:
                import base64 # Keep import local if only used here
                try:
                    screenshot_bytes = base64.b64decode(result.screenshot)
                    safe_job_id = str(current_job_id_for_log or 'unknown').replace('/', '_')
                    screenshot_filename = f"scrape_img_{safe_job_id}_{idx}_{uuid.uuid4().hex[:4]}.png" # Added UUID for more uniqueness
                    store_result = storage_manager.store_file(
                        file_data=screenshot_bytes,
                        file_type="screenshot",
                        filename=screenshot_filename,
                        content_type="image/png"
                    )
                    if "error" not in store_result:
                        _logger.info(f"Job {current_job_id_for_log}: Saved debug screenshot for url {original_url} to {store_result.get('file_path')} via {store_result.get('provider')}")
                        metadata_item.screenshot_url = store_result.get('url') # Store the accessible URL if available
                    else:
                        _logger.error(f"Job {current_job_id_for_log}: Failed to save screenshot for url {original_url}: {store_result.get('error')}")
                except Exception as e_ss:
                    _logger.error(f"Job {current_job_id_for_log}: Failed to save screenshot for url {original_url}: {e_ss}")


            if result.success:
                metadata_item.status = "success" # Mark as success
                # metadata_item.screenshot_base64 = result.screenshot # Storing raw base64 can be large, prefer URL if saved

                if result.markdown:
                    # ... (markdown extraction logic as before)
                    if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                        metadata_item.markdown_content = result.markdown.fit_markdown
                    elif hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
                        metadata_item.markdown_content = result.markdown.raw_markdown
                    elif isinstance(result.markdown, str):
                        metadata_item.markdown_content = result.markdown
                    else:
                        _logger.warning(f"Job {current_job_id_for_log}: Markdown object present for {result.url} but no known content field.")
                else:
                    _logger.warning(f"Job {current_job_id_for_log}: Crawl for {result.url} was successful but no 'markdown' attribute in result.")

                html_to_parse = None
                # ... (HTML extraction logic as before)
                if hasattr(result, 'html') and result.html: html_to_parse = result.html
                elif hasattr(result, 'raw_html') and result.raw_html: html_to_parse = result.raw_html
                elif hasattr(result, 'fit_html') and result.fit_html: html_to_parse = result.fit_html

                if html_to_parse:
                    metadata_item.raw_html_snippet = html_to_parse[:500] + "..."
                    if BeautifulSoup is None:
                        # ... (BeautifulSoup not installed logic)
                        metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + "BeautifulSoup4 not installed; HTML parsing skipped."
                    else:
                        try:
                            soup_obj = BeautifulSoup(html_to_parse, "html.parser")
                            bs_base_url = str(metadata_item.final_url) if metadata_item.final_url else str(result.url)
                            extracted_bs_meta = _extract_rich_metadata_from_html(soup_obj, bs_base_url)
                            # ... (apply extracted_bs_meta to metadata_item as before)
                            if "error" in extracted_bs_meta and extracted_bs_meta["error"]:
                                 metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + extracted_bs_meta["error"]
                            else:
                                metadata_item.title = extracted_bs_meta.get("title") or metadata_item.title
                                metadata_item.description = extracted_bs_meta.get("description")
                                metadata_item.image_url = extracted_bs_meta.get("image_url")
                                if extracted_bs_meta.get("canonical_url"):
                                    try:
                                        canonical_url_str = str(extracted_bs_meta.get("canonical_url"))
                                        if canonical_url_str: metadata_item.final_url = str(HttpUrl(canonical_url_str))
                                    except Exception: pass # Ignore HttpUrl validation error for assignment
                                metadata_item.language = extracted_bs_meta.get("language")
                                metadata_item.favicon_url = extracted_bs_meta.get("favicon_url")
                        except Exception as e_bs:
                            _logger.error(f"Job {current_job_id_for_log}: BeautifulSoup parsing error for {result.url}: {e_bs}")
                            metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + f"HTML parsing error: {e_bs}"
                else: # No HTML content from crawl result
                    _logger.warning(f"Job {current_job_id_for_log}: No HTML content (raw_html, html, fit_html) found in successful result for {result.url} to parse for metadata.")
                    # If Crawl4AI provides metadata directly, use it as a fallback
                    if hasattr(result, 'metadata') and result.metadata:
                         if not metadata_item.title and result.metadata.get('title'):
                             metadata_item.title = result.metadata.get('title')
                         # Potentially map other fields if available and not parsed by BS4
                         _logger.debug(f"Job {current_job_id_for_log}: Using direct metadata from crawl result for {result.url}: {result.metadata}")


            else: # result.success is False
                metadata_item.status = "error"
                error_msg = getattr(result, 'error_message', 'Unknown error during scraping')
                _logger.warning(f"Job {current_job_id_for_log}: Scraping failed for URL {original_url}. Error: {error_msg}")
                metadata_item.error = (metadata_item.error + "; " if metadata_item.error else "") + error_msg

            if not existing_meta_for_url: # Add if it's a new item
                results_metadata.append(metadata_item)
            # If it was an existing item, it's already in results_metadata and has been updated.

        # Ensure all originally requested URLs have a metadata entry, even if arun_many didn't return them (e.g. total failure)
        all_requested_urls_set = set(urls_to_scrape) # This is already set[str]
        # Ensure items in final_metadata_urls are strings for comparison
        final_metadata_urls = {str(item.url) for item in results_metadata}
        missing_urls = all_requested_urls_set - final_metadata_urls

        for missing_url in missing_urls:
            _logger.warning(f"Job {current_job_id_for_log}: URL {missing_url} was in the request but not found in crawl results or pre-errors. Adding error entry.")
            results_metadata.append(URLMetadata(url=missing_url, final_url=missing_url, error="URL not processed by crawler or processing failed silently.", status="error"))

    except Exception as e_outer:
        _logger.exception(f"Job {current_job_id_for_log}: An unexpected outer error in scrape_urls_efficiently: {e_outer}")
        # Ensure all URLs get an error entry if a catastrophic failure occurs at this level
        existing_urls_in_meta = {str(item.url) for item in results_metadata} # Ensure string comparison here too
        for u_original in urls_to_scrape:
            if u_original not in existing_urls_in_meta:
                results_metadata.append(URLMetadata(url=u_original, final_url=u_original, error=f"Outer exception in scraper: {str(e_outer)}", status="error"))

    _logger.debug(f"SCRAPE EXIT: Job {current_job_id_for_log} returning {len(results_metadata)} items")
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


class ScraperJobFailedException(Exception):
    pass

async def queue_scrape_request(
    urls: Union[str, List[str]],
    browser_config_dict: Optional[Dict[str, Any]] = None,
    run_config_dict: Optional[Dict[str, Any]] = None,
    dispatcher_config_dict: Optional[Dict[str, Any]] = None,
    user_query_for_bm25: Optional[str] = None,
    wait_for_result: bool = False,
    timeout: int = 300,
    polling_interval: float = 1.0,
) -> Optional[str]:
    """
    Queues a scraping request via Redis Streams.
    The worker will use the new 'scrape_urls_efficiently' method.
    The 'job_type' parameter is removed as the efficient method gets all data.
    """
    message_broker = get_message_broker()
    if not message_broker:
        _logger.warning("Redis message broker for scraper not available, cannot queue request.")
        return None

    job_id = str(uuid.uuid4()) # Generate job_id early for logging

    if not message_broker._initialized:
        try:
            _logger.info(f"Initializing RedisMessageBroker for queue_scrape_request (job_id: {job_id}).")
            await message_broker.initialize()
            _logger.info(f"RedisMessageBroker initialized successfully for queue_scrape_request (job_id: {job_id}).")
        except Exception as e_init:
            _logger.error(f"Failed to initialize RedisMessageBroker in queue_scrape_request for job {job_id}: {e_init}")
            return None

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
        # MODIFIED: Ensure publish_message is awaited
        message_id = await message_broker.publish_message(request_payload, job_id=job_id)
        success = message_id is not None
    except Exception as e:
        _logger.error(f"Failed to publish efficient scrape request to Redis Stream for job {job_id}: {e}")
        success = False

    if not success:
        _logger.error(f"Failed to add efficient scrape request {job_id} to Redis Stream.")
        return None

    _logger.debug(f"Efficient scrape request {job_id} queued")

    if wait_for_result:
        _logger.debug(f"Waiting up to {timeout}s for job {job_id} to complete")
        start_time = time.time()
        while time.time() - start_time < timeout:
            job_info_value = await message_broker.get_job_progress(job_id)
            if job_info_value:
                status = job_info_value.get("status")
                _logger.debug(f"Scraper: Polling job {job_id}, current status: {status}")
                if status in ["completed", "success", "completed_with_errors"]: # MODIFIED
                    _logger.info(f"Scraper: Job {job_id} finished with status: {status}.") # MODIFIED
                    # If "completed_with_errors", it's still a success from the perspective of the queue,
                    # as the job itself has finished processing. Specific error details would be in the result.
                    return job_id
                elif status == "failed":
                    error_details = job_info_value.get("error", "Unknown error")
                    _logger.error(f"Scraper: Job {job_id} failed. Error: {error_details}")
                    raise ScraperJobFailedException(f"Scraper job {job_id} failed: {error_details}")

            await asyncio.sleep(polling_interval)
        _logger.warning(f"Scraper: Timeout waiting for job {job_id} to complete.")
        raise TimeoutError(f"Timeout waiting for job {job_id} to complete after {timeout} seconds.")
    else:
        _logger.debug(f"Job {job_id} queued without waiting")
        return job_id

async def get_scrape_result(request_id: str) -> Optional[List[URLMetadata]]:
    """Retrieves the result of a previously queued scraping job."""
    message_broker = get_message_broker()
    if not message_broker:
        _logger.warning(f"Redis message broker for scraper not available, cannot get result for request {request_id}.")
        return None

    if not message_broker._initialized:
        try:
            _logger.info(f"Initializing RedisMessageBroker for get_scrape_result (request_id: {request_id}).")
            await message_broker.initialize()
            _logger.info(f"RedisMessageBroker initialized successfully for get_scrape_result (request_id: {request_id}).")
        except Exception as e_init:
            _logger.error(f"Failed to initialize RedisMessageBroker in get_scrape_result for request {request_id}: {e_init}")
            return None

    if not request_id:
        _logger.error("request_id is required to get scrape result.")
        return None

    job_info_value = await message_broker.get_job_progress(request_id)

    if not job_info_value:
        _logger.warning(f"Scraper: No job info found for request ID {request_id} via get_job_progress.")
        return None

    status = job_info_value.get("status")
    _logger.debug(f"Scraper: Job {request_id} status: {status}")

    if status in ["completed", "success", "completed_with_errors"]: # MODIFIED
        try:
            scraped_data_raw = job_info_value.get("data")

            if scraped_data_raw is None:
                _logger.warning(f"Job {request_id} is marked completed but has no 'data' field. Returning empty list.")
                return []

            if isinstance(scraped_data_raw, str):
                try:
                    # Attempt to parse the string as JSON
                    parsed_json = json.loads(scraped_data_raw)
                    if isinstance(parsed_json, dict) and "results" in parsed_json and isinstance(parsed_json["results"], list):
                        scraped_data_list = parsed_json["results"]
                        _logger.debug(f"Job {request_id}: Extracted 'results' list from JSON string data.")
                    elif isinstance(parsed_json, list):
                        scraped_data_list = parsed_json # The string itself was a JSON list
                        _logger.debug(f"Job {request_id}: Parsed data directly as a list from JSON string.")
                    else:
                        _logger.error(f"Job {request_id}: JSON string data was not a list or a dict with a 'results' list. Parsed: {type(parsed_json)}")
                        return None
                except json.JSONDecodeError:
                    _logger.error(f"Failed to JSON decode 'data' for job {request_id}. Raw: {scraped_data_raw[:200]}")
                    return None
            elif isinstance(scraped_data_raw, dict):
                # If it's already a dictionary, check for the 'results' key
                if "results" in scraped_data_raw and isinstance(scraped_data_raw["results"], list):
                    scraped_data_list = scraped_data_raw["results"]
                    _logger.debug(f"Job {request_id}: Extracted 'results' list from dictionary data.")
                else:
                    _logger.error(f"Job {request_id}: Data is a dict but missing 'results' list. Keys: {list(scraped_data_raw.keys())}")
                    return None
            elif isinstance(scraped_data_raw, list):
                 scraped_data_list = scraped_data_raw
                 _logger.debug(f"Job {request_id}: Data was already a list.")
            else:
                _logger.error(f"Unexpected data format for completed job {request_id}. Expected list, JSON string of list, or dict with 'results' list. Got {type(scraped_data_raw)}")
                return None

            if isinstance(scraped_data_list, list):
                # Ensure all items in the list are dictionaries before creating URLMetadata objects
                if not all(isinstance(item, dict) for item in scraped_data_list):
                    _logger.error(f"Invalid item type in data list for job {request_id}. Expected all dicts. Types found: {[type(i) for i in scraped_data_list]}")
                    # Attempt to serialize if they are Pydantic models or similar
                    processed_list = []
                    for item in scraped_data_list:
                        if isinstance(item, dict):
                            processed_list.append(item)
                        elif hasattr(item, 'model_dump') and callable(getattr(item, 'model_dump')):
                            processed_list.append(item.model_dump(mode='json'))
                        elif hasattr(item, '__dict__'): # Fallback for simple objects
                            processed_list.append(vars(item))
                        else:
                            _logger.error(f"Job {request_id}: Cannot serialize item of type {type(item)} in results list.")
                            # Add a placeholder or skip, depending on desired behavior
                            processed_list.append({"error": f"Unserializable item type: {type(item)}", "original_item_repr": repr(item)[:100]})
                    if not all(isinstance(item, dict) for item in processed_list):
                        _logger.error(f"Job {request_id}: Still have non-dict items after attempting serialization in results list.")
                        return None
                    scraped_data_list = processed_list

                return [URLMetadata(**item) for item in scraped_data_list]
            else:
                _logger.error(f"Processed data for job {request_id} is not a list: {type(scraped_data_list)}")
                return None

        except Exception as e:
            _logger.exception(f"Error parsing result data for job {request_id}: {e}")
            return None
    elif status == "failed":
        _logger.error(f"Scrape job {request_id} failed: {job_info_value.get('error')}")
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
    message_broker = get_message_broker()
    consumer_name = f"scraper-worker-unknown-{uuid.uuid4().hex[:4]}" # Initialize with a default
    active_tasks: set = set() # Initialize as an empty set

    if not message_broker:
        _logger.error("Redis message broker not available. Scrape worker cannot start.")
        return

    try:
        _logger.info(f"Initializing RedisMessageBroker for process_scrape_queue_worker.")
        await message_broker.initialize()
        _logger.info(f"RedisMessageBroker initialized successfully for process_scrape_queue_worker.")

        if not _CRAWL4AI_AVAILABLE:
            _logger.error("Crawl4AI library not available. Scrape worker cannot start.")
            return

        group_name = getattr(app_config.redis, 'SCRAPER_CONSUMER_GROUP', 'scraper-workers')
        consumer_name = f"scraper-worker-eff-{uuid.uuid4().hex[:8]}" # Refine consumer_name
        _logger.info(f"Scraper worker {consumer_name} starting main loop for group: {group_name}.")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_async_shutdown(s)))
            except (ValueError, RuntimeError) as e:
                _logger.warning(f"Could not add signal handler for {sig} to loop: {e}")
                signal.signal(sig, lambda s, f: handle_sync_shutdown(s))

        while True:
            if shutdown_event.is_set():
                _logger.info(f"Shutdown event detected in main loop of {consumer_name}. Breaking.")
                break

            available_slots = max_concurrent - len(active_tasks)
            if available_slots <= 0:
                _logger.debug(f"{consumer_name}: Max concurrency ({max_concurrent}) reached. Waiting for a task to finish.")
                if not active_tasks:
                    await asyncio.sleep(max(0.1, sleep_interval / 2.0))
                    continue

                done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=max(0.1, sleep_interval / 2.0))

                for task in done:
                    try:
                        await task
                    except Exception as task_exc:
                        _logger.error(f"{consumer_name}: Scraper task raised an unhandled exception: {task_exc}", exc_info=True)
                active_tasks = pending

                if shutdown_event.is_set():
                    _logger.info(f"Shutdown event detected after task wait in {consumer_name}. Breaking.")
                    break
                continue

            messages_to_fetch = min(available_slots, batch_size)
            _logger.debug(f"{consumer_name}: Attempting to consume up to {messages_to_fetch} messages from group '{group_name}'.")

            consumed_messages_list: List[Dict[str, Any]] = []
            consumed_messages_list = await message_broker.consume_messages(
                count=messages_to_fetch,
                block_ms=int(sleep_interval * 1000) if sleep_interval > 0 else 100
            )

            if not consumed_messages_list:
                if shutdown_event.is_set():
                    _logger.info(f"Shutdown event set, scraper worker {consumer_name} stopping consumption.")
                    break
                await asyncio.sleep(sleep_interval)
                continue

            _logger.info(f"{consumer_name}: Retrieved {len(consumed_messages_list)} messages from scrape queue.")

            for message_content in consumed_messages_list:
                message_id_from_stream = None
                job_data_dict = None
                job_id_from_payload = None

                try:
                    if not isinstance(message_content, dict):
                        _logger.error(f"{consumer_name}: Consumed message is not a dict: {str(message_content)[:200]}. Skipping.")
                        continue

                    message_id_from_stream = message_content.get('message_id')
                    job_data_dict = message_content.get('data')
                    job_id_from_payload = message_content.get('job_id')

                    if not message_id_from_stream:
                        _logger.error(f"{consumer_name}: Consumed message is missing 'message_id': {str(message_content)[:200]}. Skipping.")
                        continue

                    if job_data_dict is None:
                        _logger.error(f"{consumer_name}: Consumed message {message_id_from_stream} is missing 'data' content. Acking and skipping.")
                        await message_broker.acknowledge_message(message_id_from_stream)
                        continue

                    if not job_id_from_payload:
                        _logger.error(f"{consumer_name}: Job data for message {message_id_from_stream} is missing 'job_id' in payload. Acking and skipping. Data: {str(job_data_dict)[:100]}")
                        await message_broker.acknowledge_message(message_id_from_stream)
                        continue

                    _logger.info(f"{consumer_name}: Creating task for job {job_id_from_payload} (message {message_id_from_stream}).")
                    task = asyncio.create_task(_process_single_efficient_scrape_job(
                        job_id=job_id_from_payload,
                        message_id=message_id_from_stream,
                        job_data_dict=job_data_dict,
                        message_broker=message_broker,
                        group_name=group_name
                    ))
                    active_tasks.add(task)

                except Exception as e_loop:
                    _logger.exception(f"{consumer_name}: Error in message processing loop for message {message_id_from_stream or 'UNKNOWN_ID'}: {e_loop}")
                    if message_id_from_stream:
                        try:
                            if job_id_from_payload:
                                await message_broker.track_job_progress(
                                    job_id_from_payload, "failed",
                                    data={"error": f"Worker loop error: {str(e_loop)}"}
                                )
                            await message_broker.acknowledge_message(message_id_from_stream)
                        except Exception as e_ack_fail:
                            _logger.error(f"{consumer_name}: Failed to acknowledge or track error for message {message_id_from_stream} after loop error: {e_ack_fail}")
                    continue
    except Exception as e_outer:
        _logger.exception(f"Outer exception in process_scrape_queue_worker for {consumer_name}: {e_outer}")
    finally:
        _logger.info(f"Scraper worker {consumer_name} main loop ended or errored. Waiting for active tasks to complete...")
        if active_tasks: # active_tasks is now guaranteed to be defined
            _logger.info(f"{consumer_name}: Waiting for {len(active_tasks)} active scraping tasks to complete...")
            shutdown_task_wait_timeout = getattr(app_config.scraper, 'WORKER_SHUTDOWN_TASK_TIMEOUT', 30.0)
            # Use asyncio.gather for waiting for tasks to complete or be cancelled
            results = await asyncio.gather(*active_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    task = list(active_tasks)[i] # This is not ideal for getting the specific task, but works for logging
                    _logger.error(f"{consumer_name}: Task {task.get_name() if hasattr(task, 'get_name') else 'unknown'} completed with exception during shutdown: {result}", exc_info=isinstance(result, BaseException))

            # Check if any tasks are still pending (should not happen with gather if timeout is not used with gather directly)
            # but if some tasks were not awaited properly or if gather itself is timed out (not the case here)
            pending_tasks = [t for t in active_tasks if not t.done()]
            if pending_tasks:
                _logger.warning(f"{consumer_name}: {len(pending_tasks)} tasks still pending after gather. Cancelling them.")
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True) # Wait for cancellations

        if message_broker:
            _logger.info(f"Closing RedisMessageBroker for {consumer_name}.")
            await message_broker.close()

        _logger.info(f"Scraper worker {consumer_name} shutdown complete.")

# ... (rest of the file, including main, handle_async_shutdown, etc.)
async def main():
    # ... (existing main function code) ...
    global _message_broker # Ensure we are using the global instance
    worker_task: Optional[asyncio.Task] = None # Initialize worker_task to None and type hint
    # ... (rest of main) ...
    try:
        # ... (existing try block in main) ...
        # Example of how worker_task might be created:
        # if some_condition_to_start_worker:
        #     worker_task = asyncio.create_task(process_scrape_queue_worker())

        if worker_task: # Only await if it was actually created and is a Task
             _logger.info("Main: Awaiting worker task.")
             await worker_task
        else:
            _logger.info("Main: Worker task not started or already completed.")

    except KeyboardInterrupt:
        _logger.info("KeyboardInterrupt received in main, initiating shutdown...")
        if not shutdown_event.is_set():
            shutdown_event.set()
    except Exception as e:
        _logger.exception(f"Exception in main: {e}")
    finally:
        _logger.info("Main function finally block: Ensuring worker shutdown and broker closure.")
        if not shutdown_event.is_set(): # If not already set by signal or other means
            shutdown_event.set()

        if worker_task and not worker_task.done(): # worker_task is guaranteed to be defined
            _logger.info("Waiting for worker task to complete in main finally...")
            try:
                await asyncio.wait_for(worker_task, timeout=getattr(app_config.scraper, 'WORKER_SHUTDOWN_TASK_TIMEOUT', 30.0) + 5.0)
            except asyncio.TimeoutError:
                _logger.error("Worker task did not complete in time during main finally block.")
            except Exception as e_wait_worker:
                _logger.error(f"Error waiting for worker task in main finally: {e_wait_worker}")

        if _message_broker: # Check if the global broker was initialized
            _logger.info("Closing global RedisMessageBroker from main function's finally block.")
            try:
                await _message_broker.close() # Ensure global broker is closed
            except Exception as e_close_broker:
                _logger.error(f"Error closing global message broker in main finally: {e_close_broker}")
        _logger.info("Scraper application shutdown process completed from main.")

# ... (rest of the file) ...
async def _process_single_efficient_scrape_job(job_id: str, message_id: str, job_data_dict: Dict[str, Any],
                                              message_broker: RedisMessageBroker, group_name: str):
    """
    Processes a single efficient scraping job.
    Handles tracking job progress and acknowledging the message.
    """
    consumer_name_for_log = f"scraper-worker-eff-task-{job_id[:8]}"
    _logger.debug(f"Starting efficient scrape worker for job {job_id}, message {message_id}")
    await message_broker.track_job_progress(job_id, "processing", data={"message_id": message_id, "start_time": time.time()}) # MODIFIED: Added await

    try:
        urls_to_scrape = job_data_dict.get('urls', [])
        if not urls_to_scrape:
            _logger.error(f"[{consumer_name_for_log}] No URLs found in job data for job {job_id}. Marking as failed.")
            await message_broker.track_job_progress(job_id, "failed", data={"error": "No URLs provided in the job data."}) # MODIFIED: Added await
            await message_broker.acknowledge_message(message_id)
            return

        browser_config = job_data_dict.get('browser_config')
        run_config = job_data_dict.get('run_config')
        dispatcher_config = job_data_dict.get('dispatcher_config')
        user_query = job_data_dict.get('user_query_for_bm25')

        _logger.info(f"[{consumer_name_for_log}] Calling scrape_urls_efficiently for job {job_id} with {len(urls_to_scrape)} URLs.")
        scraped_results: List[URLMetadata] = await scrape_urls_efficiently(
            urls_to_scrape,
            browser_config_dict=browser_config,
            run_config_dict=run_config,
            dispatcher_config_dict=dispatcher_config,
            user_query_for_bm25=user_query,
            job_id_for_log=job_id
        )
        _logger.info(f"[{consumer_name_for_log}] scrape_urls_efficiently completed for job {job_id}. Results count: {len(scraped_results)}.")

        serializable_results_list = [result.model_dump(mode='json') for result in scraped_results]
        final_data_for_tracking = {"results": serializable_results_list}


        await message_broker.track_job_progress(job_id, "completed", data=final_data_for_tracking) # MODIFIED: Added await
        _logger.info(f"[{consumer_name_for_log}] Job {job_id} (message {message_id}) processed and marked as completed.")

    except Exception as e:
        _logger.exception(f"[{consumer_name_for_log}] Error processing job {job_id} (message {message_id}): {e}")
        try:
            await message_broker.track_job_progress(job_id, "failed", data={"error": str(e)})
        except Exception as e_track:
            _logger.error(f"[{consumer_name_for_log}] CRITICAL: Failed to update job status to FAILED for job {job_id}: {e_track}")
    finally:
        try:
            _logger.debug(f"[{consumer_name_for_log}] Acknowledging message {message_id} for job {job_id}.")
            await message_broker.acknowledge_message(message_id)
        except Exception as e_ack:
            _logger.error(f"[{consumer_name_for_log}] Failed to acknowledge message {message_id} for job {job_id}: {e_ack}")

# Add signal handling functions
async def handle_async_shutdown(sig):
    """Async compatible shutdown handler."""
    if not shutdown_event.is_set():
        _logger.warning(f"Received signal {sig}, initiating async shutdown for scraper worker...")
        shutdown_event.set()

def handle_sync_shutdown(sig, frame=None):
    """Synchronous shutdown handler, calls asyncio.create_task if loop is available."""
    if not shutdown_event.is_set():
        _logger.warning(f"Received signal {sig} (sync handler), setting shutdown event for scraper worker...")
        shutdown_event.set()
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                asyncio.create_task(handle_async_shutdown(sig))
            else:
                _logger.info("No running asyncio loop to schedule async_shutdown from sync handler.")
        except RuntimeError:
            _logger.info("No running asyncio loop found by sync_shutdown handler.")

# --- Cleanup Function ---
async def close_redis_connections(): # Changed to async def
    """Closes Redis connections managed by this module's broker instance. Now asynchronous."""
    global _message_broker
    if (_message_broker and hasattr(_message_broker, 'close')):
        try:
            await _message_broker.close() # Added await
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