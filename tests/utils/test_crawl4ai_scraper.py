import pytest
import asyncio
import json
import base64
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming the module is viralStoryGenerator.utils.crawl4ai_scraper
from viralStoryGenerator.utils import crawl4ai_scraper as scraper_module
from viralStoryGenerator.utils.config import app_config # For patching config values
from viralStoryGenerator.utils.redis_manager import RedisMessageBroker # For type hinting and spec for mocks
from viralStoryGenerator.utils.models import URLMetadata # For type hinting/obj creation

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_scraper_defaults(monkeypatch):
    """Set default app_config values for crawl4ai_scraper tests."""
    # Redis settings
    monkeypatch.setattr(app_config.redis, 'HOST', "mock_redis_scrape_host")
    monkeypatch.setattr(app_config.redis, 'PORT', 6379)
    monkeypatch.setattr(app_config.redis, 'SCRAPER_JOB_STREAM_NAME', "test_scrape_jobs_stream")
    monkeypatch.setattr(app_config.redis, 'API_JOB_STREAM_NAME', "test_api_jobs_stream") # Used by client part
    
    # Scraper specific settings
    monkeypatch.setattr(app_config.scraper, 'HEADLESS_BROWSER', True)
    monkeypatch.setattr(app_config.scraper, 'USER_AGENT', "TestScraperClient/1.0")
    monkeypatch.setattr(app_config.scraper, 'BROWSER_TIMEOUT_SECONDS', 5) # Short for tests
    monkeypatch.setattr(app_config.scraper, 'MAX_DEPTH_SCRAPING', 0) # No internal crawling for basic tests
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', False) # Test with fresh instances first
    monkeypatch.setattr(app_config.scraper, 'DISPATCHER_TYPE', "semaphore") # Default dispatcher
    monkeypatch.setattr(app_config.scraper, 'SEMAPHORE_MAX_CONCURRENT', 5)
    monkeypatch.setattr(app_config.scraper, 'MEMORY_ADAPTIVE_MAX_CONCURRENT', 5)
    monkeypatch.setattr(app_config.scraper, 'MEMORY_ADAPTIVE_TARGET_MEMORY_PERCENT', 70)
    monkeypatch.setattr(app_config.scraper, 'RATE_LIMIT_REQUESTS_PER_SECOND', 0) # No rate limit for tests by default
    monkeypatch.setattr(app_config.scraper, 'CONTENT_FILTER_TYPE', "none") # Default no filter
    monkeypatch.setattr(app_config.scraper, 'BM25_SCORE_THRESHOLD', 0.1)
    monkeypatch.setattr(app_config.scraper, 'PRUNING_LLM_MODEL', "mock-pruning-model")
    monkeypatch.setattr(app_config.scraper, 'PRUNING_PROMPT_TEMPLATE', "Prune: {text}")
    monkeypatch.setattr(app_config.scraper, 'PRUNING_MAX_TOKENS', 100)
    monkeypatch.setattr(app_config.scraper, 'SCRAPE_TIMEOUT_PER_BATCH_SECONDS', 10)


    # LLM settings (for PruningContentFilter)
    monkeypatch.setattr(app_config.llm, 'ENDPOINT', "http://mock-llm-pruning.com/v1/chat/completions")
    monkeypatch.setattr(app_config.llm, 'MODEL', "mock-llm-pruning-model") # Should match PRUNING_LLM_MODEL
    monkeypatch.setattr(app_config.llm, 'MAX_TOKENS', 2000)
    monkeypatch.setattr(app_config.llm, 'DEFAULT_TEMPERATURE', 0.5)


    # Client part (for queue_scrape_request, get_scrape_result)
    monkeypatch.setattr(app_config.client, 'JOB_COMPLETION_TIMEOUT_SECONDS', 0.1)
    monkeypatch.setattr(app_config.client, 'JOB_STATUS_POLL_INTERVAL_SECONDS', 0.01)

    # Storage
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "local") # Default to local for easier testing
    monkeypatch.setattr(app_config.storage, 'LOCAL_STORAGE_PATH', "/tmp/mock_scraper_storage")
    monkeypatch.setattr(app_config.storage, 'S3_PUBLIC_URL_PREFIX', "http://s3.mock.com")


@pytest.fixture
def mock_scraper_logger():
    """Fixture to mock the _logger in crawl4ai_scraper.py."""
    with patch('viralStoryGenerator.utils.crawl4ai_scraper._logger') as mock_logger:
        yield mock_logger

@pytest.fixture(autouse=True)
def reset_scraper_module_globals(monkeypatch):
    """Reset global broker instance in crawl4ai_scraper.py before each test."""
    if hasattr(scraper_module, '_message_broker'):
        monkeypatch.setattr(scraper_module, '_message_broker', None)
    if hasattr(scraper_module, '_scrape_message_broker'): # Used by client functions
        monkeypatch.setattr(scraper_module, '_scrape_message_broker', None)


# --- Tests for Scenario 1: get_message_broker (scraper version) ---

@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper.RedisMessageBroker', spec=RedisMessageBroker)
async def test_scraper_get_message_broker_instantiates_correctly(
    MockRedisMessageBroker, mock_appconfig_scraper_defaults, mock_scraper_logger
):
    # Ensure it's reset for this specific test of creation
    scraper_module._message_broker = None 
    
    mock_broker_instance = MagicMock(spec=RedisMessageBroker)
    mock_broker_instance.initialize = AsyncMock()
    MockRedisMessageBroker.return_value = mock_broker_instance

    broker = await scraper_module.get_message_broker() # This is the worker-side broker

    assert broker is mock_broker_instance
    expected_redis_url = f"redis://{app_config.redis.HOST}:{app_config.redis.PORT}"
    MockRedisMessageBroker.assert_called_once_with(
        redis_url=expected_redis_url,
        stream_name=app_config.redis.SCRAPER_JOB_STREAM_NAME # Scraper specific stream
    )
    broker.initialize.assert_called_once()
    mock_scraper_logger.info.assert_any_call(
        f"Scraper internal message broker initialized for stream: {app_config.redis.SCRAPER_JOB_STREAM_NAME}"
    )

@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper.RedisMessageBroker', spec=RedisMessageBroker)
async def test_scraper_get_message_broker_returns_existing(
    MockRedisMessageBroker, mock_appconfig_scraper_defaults
):
    mock_existing_broker = MagicMock(spec=RedisMessageBroker)
    scraper_module._message_broker = mock_existing_broker # Pre-set the global broker

    broker = await scraper_module.get_message_broker()

    assert broker is mock_existing_broker
    MockRedisMessageBroker.assert_not_called() # Should not create a new one


# --- Tests for Scenario 2: _extract_rich_metadata_from_html ---

def create_mock_soup(meta_tags=None, title_text=None, canonical_url=None, favicons=None):
    """Helper to create a MagicMock for BeautifulSoup with find/find_all."""
    mock_soup = MagicMock()
    
    # Mock title tag
    if title_text:
        mock_title_tag = MagicMock()
        mock_title_tag.string = title_text
        mock_soup.find.return_value = mock_title_tag # Default find for 'title'
    else:
        # If no title_text, find('title') should return None or a tag with no string
        mock_title_tag = MagicMock()
        mock_title_tag.string = None
        # Make find differentiate if other find calls are needed
        def find_side_effect(name, attrs=None):
            if name == 'title':
                return mock_title_tag
            if name == 'link' and attrs and attrs.get('rel') == 'canonical':
                 mock_link_tag = MagicMock()
                 mock_link_tag.get.return_value = canonical_url
                 return mock_link_tag
            return None # Default for other finds
        mock_soup.find.side_effect = find_side_effect


    # Mock find_all for meta tags and link[rel="icon"]
    found_tags = []
    if meta_tags: # List of dicts like {'name': 'description', 'content': '...'} or {'property': 'og:title', ...}
        for tag_attrs in meta_tags:
            mock_tag = MagicMock()
            mock_tag.get.side_effect = lambda key, default=None, tag_attrs_capture=tag_attrs: tag_attrs_capture.get(key, default)
            # For soup.find_all('meta'), it returns a list of these tag objects.
            # We need to simulate the .attrs access if used, or specific .get calls.
            # The code uses `tag.get('name')` or `tag.get('property')`.
            found_tags.append(mock_tag)
            
    if favicons: # List of hrefs for favicons
        for href in favicons:
            mock_link_tag = MagicMock()
            mock_link_tag.get.side_effect = lambda key, default=None, href_capture=href: href_capture if key == 'href' else default
            # Differentiate link tags for favicons from canonical
            # The code uses `soup.find_all('link', rel=lambda r: r and 'icon' in r.lower())`
            # This is hard to mock perfectly with a simple side_effect on find_all for all link tags.
            # For now, assume find_all for 'link' will be handled by specific test patches if needed.
            # Or, make find_all more complex.
            # Let's assume favicons are directly passed to a more specific mock if this doesn't work.
            # For now, just add them to found_tags and hope the lambda in find_all isn't an issue.
            # This part is tricky. The code's `soup.find_all('link', rel=lambda r: ...)` is specific.
            # A better way for favicons is to patch the find_all for link tags specifically.
            pass # Will handle favicon mocking specifically in tests needing it.


    def find_all_side_effect(name, attrs=None, rel=None): # Added rel for favicons
        if name == 'meta':
            # Filter meta tags based on 'name' or 'property' if attrs is used this way
            if attrs:
                key_to_check = list(attrs.keys())[0] # e.g. 'name' or 'property'
                value_to_check = attrs[key_to_check]
                return [t for t in found_tags if t.get(key_to_check) == value_to_check]
            return [t for t in found_tags if 'content' in t.attrs_dict] # Crude filter for meta tags
        if name == 'link' and rel: # For favicons
            # This part needs to correctly simulate the lambda r and 'icon' in r.lower()
            # This is a simplified version.
            simulated_favicon_tags = []
            if favicons:
                 for href_val in favicons:
                    mock_f_tag = MagicMock()
                    mock_f_tag.get.return_value = href_val # for .get('href')
                    simulated_favicon_tags.append(mock_f_tag)
            return simulated_favicon_tags
        return []

    mock_soup.find_all.side_effect = find_all_side_effect
    # Store attrs on mock_tag for easier debugging if needed
    for mt in found_tags:
        mt.attrs_dict = {k:v for k,v in mt.get.side_effect(None,None,None).items()} if callable(mt.get.side_effect) else {}


    return mock_soup


@pytest.mark.parametrize("meta_tags, title_text, canonical_url, favicons, expected_metadata", [
    # Basic OpenGraph and Twitter
    ([{"property": "og:title", "content": "OG Title"}, 
      {"property": "og:description", "content": "OG Desc"},
      {"property": "og:image", "content": "og_image.png"},
      {"name": "twitter:card", "content": "summary_large_image"}], 
     "Html Title", "http://example.com/canonical", ["favicon.ico"],
     {"title": "OG Title", "description": "OG Desc", "image_url": "og_image.png", 
      "canonical_url": "http://example.com/canonical", "favicon_urls": ["favicon.ico"], "twitter_card": "summary_large_image"}),
    # Standard meta description, no OG/Twitter title
    ([{"name": "description", "content": "Std Desc"}], 
     "Html Title Only", None, [],
     {"title": "Html Title Only", "description": "Std Desc", "image_url": None, 
      "canonical_url": None, "favicon_urls": [], "twitter_card": None}),
    # Only favicons and title
    ([], "Favicon Test", None, ["fav1.png", "fav2.ico"],
     {"title": "Favicon Test", "description": None, "image_url": None, 
      "canonical_url": None, "favicon_urls": ["fav1.png", "fav2.ico"], "twitter_card": None}),
    # All missing
    ([], None, None, [],
     {"title": None, "description": None, "image_url": None, 
      "canonical_url": None, "favicon_urls": [], "twitter_card": None}),
    # Twitter title takes precedence if no OG title
    ([{"name": "twitter:title", "content": "Twitter Title"},
      {"name": "description", "content": "Desc for Twitter"}],
     "HTML Title Fallback", None, [],
     {"title": "Twitter Title", "description": "Desc for Twitter", "image_url": None,
      "canonical_url": None, "favicon_urls": [], "twitter_card": None}),
    # Ensure favicon list is limited (if logic implemented, else it takes all) - Assuming up to 5
    ([], "Title", None, [f"fav{i}.ico" for i in range(7)],
     {"title": "Title", "description": None, "image_url": None, "canonical_url": None, 
      "favicon_urls": [f"fav{i}.ico" for i in range(5)], "twitter_card": None}), # Assuming it takes first 5
])
@patch('viralStoryGenerator.utils.crawl4ai_scraper.BeautifulSoup') # Patch where BS is imported
def test_extract_rich_metadata_from_html_scenarios(
    MockBeautifulSoup, meta_tags, title_text, canonical_url, favicons, expected_metadata, 
    mock_scraper_logger, mock_appconfig_scraper_defaults, monkeypatch
):
    monkeypatch.setattr(scraper_module, 'BEAUTIFULSOUP_AVAILABLE', True)
    raw_html_content = "<html><body>Mock HTML</body></html>" # Content doesn't matter as BS is mocked
    
    # Create a mock soup object that will behave as if it parsed the HTML
    # and found the specified tags.
    # The helper `create_mock_soup` needs to be more robust for favicons.
    # Let's refine the mocking of soup.find and soup.find_all for this test.

    mock_soup_instance = MagicMock()

    # Mock title
    mock_title_tag_instance = MagicMock()
    mock_title_tag_instance.string = title_text
    
    # Mock canonical URL
    mock_canonical_tag_instance = MagicMock()
    mock_canonical_tag_instance.get.return_value = canonical_url

    def find_side_effect(name, attrs=None):
        if name == 'title':
            return mock_title_tag_instance if title_text else None
        if name == 'link' and attrs and attrs.get('rel') == 'canonical':
            return mock_canonical_tag_instance if canonical_url else None
        return None
    mock_soup_instance.find.side_effect = find_side_effect

    # Mock meta tags and favicon links
    mock_meta_tag_list = []
    for tag_data in meta_tags:
        tag = MagicMock()
        # .get('name') or .get('property')
        tag.get.side_effect = lambda key, default=None, td=tag_data: td.get(key, default)
        mock_meta_tag_list.append(tag)

    mock_favicon_link_list = []
    if favicons:
        for fav_href in favicons:
            link_tag = MagicMock()
            link_tag.get.side_effect = lambda key, default=None, fh=fav_href: fh if key == 'href' else default
            mock_favicon_link_list.append(link_tag)

    def find_all_side_effect_detailed(name, attrs=None, rel=None): # rel for link tags
        if name == 'meta':
            # This simplified version returns all mocked meta tags.
            # The function itself filters them by 'name' or 'property'.
            return mock_meta_tag_list
        if name == 'link' and rel: # rel is a lambda function here
            # Simulate the lambda for favicons
            # This is a simplified check; real lambda is `lambda r: r and 'icon' in r.lower()`
            # We assume the mock setup for favicons here is sufficient for the test.
            # The provided `rel` in the actual code is a function.
            # We'd have to inspect `rel` if we wanted to be super precise.
            # For this mock, we just return all prepared favicon links.
            return mock_favicon_link_list
        return []
    mock_soup_instance.find_all.side_effect = find_all_side_effect_detailed
    
    MockBeautifulSoup.return_value = mock_soup_instance

    metadata = scraper_module._extract_rich_metadata_from_html(raw_html_content, "http://example.com/testpage")

    assert metadata.title == expected_metadata["title"]
    assert metadata.description == expected_metadata["description"]
    assert metadata.image_url == expected_metadata["image_url"]
    assert metadata.canonical_url == expected_metadata["canonical_url"]
    assert metadata.favicon_urls == expected_metadata["favicon_urls"]
    assert metadata.twitter_card == expected_metadata["twitter_card"]
    
    MockBeautifulSoup.assert_called_once_with(raw_html_content, 'html.parser')


def test_extract_rich_metadata_beautifulsoup_not_available(mock_scraper_logger, monkeypatch):
    monkeypatch.setattr(scraper_module, 'BEAUTIFULSOUP_AVAILABLE', False)
    raw_html_content = "<html>Some HTML</html>"
    
    metadata = scraper_module._extract_rich_metadata_from_html(raw_html_content, "http://example.com/no_bs_page")
    
    # Should return a default/empty URLMetadata object
    assert metadata.title is None
    assert metadata.description is None
    assert metadata.image_url is None
    mock_scraper_logger.warning.assert_any_call(
        "BeautifulSoup4 is not installed. Cannot extract rich metadata from HTML. Please run 'pip install beautifulsoup4'."
    )

# --- Tests for Scenario 3: scrape_urls_efficiently ---

@pytest.fixture
def mock_crawl4ai_components(monkeypatch):
    """Mocks all imported crawl4ai components."""
    mocks = {
        'AsyncWebCrawler': MagicMock(spec=scraper_module.crawl4ai.AsyncWebCrawler if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'BrowserConfig': MagicMock(spec=scraper_module.crawl4ai.BrowserConfig if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'CrawlerRunConfig': MagicMock(spec=scraper_module.crawl4ai.CrawlerRunConfig if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'DefaultMarkdownGenerator': MagicMock(spec=scraper_module.crawl4ai.DefaultMarkdownGenerator if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'BM25ContentFilter': MagicMock(spec=scraper_module.crawl4ai.BM25ContentFilter if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'PruningContentFilter': MagicMock(spec=scraper_module.crawl4ai.PruningContentFilter if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'RateLimiter': MagicMock(spec=scraper_module.crawl4ai.RateLimiter if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'CrawlerMonitor': MagicMock(spec=scraper_module.crawl4ai.CrawlerMonitor if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'MemoryAdaptiveDispatcher': MagicMock(spec=scraper_module.crawl4ai.MemoryAdaptiveDispatcher if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
        'SemaphoreDispatcher': MagicMock(spec=scraper_module.crawl4ai.SemaphoreDispatcher if scraper_module.CRAWL4AI_AVAILABLE else MagicMock),
    }
    
    # Make these mocks available by patching where they are imported in scraper_module
    for class_name, mock_class in mocks.items():
        monkeypatch.setattr(scraper_module, class_name, mock_class, raising=False) # raising=False if class might not exist when CRAWL4AI_AVAILABLE is False
        # If CRAWL4AI_AVAILABLE is False, these classes in scraper_module are already MagicMock.
        # We want to control them regardless.
        if scraper_module.CRAWL4AI_AVAILABLE: # Patching the actual classes from crawl4ai
             monkeypatch.setattr(scraper_module.crawl4ai, class_name, mock_class, raising=False)
        else: # Patching the MagicMock placeholders in scraper_module
             monkeypatch.setattr(scraper_module, class_name, mock_class)


    # Configure return values for constructors if needed
    mocks['AsyncWebCrawler'].return_value.arun_many = AsyncMock() # arun_many is an async method
    mocks['AsyncWebCrawler'].return_value.close = AsyncMock() # close is async
    
    return mocks

@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html')
@patch('viralStoryGenerator.utils.storage_manager.store_file', new_callable=AsyncMock)
@patch('asyncio.wait_for') # To control timeout of arun_many
async def test_scrape_urls_efficiently_success_bm25(
    mock_asyncio_wait_for, mock_store_file, mock_extract_metadata,
    mock_crawl4ai_components, mock_scraper_logger, mock_appconfig_scraper_defaults, monkeypatch
):
    monkeypatch.setattr(scraper_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(app_config.scraper, 'CONTENT_FILTER_TYPE', "bm25")
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', False) # Test with fresh instance

    urls_to_scrape = ["http://example.com/page1", "http://example.com/page2"]
    job_id = "job_scrape_bm25_001"
    
    # Mock arun_many results from AsyncWebCrawler instance
    mock_crawl_result_1 = MagicMock()
    mock_crawl_result_1.url = urls_to_scrape[0]
    mock_crawl_result_1.markdown = "Markdown for page 1"
    mock_crawl_result_1.raw_html = "<html>Page 1</html>"
    mock_crawl_result_1.screenshot_path = "/tmp/page1.png" # Local path from crawl4ai
    mock_crawl_result_1.error_message = None

    mock_crawl_result_2 = MagicMock()
    mock_crawl_result_2.url = urls_to_scrape[1]
    mock_crawl_result_2.markdown = "Markdown for page 2"
    mock_crawl_result_2.raw_html = "<html>Page 2</html>"
    mock_crawl_result_2.screenshot_path = "/tmp/page2.png"
    mock_crawl_result_2.error_message = None
    
    # arun_many on the instance of AsyncWebCrawler
    mock_crawler_instance = mock_crawl4ai_components['AsyncWebCrawler'].return_value
    mock_crawler_instance.arun_many.return_value = [mock_crawl_result_1, mock_crawl_result_2]
    
    # asyncio.wait_for just returns the result of arun_many in this success case
    mock_asyncio_wait_for.side_effect = lambda coro, timeout: coro 

    # Mock _extract_rich_metadata_from_html
    mock_metadata_1 = URLMetadata(url=urls_to_scrape[0], title="Title 1", markdown_content="MD1", raw_html="HTML1", screenshot_url=None, error_message=None)
    mock_metadata_2 = URLMetadata(url=urls_to_scrape[1], title="Title 2", markdown_content="MD2", raw_html="HTML2", screenshot_url=None, error_message=None)
    mock_extract_metadata.side_effect = [mock_metadata_1, mock_metadata_2]

    # Mock store_file for screenshots
    screenshot_url_1 = f"s3://{job_id}/screenshots/page1.png"
    screenshot_url_2 = f"s3://{job_id}/screenshots/page2.png"
    mock_store_file.side_effect = [screenshot_url_1, screenshot_url_2]

    # --- Call the function ---
    results = await scraper_module.scrape_urls_efficiently(urls_to_scrape, job_id)

    # --- Assertions ---
    assert len(results) == 2
    
    # Result 1
    assert results[0].url == urls_to_scrape[0]
    assert results[0].title == "Title 1" # From _extract_rich_metadata
    assert results[0].markdown_content == "Markdown for page 1" # From crawl_result
    assert results[0].raw_html == "<html>Page 1</html>"
    assert results[0].screenshot_url == screenshot_url_1
    assert results[0].error_message is None

    # Result 2
    assert results[1].url == urls_to_scrape[1]
    assert results[1].title == "Title 2"
    assert results[1].markdown_content == "Markdown for page 2"
    assert results[1].screenshot_url == screenshot_url_2

    # Verify AsyncWebCrawler instantiation and arun_many call
    mock_crawl4ai_components['AsyncWebCrawler'].assert_called_once() # New instance created
    mock_crawler_instance.arun_many.assert_called_once()
    args_arun, _ = mock_crawler_instance.arun_many.call_args
    assert args_arun[0] == urls_to_scrape # urls
    # Check CrawlerRunConfig
    run_config_arg = args_arun[1]
    assert isinstance(run_config_arg, mock_crawl4ai_components['CrawlerRunConfig'])
    assert run_config_arg.max_depth == app_config.scraper.MAX_DEPTH_SCRAPING
    # Check content_filter is BM25
    assert isinstance(run_config_arg.content_filter, mock_crawl4ai_components['BM25ContentFilter'])
    assert run_config_arg.content_filter.score_threshold == app_config.scraper.BM25_SCORE_THRESHOLD

    # Verify _extract_rich_metadata_from_html calls
    mock_extract_metadata.assert_any_call("<html>Page 1</html>", urls_to_scrape[0])
    mock_extract_metadata.assert_any_call("<html>Page 2</html>", urls_to_scrape[1])
    assert mock_extract_metadata.call_count == 2

    # Verify store_file calls for screenshots
    mock_store_file.assert_any_call(mock_crawl_result_1.screenshot_path, job_id, f"screenshots/{job_id}_page1.png", is_temp_file=True)
    mock_store_file.assert_any_call(mock_crawl_result_2.screenshot_path, job_id, f"screenshots/{job_id}_page2.png", is_temp_file=True)
    assert mock_store_file.call_count == 2
    
    # Verify crawler.close() was called
    mock_crawler_instance.close.assert_called_once()
    mock_scraper_logger.info.assert_any_call(f"Scraping complete for job {job_id}. Results: {len(results)}")


# Scenario 3.2: Successful scrape (Pruning filter)
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html')
@patch('viralStoryGenerator.utils.storage_manager.store_file', new_callable=AsyncMock)
@patch('asyncio.wait_for')
async def test_scrape_urls_efficiently_success_pruning_filter(
    mock_asyncio_wait_for, mock_store_file, mock_extract_metadata,
    mock_crawl4ai_components, mock_scraper_logger, mock_appconfig_scraper_defaults, monkeypatch
):
    monkeypatch.setattr(scraper_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(app_config.scraper, 'CONTENT_FILTER_TYPE', "pruning")
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', False)

    urls_to_scrape = ["http://example.com/prune_page"]
    job_id = "job_scrape_pruning_002"
    
    mock_crawl_result = MagicMock()
    mock_crawl_result.url = urls_to_scrape[0]
    mock_crawl_result.markdown = "Pruned markdown content"
    mock_crawl_result.raw_html = "<html>Prune Page</html>"
    mock_crawl_result.screenshot_path = None # No screenshot for this test
    mock_crawl_result.error_message = None
    
    mock_crawler_instance = mock_crawl4ai_components['AsyncWebCrawler'].return_value
    mock_crawler_instance.arun_many.return_value = [mock_crawl_result]
    mock_asyncio_wait_for.side_effect = lambda coro, timeout: coro 

    mock_metadata = URLMetadata(url=urls_to_scrape[0], title="Pruned Title", markdown_content="MD_pruned", raw_html="HTML_pruned", screenshot_url=None, error_message=None)
    mock_extract_metadata.return_value = mock_metadata

    results = await scraper_module.scrape_urls_efficiently(urls_to_scrape, job_id)

    assert len(results) == 1
    assert results[0].markdown_content == "Pruned markdown content"
    
    mock_crawler_instance.arun_many.assert_called_once()
    run_config_arg = mock_crawler_instance.arun_many.call_args[0][1]
    assert isinstance(run_config_arg.content_filter, mock_crawl4ai_components['PruningContentFilter'])
    assert run_config_arg.content_filter.llm_model == app_config.scraper.PRUNING_LLM_MODEL
    assert run_config_arg.content_filter.llm_endpoint == app_config.llm.ENDPOINT # Uses main LLM endpoint
    mock_store_file.assert_not_called() # No screenshot path
    mock_crawler_instance.close.assert_called_once()


# Scenario 3.3: arun_many returns partial failures
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html')
@patch('viralStoryGenerator.utils.storage_manager.store_file', new_callable=AsyncMock)
@patch('asyncio.wait_for')
async def test_scrape_urls_efficiently_partial_failures(
    mock_asyncio_wait_for, mock_store_file, mock_extract_metadata,
    mock_crawl4ai_components, mock_scraper_logger, mock_appconfig_scraper_defaults, monkeypatch
):
    monkeypatch.setattr(scraper_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(app_config.scraper, 'CONTENT_FILTER_TYPE', "none")
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', False)

    urls = ["http://ok.com/page1", "http://fail.com/page2"]
    job_id = "job_partial_fail_003"

    mock_crawl_ok = MagicMock(url=urls[0], markdown="OK MD", raw_html="OK HTML", screenshot_path=None, error_message=None)
    mock_crawl_fail = MagicMock(url=urls[1], markdown=None, raw_html=None, screenshot_path=None, error_message="Connection Timeout")
    
    mock_crawler_instance = mock_crawl4ai_components['AsyncWebCrawler'].return_value
    mock_crawler_instance.arun_many.return_value = [mock_crawl_ok, mock_crawl_fail]
    mock_asyncio_wait_for.side_effect = lambda coro, timeout: coro

    mock_metadata_ok = URLMetadata(url=urls[0], title="OK", markdown_content="OK MD", raw_html="OK HTML", error_message=None)
    # _extract_rich_metadata_from_html is not called for failed pages by scrape_urls_efficiently
    # as raw_html would be None. It's called with None, and should handle it.
    mock_extract_metadata.side_effect = lambda html, url: mock_metadata_ok if html == "OK HTML" else URLMetadata(url=url)


    results = await scraper_module.scrape_urls_efficiently(urls, job_id)

    assert len(results) == 2
    assert results[0].url == urls[0]
    assert results[0].markdown_content == "OK MD"
    assert results[0].error_message is None
    
    assert results[1].url == urls[1]
    assert results[1].markdown_content is None
    assert results[1].error_message == "Connection Timeout"
    
    mock_extract_metadata.assert_any_call("OK HTML", urls[0])
    # Check if it was called for the failed one (it would be with None if raw_html is None)
    mock_extract_metadata.assert_any_call(None, urls[1])
    
    mock_crawler_instance.close.assert_called_once()


# Scenario 3.4: arun_many raises an exception (e.g., TimeoutError)
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html') # Should not be called
@patch('asyncio.wait_for')
async def test_scrape_urls_efficiently_arun_many_timeout_exception(
    mock_asyncio_wait_for, mock_extract_metadata,
    mock_crawl4ai_components, mock_scraper_logger, mock_appconfig_scraper_defaults, monkeypatch
):
    monkeypatch.setattr(scraper_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', False)
    urls = ["http://timeout.com/page1"]
    job_id = "job_timeout_exc_004"

    # asyncio.wait_for wrapping arun_many raises TimeoutError
    mock_asyncio_wait_for.side_effect = asyncio.TimeoutError("Simulated arun_many timeout")
    
    mock_crawler_instance = mock_crawl4ai_components['AsyncWebCrawler'].return_value
    # arun_many itself might not be directly asserted if wait_for fails first,
    # but it's good to have it as an AsyncMock.
    mock_crawler_instance.arun_many = AsyncMock() 


    results = await scraper_module.scrape_urls_efficiently(urls, job_id)

    assert len(results) == 1
    assert results[0].url == urls[0]
    assert results[0].error_message == "Scraping timed out for URL http://timeout.com/page1 after 10 seconds."
    assert results[0].markdown_content is None
    
    mock_asyncio_wait_for.assert_called_once() # wait_for was called
    mock_extract_metadata.assert_not_called() # Not called if arun_many effectively fails
    mock_crawler_instance.close.assert_called_once() # Close should still be called in finally
    mock_scraper_logger.error.assert_any_call(f"Scraping process for job {job_id} timed out: Simulated arun_many timeout")


# Scenario 3.5: CRAWL4AI_AVAILABLE is False
@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html') # Should not be called
async def test_scrape_urls_efficiently_crawl4ai_not_available(
    mock_extract_metadata, mock_scraper_logger, mock_appconfig_scraper_defaults, monkeypatch
):
    monkeypatch.setattr(scraper_module, 'CRAWL4AI_AVAILABLE', False) # Simulate library not available
    urls = ["http://example.com/no_crawl4ai"]
    job_id = "job_no_crawl4ai_005"

    results = await scraper_module.scrape_urls_efficiently(urls, job_id)

    assert len(results) == 1
    assert results[0].url == urls[0]
    assert "Crawl4AI library is not available" in results[0].error_message
    assert results[0].markdown_content is None
    mock_extract_metadata.assert_not_called()
    mock_scraper_logger.error.assert_any_call("Crawl4AI library not available, cannot perform scraping.")


# Scenario 3.6: Dispatcher configuration
@pytest.mark.asyncio
@patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro) # Pass through
async def test_scrape_urls_efficiently_dispatcher_config(
    mock_asyncio_wait_for, mock_crawl4ai_components, mock_appconfig_scraper_defaults, monkeypatch
):
    urls = ["http://example.com/dispatcher_test"]
    job_id = "job_dispatcher_006"
    
    # Mock arun_many on the crawler instance
    mock_crawler_instance = mock_crawl4ai_components['AsyncWebCrawler'].return_value
    mock_crawl_result = MagicMock(url=urls[0], markdown="MD", raw_html="HTML", screenshot_path=None, error_message=None)
    mock_crawler_instance.arun_many.return_value = [mock_crawl_result]

    # Test with SemaphoreDispatcher
    monkeypatch.setattr(app_config.scraper, 'DISPATCHER_TYPE', "semaphore")
    monkeypatch.setattr(app_config.scraper, 'SEMAPHORE_MAX_CONCURRENT', 7)
    monkeypatch.setattr(scraper_module, 'CRAWL4AI_AVAILABLE', True)
    monkeypatch.setattr(app_config.scraper, 'USE_SHARED_CRAWLER_INSTANCE', False)


    with patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html', return_value=URLMetadata()):
        await scraper_module.scrape_urls_efficiently(urls, job_id, crawler=None) # Pass crawler=None to force new instance

    mock_crawl4ai_components['SemaphoreDispatcher'].assert_called_once_with(max_concurrent_tasks=7)
    run_config_arg_sema = mock_crawler_instance.arun_many.call_args[0][1]
    assert isinstance(run_config_arg_sema.dispatcher, mock_crawl4ai_components['SemaphoreDispatcher'])
    mock_crawler_instance.close.assert_called_once() # From this call

    # Reset for next part of test
    mock_crawl4ai_components['AsyncWebCrawler'].reset_mock()
    mock_crawl4ai_components['SemaphoreDispatcher'].reset_mock()
    mock_crawl4ai_components['MemoryAdaptiveDispatcher'].reset_mock()
    mock_crawler_instance.reset_mock() # Reset the instance too
    mock_crawler_instance.arun_many.return_value = [mock_crawl_result] # Re-assign side effect
    mock_crawler_instance.close = AsyncMock() # Re-assign close mock

    # Test with MemoryAdaptiveDispatcher
    monkeypatch.setattr(app_config.scraper, 'DISPATCHER_TYPE', "memory_adaptive")
    monkeypatch.setattr(app_config.scraper, 'MEMORY_ADAPTIVE_MAX_CONCURRENT', 9)
    monkeypatch.setattr(app_config.scraper, 'MEMORY_ADAPTIVE_TARGET_MEMORY_PERCENT', 65)
    
    with patch('viralStoryGenerator.utils.crawl4ai_scraper._extract_rich_metadata_from_html', return_value=URLMetadata()):
        await scraper_module.scrape_urls_efficiently(urls, job_id, crawler=None)

    mock_crawl4ai_components['MemoryAdaptiveDispatcher'].assert_called_once_with(
        max_concurrent_tasks=9, target_memory_percent=65
    )
    run_config_arg_mem = mock_crawler_instance.arun_many.call_args[0][1]
    assert isinstance(run_config_arg_mem.dispatcher, mock_crawl4ai_components['MemoryAdaptiveDispatcher'])
    mock_crawler_instance.close.assert_called_once()

# --- Tests for Scenario 4: queue_scrape_request ---
# These tests use the _scrape_message_broker (client-side)

@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker') # Patch the client-side broker getter
@patch('uuid.uuid4')
async def test_queue_scrape_request_client_no_wait_successful(
    mock_uuid4_client, mock_get_client_broker, mock_appconfig_scraper_defaults
):
    job_id = "client_scrape_no_wait_123"
    message_id = "client_msg_id_scrape_no_wait"
    mock_uuid4_client.return_value = MagicMock(hex=job_id)
    
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    mock_client_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_client_broker.return_value = mock_client_broker

    urls_to_scrape = ["http://client.example.com/scrape1"]
    custom_payload = {"user_id": "user123", "priority": "high"}
    
    returned_job_id = await scraper_module.queue_scrape_request(
        urls_to_scrape, payload_override=custom_payload, wait_for_result=False
    )

    assert returned_job_id == job_id
    mock_get_client_broker.assert_called_once()

# --- Tests for Scenario 5: get_scrape_result ---
# These tests also use the _scrape_message_broker (client-side)

@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
async def test_get_scrape_result_client_completed_json_payload(
    mock_get_client_broker, mock_appconfig_scraper_defaults
):
    job_id = "client_scrape_res_json_001"
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    
    scraped_results_list = [{"url": "u1", "title": "T1", "markdown": "MD1"}]
    # Simulate payload being a JSON string that needs parsing
    completed_status_json_payload = {
        "job_id": job_id, 
        "status": "completed", 
        "payload": json.dumps({"scraped_content": scraped_results_list}) 
    }
    mock_client_broker.get_job_status = AsyncMock(return_value=completed_status_json_payload)
    mock_get_client_broker.return_value = mock_client_broker

    result = await scraper_module.get_scrape_result(job_id)

    # Result should be the parsed list from scraped_content
    assert result == scraped_results_list 
    mock_client_broker.get_job_status.assert_called_once_with(job_id)


@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
async def test_get_scrape_result_client_completed_direct_payload(
    mock_get_client_broker, mock_appconfig_scraper_defaults
):
    job_id = "client_scrape_res_direct_002"
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    
    scraped_results_list = [{"url": "u2", "title": "T2", "markdown": "MD2"}]
    # Simulate payload being a direct dictionary
    completed_status_direct_payload = {
        "job_id": job_id, 
        "status": "completed", 
        "payload": {"scraped_content": scraped_results_list}
    }
    mock_client_broker.get_job_status = AsyncMock(return_value=completed_status_direct_payload)
    mock_get_client_broker.return_value = mock_client_broker

    result = await scraper_module.get_scrape_result(job_id)

    assert result == scraped_results_list
    mock_client_broker.get_job_status.assert_called_once_with(job_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status_return", [
    {"job_id": "s_id", "status": "processing"},
    {"job_id": "s_id", "status": "failed", "error_message": "Scrape failed hard"},
    None, # Job status not found
    {"job_id": "s_id", "status": "completed"}, # Completed but no payload
    {"job_id": "s_id", "status": "completed", "payload": {}}, # Completed, payload, but no scraped_content
    {"job_id": "s_id", "status": "completed", "payload": {"scraped_content": "not_a_list"}}, # scraped_content wrong type
    {"job_id": "s_id", "status": "completed", "payload": "not_a_dict_or_json_string"}, # payload wrong type
])
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
async def test_get_scrape_result_client_non_successful_or_malformed(
    mock_get_client_broker, job_status_return, mock_appconfig_scraper_defaults, mock_scraper_logger
):
    job_id = "client_scrape_res_various_fails_003"
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    mock_client_broker.get_job_status = AsyncMock(return_value=job_status_return)
    mock_get_client_broker.return_value = mock_client_broker

    result = await scraper_module.get_scrape_result(job_id)

    assert result is None # Should be None for all these cases
    mock_client_broker.get_job_status.assert_called_once_with(job_id)
    
    # Check for logging if status was found but was not usable
    if job_status_return and job_status_return.get("status") == "completed":
        if not isinstance(job_status_return.get("payload"), (dict, str)):
            mock_scraper_logger.warning.assert_any_call(
                f"Job {job_id} status indicates completion, but payload is not a dictionary or JSON string. Payload: {job_status_return.get('payload')}"
            )
        elif "scraped_content" not in (json.loads(job_status_return["payload"]) if isinstance(job_status_return["payload"], str) else job_status_return["payload"]):
             mock_scraper_logger.warning.assert_any_call(
                f"Job {job_id} status indicates completion, but 'scraped_content' key is missing in payload. Payload: {job_status_return['payload']}"
            )
        elif not isinstance((json.loads(job_status_return["payload"]) if isinstance(job_status_return["payload"], str) else job_status_return["payload"]).get("scraped_content"), list):
             mock_scraper_logger.warning.assert_any_call(
                f"Job {job_id} status indicates completion, but 'scraped_content' is not a list. Found: {type((json.loads(job_status_return['payload']) if isinstance(job_status_return['payload'], str) else job_status_return['payload']).get('scraped_content'))}"
            )
    elif job_status_return and job_status_return.get("status") == "failed":
        mock_scraper_logger.info.assert_any_call(f"Scrape job {job_id} failed. Status: {job_status_return}")
    elif job_status_return and job_status_return.get("status") == "processing":
         mock_scraper_logger.info.assert_any_call(f"Scrape job {job_id} is still {job_status_return['status']}.")
    elif job_status_return is None:
        mock_scraper_logger.info.assert_any_call(f"No status found for scrape job {job_id}.")

# --- Tests for Scenario 6: wait_for_job_result (client-side) ---

@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_client_wait_for_job_result_completes_quickly(
    mock_asyncio_sleep_client, mock_appconfig_scraper_defaults
):
    job_id = "client_wait_completes_001"
    mock_broker_client_wait = AsyncMock(spec=RedisMessageBroker) # Broker instance for this test
    
    completed_status = {"job_id": job_id, "status": "completed", "result": "client_done"}
    call_count = 0
    async def get_status_side_effect_client(jid):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"job_id": jid, "status": "processing"}
        return completed_status
        
    mock_broker_client_wait.get_job_status = AsyncMock(side_effect=get_status_side_effect_client)

    result = await scraper_module.wait_for_job_result(
        job_id, 
        mock_broker_client_wait, 
        timeout_seconds=app_config.client.JOB_COMPLETION_TIMEOUT_SECONDS, # From fixture
        poll_interval_seconds=app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS
    )

    assert result == completed_status
    assert mock_broker_client_wait.get_job_status.call_count == 2
    mock_asyncio_sleep_client.assert_called_once_with(app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS)


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_client_wait_for_job_result_times_out(
    mock_asyncio_sleep_client, mock_appconfig_scraper_defaults, monkeypatch # Use monkeypatch from pytest
):
    job_id = "client_wait_timeout_002"
    mock_broker_client_wait_timeout = AsyncMock(spec=RedisMessageBroker)
    
    processing_status = {"job_id": job_id, "status": "processing"}
    mock_broker_client_wait_timeout.get_job_status = AsyncMock(return_value=processing_status)

    # Use specific, short timeout for this test
    test_timeout = 0.03
    test_poll_interval = 0.01
    # Monkeypatch app_config directly if wait_for_job_result re-reads from global app_config
    # Or, ensure the values from mock_appconfig_scraper_defaults are used by passing them.
    # The function signature allows explicit timeout/poll, so we use that.

    result = await scraper_module.wait_for_job_result(
        job_id, 
        mock_broker_client_wait_timeout,
        timeout_seconds=test_timeout,
        poll_interval_seconds=test_poll_interval
    )

    assert result["job_id"] == job_id
    assert result["status"] == "timeout"
    assert f"Job {job_id} timed out after {test_timeout} seconds." in result["error_message"]
    
    # Expected calls: timeout / poll_interval = 0.03 / 0.01 = 3 attempts for status
    # Sleep calls = 2 (approx)
    assert mock_broker_client_wait_timeout.get_job_status.call_count >= (test_timeout / test_poll_interval)
    assert mock_asyncio_sleep_client.call_count >= (test_timeout / test_poll_interval) -1


@pytest.mark.asyncio
async def test_client_wait_for_job_result_fails_quickly(mock_appconfig_scraper_defaults):
    job_id = "client_wait_fails_003"
    mock_broker_client_wait_fail = AsyncMock(spec=RedisMessageBroker)
    
    failed_status = {"job_id": job_id, "status": "failed", "error_message": "Client job processing error"}
    mock_broker_client_wait_fail.get_job_status = AsyncMock(return_value=failed_status)

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_asyncio_sleep_not_called:
        result = await scraper_module.wait_for_job_result(
            job_id, mock_broker_client_wait_fail, timeout_seconds=0.1, poll_interval_seconds=0.01
        )
        assert result == failed_status
        mock_broker_client_wait_fail.get_job_status.assert_called_once()
        mock_asyncio_sleep_not_called.assert_not_called()


@pytest.mark.asyncio
async def test_client_wait_for_job_result_job_id_none(mock_appconfig_scraper_defaults, mock_scraper_logger):
    mock_broker_client_wait_none = AsyncMock(spec=RedisMessageBroker) # Not really used if job_id is None
    
    result = await scraper_module.wait_for_job_result(
        None, mock_broker_client_wait_none, timeout_seconds=0.1, poll_interval_seconds=0.01
    )
    
    assert result is None
    mock_broker_client_wait_none.get_job_status.assert_not_called()
    mock_scraper_logger.error.assert_called_once_with("wait_for_job_result called with no job_id.")
    
    mock_client_broker.publish_message.assert_called_once()
    args, _ = mock_client_broker.publish_message.call_args
    published_payload = args[0]
    
    assert published_payload["job_id"] == job_id
    assert published_payload["message_type"] == "scrape_request_efficient" # Correct type
    assert published_payload["payload"]["urls"] == urls_to_scrape
    assert published_payload["payload"]["user_id"] == "user123" # Check override
    assert published_payload["payload"]["priority"] == "high"


@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper.wait_for_job_result') # Patch client's wait_for_job_result
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_client_with_wait_completes(
    mock_uuid4_client, mock_get_client_broker, mock_client_wait_for_result, 
    mock_appconfig_scraper_defaults
):
    job_id = "client_scrape_wait_comp_456"
    message_id = "client_msg_id_scrape_wait_comp"
    mock_uuid4_client.return_value = MagicMock(hex=job_id)

    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    mock_client_broker.publish_message = AsyncMock(return_value=message_id)
    mock_get_client_broker.return_value = mock_client_broker

    completed_status = {"job_id": job_id, "status": "completed", "results": [{"url": "u1", "markdown": "md1"}]}
    mock_client_wait_for_result.return_value = completed_status

    urls_to_scrape = ["http://client.example.com/scrape_wait_comp"]
    
    result = await scraper_module.queue_scrape_request(urls_to_scrape, wait_for_result=True)

    assert result == completed_status
    mock_get_client_broker.assert_called_once()
    mock_client_broker.publish_message.assert_called_once()
    mock_client_wait_for_result.assert_called_once_with(
        job_id, 
        mock_client_broker, # The broker instance
        timeout_seconds=app_config.client.JOB_COMPLETION_TIMEOUT_SECONDS,
        poll_interval_seconds=app_config.client.JOB_STATUS_POLL_INTERVAL_SECONDS
    )


@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper.wait_for_job_result')
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_client_with_wait_times_out(
    mock_uuid4_client, mock_get_client_broker, mock_client_wait_for_result, 
    mock_appconfig_scraper_defaults
):
    job_id = "client_scrape_wait_timeout_789"
    mock_uuid4_client.return_value = MagicMock(hex=job_id)
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    mock_client_broker.publish_message = AsyncMock(return_value="msg_id_timeout")
    mock_get_client_broker.return_value = mock_client_broker

    timeout_status = {"job_id": job_id, "status": "timeout", "error_message": "Client job timed out"}
    mock_client_wait_for_result.return_value = timeout_status
    
    result = await scraper_module.queue_scrape_request(["url"], wait_for_result=True)
    assert result == timeout_status


@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_client_publish_fails(
    mock_uuid4_client, mock_get_client_broker, mock_appconfig_scraper_defaults
):
    mock_uuid4_client.return_value = MagicMock(hex="job_pub_fail")
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    mock_client_broker.publish_message = AsyncMock(return_value=None) # Publish fails
    mock_get_client_broker.return_value = mock_client_broker
    
    returned_job_id = await scraper_module.queue_scrape_request(["url"], wait_for_result=False)
    assert returned_job_id is None


@pytest.mark.asyncio
@patch('viralStoryGenerator.utils.crawl4ai_scraper._get_scrape_message_broker')
@patch('uuid.uuid4')
async def test_queue_scrape_request_client_publish_raises(
    mock_uuid4_client, mock_get_client_broker, mock_appconfig_scraper_defaults
):
    mock_uuid4_client.return_value = MagicMock(hex="job_pub_raise")
    mock_client_broker = AsyncMock(spec=RedisMessageBroker)
    mock_client_broker.publish_message = AsyncMock(side_effect=Exception("Client Redis dead"))
    mock_get_client_broker.return_value = mock_client_broker
    
    returned_job_id = await scraper_module.queue_scrape_request(["url"], wait_for_result=False)
    assert returned_job_id is None
