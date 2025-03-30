#/viralStoryGenerator/utils/crawl4ai_scraper.py
import asyncio
import logging
from typing import List, Union, Optional, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def scrape_urls(
    urls: Union[str, List[str]],
    browser_config: Optional[BrowserConfig] = None,
    run_config: Optional[CrawlerRunConfig] = None
) -> List[Tuple[str, Optional[str]]]:
    """
    Scrape the given URLs using Crawl4AI and return the content as Markdown.

    Args:
        urls (Union[str, List[str]]): A single URL or a list of URLs to scrape.
        browser_config (Optional[BrowserConfig]): Configuration for the browser, e.g., headless mode.
        run_config (Optional[CrawlerRunConfig]): Configuration for the crawl run, e.g., extraction strategies.

    Returns:
        List[Tuple[str, Optional[str]]]: A list of tuples, each containing the URL and its corresponding
        Markdown content. If a URL fails to crawl, its Markdown will be None.

    Notes:
        - This function is asynchronous and must be run within an asyncio event loop, e.g., using `asyncio.run()`.
        - Errors during crawling are logged using the `logging` module and do not halt the process.

    Example:
        ```python
        import asyncio
        from crawl4ai_scraper import scrape_urls

        async def main():
            urls = ["https://example.com", "https://another.com"]
            results = await scrape_urls(urls)
            for url, markdown in results:
                if markdown:
                    print(f"Markdown for {url}:\n{markdown[:100]}...")
                else:
                    print(f"Failed to crawl {url}")

        if __name__ == "__main__":
            asyncio.run(main())
        ```
    """
    # Convert single URL string to a list for uniform processing
    if isinstance(urls, str):
        urls = [urls]

    # Initialize the AsyncWebCrawler with optional browser configuration
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Create a list of crawl tasks for each URL with optional run configuration
        tasks = [crawler.arun(url, config=run_config) for url in urls]
        # Execute all tasks concurrently, capturing exceptions without stopping
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results into a list of (url, markdown) tuples
    output = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            # Log errors and append None for failed crawls
            logging.error(f"Error crawling {url}: {result}")
            output.append((url, None))
        else:
            # Append successful crawl results as Markdown
            output.append((url, result.markdown))

    return output
