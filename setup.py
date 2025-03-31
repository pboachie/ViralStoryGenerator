# setup.py
from setuptools import setup, find_packages

# Citation for crawl4ai:
# @software{crawl4ai2024,
#   author = {UncleCode},
#   title = {Crawl4AI: Open-source LLM Friendly Web Crawler & Scraper},
#   year = {2024},
#   publisher = {GitHub},
#   journal = {GitHub Repository},
#   howpublished = {\url{https://github.com/unclecode/crawl4ai}},
#   commit = 9c58e4c
# }

setup(
    name="viralStoryGenerator",
    version="0.1.2",
    description="Generate short story scripts via a local LLM endpoint.",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.2",
        "python-dotenv>=1.0.0",
        "crawl4ai==0.5.0.post8",
        "redis>=4.5.4",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.1",
        "pydantic>=1.10.7",
        "starlette==0.36.3",
        "python-multipart>=0.0.6",
        "aiohttp>=3.8.4",
        "asyncio==3.4.3",
        "prometheus-client>=0.16.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.2",
        "schedule>=1.2.0",
        "pytest>=7.3.1",
        "httpx>=0.24.0",
        "boto3>=1.26.114",
        "azure-storage-blob>=12.16.0",
        "prometheus_client>=0.16.0"
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "viralStoryGenerator=viralStoryGenerator.main:main",
        ],
    },
    license_files=["LICENSE"]
)
