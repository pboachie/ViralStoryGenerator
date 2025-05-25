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
    version="0.1.3",
    description="Generate short story scripts via a local LLM endpoint.",
    packages=find_packages(include=["viralStoryGenerator", "viralStoryGenerator.*"]),
    package_data={
        "viralStoryGenerator": ["src/*", "utils/*", "models/*", "prompts/*"]
    },
    include_package_data=True,
    install_requires=[
        "requests==2.32.3",
        "python-dotenv==1.0.0",
        "crawl4ai==0.6.3",
        "redis==5.2.1",
        "fastapi==0.115.9",
        "uvicorn==0.34.0",
        "pydantic==2.11.2",
        "python-multipart==0.0.6",
        "aiohttp>=3.11.18",
        "asyncio==3.4.3",
        "prometheus-client==0.16.0",
        "beautifulsoup4==4.12.0",
        "lxml~=5.3.0",
        "schedule==1.2.2",
        "pytest==7.3.1",
        "httpx>=0.28.1",
        "boto3==1.37.28",
        "azure-storage-blob==12.16.0",
        "sentence-transformers==2.2.2",
        "chromadb>=1.0.9",
        "pydantic-settings==2.0.0",
        # "torch==2.3.1",
        "tenacity==8.2.3",
        "mypy_boto3_s3>=1.38.0",
        "typing-extensions>=4.13.2"
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "viralStoryGenerator=viralStoryGenerator.main:main",
        ],
    },
    license_files=["LICENSE"]
)