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
        "requests==2.32.3",
        "python-dotenv==1.1.0",
        "crawl4ai==0.5.0.post8",
        "redis==5.2.1",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "viralStoryGenerator=viralStoryGenerator.main:cli_main",
        ],
    },
    license_files=["LICENSE"]
)
