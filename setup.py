# setup.py
from setuptools import setup, find_packages

setup(
    name="viralStoryGenerator",
    version="0.1.2",
    description="Generate short story scripts via a local LLM endpoint.",
    packages=find_packages(),
    install_requires=[
        "requests==2.32.3",
        "python-dotenv==1.1.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "viralStoryGenerator=viralStoryGenerator.main:cli_main",
        ],
    },
)
