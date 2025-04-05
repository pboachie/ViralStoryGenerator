# viralStoryGenerator/main.py
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.src.cli import cli_main


if __name__ == "__main__":

    # Configure basic logging
    _logger.debug("Starting the Viral Story Generator CLI...")
    cli_main()
