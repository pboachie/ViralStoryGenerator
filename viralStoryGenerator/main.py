# viralStoryGenerator/main.py
import logging
from viralStoryGenerator.src.cli import cli_main


if __name__ == "__main__":

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger(__name__)

    _logger.info("Starting the Viral Story Generator CLI...")
    cli_main()
