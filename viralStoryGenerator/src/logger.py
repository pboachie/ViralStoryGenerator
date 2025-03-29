# viralStoryGenerator/src/logger.py
from dotenv import load_dotenv
import logging
import os

# Create a logger
load_dotenv()
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
if log_level == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif log_level == "INFO":
    logger.setLevel(logging.INFO)
elif log_level == "WARNING":
    logger.setLevel(logging.WARNING)
elif log_level == "ERROR":
    logger.setLevel(logging.ERROR)
elif log_level == "CRITICAL":
    logger.setLevel(logging.CRITICAL)
else:
    logger.setLevel(logging.DEBUG)  # Default to DEBUG if LOG_LEVEL is not recognized

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to console handler
ch.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(ch)

# Create a file handler for production environment
if os.environ.get("ENVIRONMENT") == "production":
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Prevent duplicate logs by disabling propagation to the root logger
logger.propagate = False

# How to use:
# from viralStoryGenerator.src.logger import logger
# logger.debug('This is a debug message')