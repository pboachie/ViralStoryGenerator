# viralStoryGenerator/src/logger.py
from dotenv import load_dotenv
import logging
import os
import sys
import platform
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Define color codes for different logging levels
class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages based on level
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m',      # Reset to default

        # Environment colors
        'development': '\033[35m',  # Magenta
        'production': '\033[31m',   # Red
        'testing': '\033[36m',      # Cyan
        'local': '\033[34m',        # Blue
        'default_env': '\033[0m'    # Default color
    }

    # Check if the system supports colors
    # Windows cmd has limited ANSI support before Windows 10
    COLOR_SUPPORTED = (
        platform.system() != 'Windows'
        or 'ANSICON' in os.environ
        or 'WT_SESSION' in os.environ  # Windows Terminal
        or os.environ.get('TERM_PROGRAM') == 'vscode'
        or 'ConEmuANSI' in os.environ
    )

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, use_colors=None):
        super().__init__(fmt, datefmt, style, validate)
        # Allow explicit override of color detection
        if use_colors is None:
            self.use_colors = self.COLOR_SUPPORTED and sys.stdout.isatty()
        else:
            self.use_colors = use_colors

    def format(self, record):
        # Save original values
        levelname = record.levelname
        message = record.msg

        # Check if this is an environment log message
        if hasattr(record, 'environment'):
            env = record.environment.lower()
            if self.use_colors:
                color = self.COLORS.get(env, self.COLORS['default_env'])
                record.msg = f"{color}[{env.upper()}]{self.COLORS['RESET']} {message}"

        # Add color to the levelname if supported
        if self.use_colors:
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Restore original values
        record.levelname = levelname
        record.msg = message

        return result

# Create a custom logger adapter to handle environment logging
class EnvironmentLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, environment=None):
        super().__init__(logger, extra={})
        self.environment = environment or os.environ.get("ENVIRONMENT", "development").lower()

    def process(self, msg, kwargs):
        kwargs.setdefault('extra', {})
        kwargs['extra']['environment'] = self.environment
        return msg, kwargs

# Create a logger
logger_base = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# Set logging level based on environment variable
if log_level == "DEBUG":
    logger_base.setLevel(logging.DEBUG)
elif log_level == "INFO":
    logger_base.setLevel(logging.INFO)
elif log_level == "WARNING":
    logger_base.setLevel(logging.WARNING)
elif log_level == "ERROR":
    logger_base.setLevel(logging.ERROR)
elif log_level == "CRITICAL":
    logger_base.setLevel(logging.CRITICAL)
else:
    logger_base.setLevel(logging.DEBUG)  # Default to DEBUG if LOG_LEVEL is not recognized

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatters
# Plain formatter for environments that don't support colors
plain_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Color formatter for terminals that support it
color_formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Determine which formatter to use
use_colors = os.environ.get("USE_COLOR_LOGS", "auto").lower()
if use_colors == "auto":
    ch.setFormatter(color_formatter)
elif use_colors in ("yes", "true", "1"):
    # Force color even if terminal doesn't seem to support it
    ch.setFormatter(ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', use_colors=True))
else:
    # Explicitly disable colors
    ch.setFormatter(plain_formatter)

# Add console handler to logger
logger_base.addHandler(ch)

# Create a file handler for production environment
if os.environ.get("ENVIRONMENT") == "production":
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.INFO)
    # Always use plain formatting in log files
    fh.setFormatter(plain_formatter)
    logger_base.addHandler(fh)

# Prevent duplicate logs by disabling propagation to the root logger
logger_base.propagate = False

# Create an environment-aware logger
logger = EnvironmentLoggerAdapter(logger_base)

def log_startup(environment: str = None, version: str = None, storage_provider: str = None):
    """Log application startup information with environment highlighting"""
    env = environment or os.environ.get("ENVIRONMENT", "development").lower()
    ver = version or os.environ.get("APP_VERSION", "0.1.2")
    storage = storage_provider or os.environ.get("STORAGE_PROVIDER", "local")

    logger.debug("Startup event triggered.", extra={'environment': env})
    logger.info(f"Starting Viral Story Generator API v{ver}", extra={'environment': env})
    logger.info(f"Environment: {env}", extra={'environment': env})
    logger.info(f"Storage provider: {storage}", extra={'environment': env})
