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
                color_code = self.COLORS.get(env, self.COLORS['default_env'])
                formatted_message = f"{color_code}[{env.upper()}]{self.COLORS['RESET']} {message}"
            else:
                formatted_message = f"[{env.upper()}] {message}"
        else:
            formatted_message = message

        original_msg = record.msg
        if formatted_message is not message:
            record.msg = formatted_message


        # Add color to the levelname if supported
        if self.use_colors:
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Restore original values
        record.levelname = levelname
        record.msg = original_msg

        return result

class EnvironmentFilter(logging.Filter):
    def __init__(self, default_environment=None):
        super().__init__()
        self.default_environment = default_environment or os.environ.get("ENVIRONMENT", "development").lower()

    def filter(self, record):
        if not hasattr(record, 'environment'):
            record.environment = self.default_environment
        return True

base_app_logger = logging.getLogger("viralStoryGenerator")

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_log_level = getattr(logging, log_level_str, logging.INFO)
base_app_logger.setLevel(numeric_log_level)

if not base_app_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    plain_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s')
    color_formatter_instance = ColorFormatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s', use_colors=None)

    use_colors_env = os.environ.get("USE_COLOR_LOGS", "auto").lower()
    if use_colors_env == "auto":
        ch.setFormatter(color_formatter_instance)
    elif use_colors_env in ("yes", "true", "1"):
        ch.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s', use_colors=True))
    else:
        ch.setFormatter(plain_formatter)

    base_app_logger.addHandler(ch)

    # Create a file handler for production environment
    if os.environ.get("ENVIRONMENT") == "production":
        fh = logging.FileHandler('app.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        base_app_logger.addHandler(fh)

    env_filter = EnvironmentFilter()
    base_app_logger.addFilter(env_filter)

    base_app_logger.propagate = False
    _module_logger_internal = logging.getLogger(__name__)
    _module_logger_internal.debug(f"Logger '{base_app_logger.name}' configured by {__name__}.")
else:
    original_propagate = base_app_logger.propagate
    base_app_logger.propagate = True
    base_app_logger.info(f"Logger '{base_app_logger.name}' already has handlers. Skipping reconfiguration by {__name__}")
    base_app_logger.propagate = original_propagate

_module_logger = logging.getLogger(__name__)

def log_startup(environment: str = None, version: str = None, storage_provider: str = None):
    """Log application startup information with environment highlighting"""
    env_to_log = environment or os.environ.get("ENVIRONMENT", "development").lower()
    ver = version or os.environ.get("APP_VERSION", "0.1.2")
    storage = storage_provider or os.environ.get("STORAGE_PROVIDER", "local")

    logger = logging.getLogger()
    logger.debug("Startup event triggered", extra={'environment': env_to_log})
    logger.info(f"Starting Viral Story Generator API v{ver}", extra={'environment': env_to_log})
    logger.info(f"Environment for startup: {env_to_log}", extra={'environment': env_to_log}) # Message clarifies if different from default
    logger.info(f"Storage provider: {storage}", extra={'environment': env_to_log})

# Other modules should use 'import logging; _logger = logging.getLogger(__name__)'
