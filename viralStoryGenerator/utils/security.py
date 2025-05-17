# viralStoryGenerator/utils/security.py

import os
import re
import uuid
import hmac
from typing import Optional, List

import logging
from viralStoryGenerator.utils.config import config as app_config

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

def is_safe_filename(path_or_filename: str) -> bool:
    """
    Validate that the basename of a given path or filename string is safe.
    It checks the component after the last path separator.
    Allows typical filename characters (alphanumeric, underscore, hyphen, period).
    Disallows path traversal ('..') in the basename component.

    Args:
        path_or_filename: The string to extract the basename from and validate.

    Returns:
        bool: True if the extracted basename is safe, False otherwise
    """
    if not path_or_filename:
        _logger.debug(f"Input to is_safe_filename is empty.")
        return False

    filename_component = os.path.basename(path_or_filename)

    if not filename_component:
        _logger.debug(f"Extracted basename from '{path_or_filename}' is empty.")
        return False

    if '..' in filename_component or '/' in filename_component or '\\' in filename_component:
        _logger.debug(f"Unsafe basename detected (path traversal or separators in component '{filename_component}' from input '{path_or_filename}')")
        return False

    if '\0' in filename_component:
        _logger.debug(f"Unsafe basename detected (null byte in component '{filename_component}' from input '{path_or_filename}')")
        return False

    pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    if not pattern.match(filename_component):
        _logger.debug(f"Unsafe basename detected (invalid characters in component '{filename_component}' from input '{path_or_filename}')")
        return False

    MAX_FILENAME_LENGTH = 255
    if len(filename_component) > MAX_FILENAME_LENGTH:
         _logger.debug(f"Unsafe basename detected (component '{filename_component}' from input '{path_or_filename}' is too long)")
         return False

    return True

def is_file_in_directory(file_path: str, directory: str) -> bool:
    try:
        abs_directory = os.path.realpath(os.path.abspath(directory))
        abs_file_path = os.path.realpath(os.path.abspath(file_path))
        return os.path.commonprefix([abs_file_path, abs_directory]) == abs_directory
    except Exception as e:
        _logger.error(f"Error checking file path {file_path} in directory {directory}: {e}")
        return False

def is_valid_uuid(uuid_string: Optional[str]) -> bool:
    if not isinstance(uuid_string, str):
        return False
    try:
        uuid_obj = uuid.UUID(uuid_string)
        return str(uuid_obj) == uuid_string
    except (ValueError, AttributeError, TypeError):
        return False

def is_valid_voice_id(voice_id: Optional[str]) -> bool:
    if not isinstance(voice_id, str):
        return False

    pattern_str = app_config.security.VOICE_ID_PATTERN

    if not pattern_str:
        _logger.debug("No VOICE_ID_PATTERN configured, skipping voice ID validation.")
        return True

    try:
        pattern = re.compile(pattern_str)
        is_match = bool(pattern.match(voice_id))
        if not is_match:
            _logger.debug(f"Invalid voice ID '{voice_id}' for pattern '{pattern_str}'")
        return is_match
    except re.error as e:
        _logger.error(f"Invalid regex pattern for VOICE_ID_PATTERN: {pattern_str} - {e}")
        return False

def sanitize_input(input_string: Optional[str], max_length: Optional[int] = None, remove_chars: Optional[List[str]] = None) -> str:
    if input_string is None:
        return ""

    max_len = max_length if max_length is not None else app_config.security.SANITIZE_MAX_LENGTH
    chars_to_remove = remove_chars if remove_chars is not None else app_config.security.DANGEROUS_CHARS

    sanitized = input_string[:max_len]

    for char in chars_to_remove:
        sanitized = sanitized.replace(char, '')

    sanitized = ' '.join(sanitized.split())

    return sanitized

def validate_path_component(path_component: str) -> bool:
    if not path_component:
        return False

    if '..' in path_component:
        return False

    if '/' in path_component or '\\' in path_component:
        return False

    if '\0' in path_component:
        return False

    pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    if not pattern.match(path_component):
        return False

    return True

def sanitize_for_filename(text: str, max_length: int = 100) -> str:
    if not text: return "untitled"
    sanitized = re.sub(r'[\\/*?:"<>|\0]', '_', text)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('._ ')
    sanitized = sanitized[:max_length]
    return sanitized if sanitized else "sanitized_topic"
