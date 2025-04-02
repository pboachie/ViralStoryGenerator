# viralStoryGenerator/utils/security.py

import os
import re
import uuid
import hmac
from typing import Optional, List

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as app_config

def is_safe_filename(filename: str) -> bool:
    """
    Validate that a filename is safe and doesn't contain path traversal
    or other potentially harmful patterns. Allows typical filename characters.

    Args:
        filename: The filename to validate

    Returns:
        bool: True if the filename is safe, False otherwise
    """
    if not filename:
        return False

    # 1. Check for path traversal or separators
    if '..' in filename or '/' in filename or '\\' in filename:
        _logger.debug(f"Unsafe filename detected (path traversal): {filename}")
        return False

    # 2. Check for null bytes
    if '\0' in filename:
        _logger.debug(f"Unsafe filename detected (null byte): {filename}")
        return False

    # 3. Check character whitelist (allow alphanumeric, underscore, hyphen, period)
    # Allows for common filename patterns like 'my-file_v2.txt'
    pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    if not pattern.match(filename):
        _logger.debug(f"Unsafe filename detected (invalid characters): {filename}")
        return False

    # 4. Optional: Check for reserved filenames on different OS (e.g., CON, PRN on Windows)
    # reserved_windows = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'LPT1'}
    # if os.name == 'nt' and filename.upper().split('.')[0] in reserved_windows:
    #    _logger.debug(f"Unsafe filename detected (reserved name): {filename}")
    #    return False

    # 5. Optional: Check length
    MAX_FILENAME_LENGTH = 255 # Common filesystem limit
    if len(filename) > MAX_FILENAME_LENGTH:
         _logger.debug(f"Unsafe filename detected (too long): {filename}")
         return False

    return True


def is_file_in_directory(file_path: str, directory: str) -> bool:
    """
    Check if a file path is safely contained within the specified directory.
    Prevents path traversal attacks by resolving real paths.

    Args:
        file_path: The file path to check
        directory: The directory that should contain the file

    Returns:
        bool: True if the file is within the directory, False otherwise
    """
    try:
        # Resolve symlinks and normalize paths to absolute paths
        abs_directory = os.path.realpath(os.path.abspath(directory))
        abs_file_path = os.path.realpath(os.path.abspath(file_path))

        # Check if the file's path starts with the directory's path
        return os.path.commonpath([abs_file_path, abs_directory]) == abs_directory
    except Exception as e:
         _logger.error(f"Error checking file path {file_path} in directory {directory}: {e}")
         return False


def is_valid_uuid(uuid_string: Optional[str]) -> bool:
    """
    Validate that a string is a valid UUID (version 1, 3, 4, or 5).

    Args:
        uuid_string: The string to validate as a UUID

    Returns:
        bool: True if the string is a valid UUID, False otherwise
    """
    if not isinstance(uuid_string, str):
        return False
    try:
        # Attempt to create a UUID object from the string
        uuid_obj = uuid.UUID(uuid_string)
        return str(uuid_obj) == uuid_string
    except (ValueError, AttributeError, TypeError):
        return False


def is_valid_voice_id(voice_id: Optional[str]) -> bool:
    """
    Validate that a string matches the configured regex pattern for a voice ID.
    Relies on `config.security.VOICE_ID_PATTERN`.

    Args:
        voice_id: The voice ID string to validate.

    Returns:
        bool: True if the voice ID is valid according to the pattern, or if no
              pattern is configured (fail open). False otherwise.
    """
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
             _logger.debug(f"Invalid voice ID format: '{voice_id}' does not match pattern '{pattern_str}'")
        return is_match
    except re.error as e:
        _logger.error(f"Invalid regex pattern configured for VOICE_ID_PATTERN: {pattern_str} - {e}")
        return False


def sanitize_input(input_string: Optional[str],
                  max_length: int = None,
                  remove_chars: Optional[List[str]] = None) -> str:
    """
    Basic sanitization for user input strings. Primarily truncates length
    and removes a configurable list of potentially dangerous characters.

    Args:
        input_string: The string to sanitize.
        max_length: Maximum allowed length. Defaults to config.security.SANITIZE_MAX_LENGTH.
        remove_chars: List of characters to remove. Defaults to config.security.DANGEROUS_CHARS.

    Returns:
        str: Sanitized string.

    Warning:
        This function uses a blacklist approach (removing known bad characters),
        which is generally less secure than allowlisting (defining allowed characters).
        It should NOT be relied upon as the primary defense against injection attacks
        (e.g., XSS, SQLi). Prefer strict validation at API boundaries (Pydantic, regex)
        and context-specific output encoding/escaping (e.g., HTML escaping,
        parameterized SQL queries). Use this function cautiously, primarily for
        basic cleanup or limiting input length.
    """
    if input_string is None:
        return ""

    # Use defaults from config if not provided
    max_len = max_length if max_length is not None else app_config.security.SANITIZE_MAX_LENGTH
    chars_to_remove = remove_chars if remove_chars is not None else app_config.security.DANGEROUS_CHARS

    # 1. Truncate to max length
    sanitized = input_string[:max_len]

    # 2. Remove specified dangerous characters
    for char in chars_to_remove:
        sanitized = sanitized.replace(char, '')

    # 3. Optional: Normalize whitespace (e.g., replace multiple spaces with one)
    sanitized = ' '.join(sanitized.split())

    return sanitized


def validate_path_component(path_component: str) -> bool:
    """
    Validate that a single path component (like a folder or file name within a path)
    doesn't contain dangerous patterns like traversal or separators.

    Args:
        path_component: The path component string to validate.

    Returns:
        bool: True if the path component is safe, False otherwise.
    """
    if not path_component:
        return False

    # Check for path traversal attempts
    if '..' in path_component:
        return False

    # Check for path separators (should not be in a single component)
    if '/' in path_component or '\\' in path_component:
        return False

    # Check for null bytes
    if '\0' in path_component:
        return False

    # Optional: Check for problematic characters depending on context/OS
    pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    if not pattern.match(path_component):
       return False

    # Check if it looks like an absolute path (shouldn't be a component)
    # This check is less reliable across OSes, better handled by `is_file_in_directory`
    # if path_component.startswith('/') or path_component.startswith('\\') or ':' in path_component:
    #    return False

    return True

def sanitize_for_filename(text: str, max_length: int = 100) -> str:
    """Removes unsafe characters and shortens text for use in filenames."""
    if not text: return "untitled"
    sanitized = re.sub(r'[\\/*?:"<>|\0]', '_', text)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('._ ')
    sanitized = sanitized[:max_length]
    return sanitized if sanitized else "sanitized_topic"
