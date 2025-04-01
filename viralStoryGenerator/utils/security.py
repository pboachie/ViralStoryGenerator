#!/usr/bin/env python
# viralStoryGenerator/utils/security.py

import os
import re
import uuid
from typing import Optional

from viralStoryGenerator.src.logger import logger as _logger

def is_safe_filename(filename: str) -> bool:
    """
    Validate that a filename is safe and doesn't contain path traversal attempts.

    Args:
        filename: The filename to validate

    Returns:
        bool: True if the filename is safe, False otherwise
    """
    # Check for common path traversal patterns
    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return False

    # Ensure the filename only contains allowed characters
    # Only allow alphanumeric chars, underscore, hyphen, and period (for extension)
    pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    return bool(pattern.match(filename))

def is_file_in_directory(file_path: str, directory: str) -> bool:
    """
    Check if a file path is contained within the specified directory.
    Prevents path traversal attacks.

    Args:
        file_path: The path to check
        directory: The directory that should contain the file

    Returns:
        bool: True if the file is within the directory, False otherwise
    """
    # Normalize paths to handle different path separators and resolve symlinks
    file_path = os.path.realpath(os.path.abspath(file_path))
    directory = os.path.realpath(os.path.abspath(directory))

    # Check if the file's path is within the directory path
    return os.path.commonpath([file_path, directory]) == directory
def is_valid_uuid(uuid_string: str) -> bool:
    """
    Validate that a string is a valid UUID format.

    Args:
        uuid_string: The string to validate as a UUID

    Returns:
        bool: True if the string is a valid UUID, False otherwise
    """
    try:
        uuid_obj = uuid.UUID(uuid_string)
        return str(uuid_obj) == uuid_string
    except (ValueError, AttributeError):
        return False

def is_valid_voice_id(voice_id: str) -> bool:
    """
    Validate that a string matches the expected format for an ElevenLabs voice ID.
    ElevenLabs voice IDs are typically alphanumeric strings of specific length.

    Args:
        voice_id: The voice ID to validate

    Returns:
        bool: True if the voice ID has a valid format
    """
    # Check if we have any configuration about voice ID format
    from viralStoryGenerator.utils.config import config

    # If we have a specific format defined in config, use that instead
    if hasattr(config.security, 'VOICE_ID_PATTERN') and config.security.VOICE_ID_PATTERN:
        pattern = re.compile(config.security.VOICE_ID_PATTERN)
        return bool(pattern.match(voice_id))

    # Default check: ElevenLabs voice IDs are typically 20-character alphanumeric strings
    # This might need to be adjusted based on the actual voice ID format
    return len(voice_id) >= 10 and voice_id.isalnum()

def sanitize_input(input_string: Optional[str], max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        input_string: The string to sanitize
        max_length: Maximum allowed length

    Returns:
        str: Sanitized string
    """
    if input_string is None:
        return ""

    # Truncate to max length
    sanitized = input_string[:max_length]

    # Remove potentially dangerous characters
    dangerous_chars = ['&', '|', ';', '$', '`', '\\']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    return sanitized

def validate_path_component(path_component: str) -> bool:
    """
    Validate that a path component doesn't contain dangerous patterns.

    Args:
        path_component: The path component to validate

    Returns:
        bool: True if the path component is safe
    """
    if not path_component:
        return False

    # Check for path traversal attempts
    if '..' in path_component:
        return False

    # Check for absolute paths
    if path_component.startswith('/') or path_component.startswith('\\'):
        return False

    # Ensure it doesn't contain path separators
    if '/' in path_component or '\\' in path_component:
        return False

    return True
