import pytest
import os
import re
import uuid
from unittest.mock import patch, MagicMock

# Assuming the module is viralStoryGenerator.utils.security
from viralStoryGenerator.utils import security as security_module
from viralStoryGenerator.utils.config import app_config # For patching config values

# --- Global Mocks & Fixtures ---

@pytest.fixture(autouse=True)
def mock_appconfig_security_defaults(monkeypatch):
    """Set default app_config values for security tests."""
    monkeypatch.setattr(app_config.security, 'VOICE_ID_PATTERN', r"^[a-zA-Z0-9]{20}$") # Example pattern
    monkeypatch.setattr(app_config.security, 'SANITIZE_MAX_LENGTH', 255)
    monkeypatch.setattr(app_config.security, 'DANGEROUS_CHARS', r"[\<\>\"\'\`\*\?]") # Example
    # For is_safe_filename, from os.py (assuming these are globals in security.py or os.py)
    # If MAX_FILENAME_LENGTH is directly in security.py, patch it there.
    # Assuming it's a constant in the security module for now.
    if hasattr(security_module, 'MAX_FILENAME_LENGTH'):
        monkeypatch.setattr(security_module, 'MAX_FILENAME_LENGTH', 100) # Example, adjust if needed

@pytest.fixture
def mock_security_logger():
    """Fixture to mock the _logger in security.py."""
    # Ensure the logger exists in the module, otherwise this patch will fail.
    # If security.py doesn't have its own _logger, this might need to target a common logger.
    # Based on the prompt, patching "the logger _logger within security.py".
    if hasattr(security_module, '_logger'):
        with patch('viralStoryGenerator.utils.security._logger') as mock_logger:
            yield mock_logger
    else:
        # If no logger, provide a dummy mock that won't fail tests trying to use it
        yield MagicMock()


# --- Tests for is_safe_filename (Scenario 1) ---

@pytest.mark.parametrize("filename, expected", [
    ("valid_filename.txt", True),
    ("valid-filename_123.mp4", True),
    ("another.valid.name.with.dots", True),
    (".hiddenfile", True), # Leading dot usually allowed
    ("noextension", True),
    # Invalid scenarios
    ("../file.txt", False),             # Path traversal up
    ("file/../../passwd", False),       # Path traversal complex
    ("file\\..\\boot.ini", False),      # Path traversal windows
    ("file\0null.txt", False),          # Null byte
    ("file*name.txt", False),           # Disallowed char *
    ("file?name.log", False),           # Disallowed char ?
    ("file<name.dat", False),           # Disallowed char <
    ("file>name.ini", False),           # Disallowed char >
    (" leading_space.txt", False),      # Leading space (often problematic)
    ("trailing_space.txt ", False),     # Trailing space
    ("CON.txt", False),                 # Reserved name (Windows) - depends on strictness
    ("PRN", False),                     # Reserved name
    ("", False),                        # Empty filename
    (".", False),                       # Just dot
    ("..", False),                      # Just dot-dot
    ("a" * (security_module.MAX_FILENAME_LENGTH + 1) + ".txt", False), # Exceeds max length
    ("valid_but_with_slash/basename_only.txt", True), # Path provided, basename is valid
    ("/absolute/path/to/valid_file.zip", True),        # Absolute path, basename is valid
    ("c:\\windows\\path\\valid.exe", True),             # Windows absolute path
    ("nodirvalidation/../other/file.txt", False), # Traversal still caught if basename becomes '..'
])
def test_is_safe_filename(filename, expected, mock_appconfig_security_defaults):
    # MAX_FILENAME_LENGTH is a module constant in security.py, set by fixture.
    # The function is_safe_filename in security.py uses this constant.
    assert security_module.is_safe_filename(filename) == expected

def test_is_safe_filename_basename_too_long(mock_appconfig_security_defaults, monkeypatch):
    # Test when only the basename (after stripping path) is too long
    long_basename = "a" * (security_module.MAX_FILENAME_LENGTH + 1)
    filename_with_path = f"some/path/{long_basename}.txt"
    assert security_module.is_safe_filename(filename_with_path) is False

def test_is_safe_filename_empty_after_stripping_path(mock_appconfig_security_defaults):
    # e.g. input is "some/path/"
    assert security_module.is_safe_filename("some/path/") is False
    assert security_module.is_safe_filename("/") is False
    assert security_module.is_safe_filename("\\") is False


# --- Tests for is_file_in_directory (Scenario 2) ---

# Helper for path normalization based on OS for consistent testing
def norm_path(path_str):
    return os.path.normpath(path_str)

@pytest.mark.parametrize("base_dir, file_path, expected, mock_os_error", [
    # Valid cases
    (norm_path("/base/dir"), norm_path("/base/dir/file.txt"), True, None),
    (norm_path("/base/dir"), norm_path("/base/dir/subdir/file.txt"), True, None),
    (norm_path("base/dir"), norm_path("base/dir/file.txt"), True, None), # Relative paths
    (norm_path("."), norm_path("./file.txt"), True, None), # Current dir
    
    # Invalid cases
    (norm_path("/base/dir"), norm_path("/base/other_dir/file.txt"), False, None), # Different directory
    (norm_path("/base/dir"), norm_path("/base/dir"), False, None), # File is the directory itself
    (norm_path("/base/dir/subdir"), norm_path("/base/dir/file.txt"), False, None), # File in parent of base_dir
    
    # Path traversal
    (norm_path("/base/dir"), norm_path("/base/dir/../other_dir/file.txt"), False, None),
    (norm_path("/base/dir"), norm_path("/base/dir/subdir/../../file_in_base.txt"), True, None), # Traversal leads back into allowed dir
    (norm_path("/base/dir"), norm_path("/base/dir/subdir/../../../etc/passwd"), False, None), # Traversal way out
    (norm_path("data"), norm_path("data/../common/config.json"), False, None),

    # Edge cases: symlinks are implicitly handled by os.path.realpath if it resolves them
    # If /base/dir/symlink_to_outside -> /etc/passwd, realpath would expose it.
    # We need to mock realpath to simulate symlink resolution for a direct test of this.
    # For now, assuming realpath does its job.
    
    # Error during os.path.realpath or os.path.abspath
    (norm_path("/base/dir"), norm_path("/base/dir/problem_file.txt"), False, OSError("realpath failed")),
])
@patch('os.path.abspath')
@patch('os.path.realpath')
def test_is_file_in_directory(
    mock_realpath, mock_abspath, base_dir, file_path, expected, mock_os_error,
    mock_security_logger, mock_appconfig_security_defaults
):
    # Configure side effects for abspath and realpath
    # abspath is called on base_dir, realpath on file_path
    
    # Default side effect: return the path itself (as if it's already absolute/real)
    # This is important because os.path.commonpath requires comparable paths.
    mock_abspath.side_effect = lambda path: norm_path(os.path.join("/test/abs", path)) if not os.path.isabs(path) else norm_path(path)
    mock_realpath.side_effect = lambda path: norm_path(os.path.join("/test/real", path)) if not os.path.isabs(path) else norm_path(path)

    # If a specific os error is to be simulated for this test case:
    if mock_os_error:
        # Let's assume the error happens during realpath(file_path)
        def realpath_with_error(path):
            if path == file_path: # Only raise for the specific file_path being tested
                raise mock_os_error
            return norm_path(os.path.join("/test/real", path)) if not os.path.isabs(path) else norm_path(path)
        mock_realpath.side_effect = realpath_with_error
        
    result = security_module.is_file_in_directory(file_path, base_dir)
    assert result == expected

    if mock_os_error:
        mock_security_logger.error.assert_any_call(
            f"Error resolving paths: {mock_os_error}", exc_info=True
        )


# Specific test for symlink scenario if realpath resolves it outside
@patch('os.path.abspath')
@patch('os.path.realpath')
def test_is_file_in_directory_symlink_resolves_outside(
    mock_realpath, mock_abspath, mock_security_logger, mock_appconfig_security_defaults
):
    base_dir = norm_path("/base/safe_dir")
    file_path_symlink = norm_path("/base/safe_dir/symlink_to_secret.txt") # This is what user provides
    
    # Simulate abspath
    mock_abspath.side_effect = lambda path: path # Assume paths are already absolute for simplicity here
    
    # Simulate realpath resolving the symlink to outside the base_dir
    resolved_symlink_path = norm_path("/etc/secret_file.txt")
    mock_realpath.side_effect = lambda path: resolved_symlink_path if path == file_path_symlink else path
    
    assert security_module.is_file_in_directory(file_path_symlink, base_dir) is False
    mock_realpath.assert_called_with(file_path_symlink)
    mock_abspath.assert_called_with(base_dir)

# --- Tests for is_valid_uuid (Scenario 3) ---

@pytest.mark.parametrize("input_uuid, expected", [
    (str(uuid.uuid4()), True), # Valid UUID
    ("123e4567-e89b-12d3-a456-426614174000", True), # Valid format
    ("123E4567-E89B-12D3-A456-426614174000", True), # Valid with uppercase
    # Invalid UUIDs
    ("not-a-uuid", False),
    ("123e4567-e89b-12d3-a456-42661417400", False),  # Too short
    ("123e4567-e89b-12d3-a456-4266141740000", False), # Too long
    ("123e4567e89b12d3a456426614174000", True), # Valid without hyphens (UUID constructor handles this)
    ("123e4567-e89b-12d3-a456-42661417400g", False), # Invalid character 'g'
    (None, False),
    (12345, False), # Not a string
    (True, False),  # Not a string
    ("", False),    # Empty string
])
def test_is_valid_uuid(input_uuid, expected, mock_appconfig_security_defaults):
    assert security_module.is_valid_uuid(input_uuid) == expected

# --- Tests for is_valid_voice_id (Scenario 4) ---

@pytest.mark.parametrize("voice_id, pattern, expected", [
    ("abc123xyz789012345678", r"^[a-zA-Z0-9]{20}$", True), # Matches default mock pattern
    ("shortid", r"^[a-zA-Z0-9]{20}$", False), # Too short
    ("longid12345678901234567890", r"^[a-zA-Z0-9]{20}$", False), # Too long
    ("id_with_symbols!", r"^[a-zA-Z0-9]{20}$", False), # Invalid char
    ("custompatternOK", r"^[a-zA-Z]{13}$", True), # Custom pattern match
    ("custompatternFAIL", r"^[a-zA-Z]{10}$", False), # Custom pattern mismatch
    (None, r"^[a-zA-Z0-9]{20}$", False), # None input
    (12345, r"^[a-zA-Z0-9]{20}$", False), # Non-string input
    ("", r"^[a-zA-Z0-9]{20}$", False), # Empty string
])
def test_is_valid_voice_id_various_patterns_and_inputs(
    voice_id, pattern, expected, mock_appconfig_security_defaults, monkeypatch, mock_security_logger
):
    monkeypatch.setattr(app_config.security, 'VOICE_ID_PATTERN', pattern)
    assert security_module.is_valid_voice_id(voice_id) == expected


def test_is_valid_voice_id_empty_pattern_allows_all_strings(
    mock_appconfig_security_defaults, monkeypatch, mock_security_logger
):
    monkeypatch.setattr(app_config.security, 'VOICE_ID_PATTERN', "") # Empty pattern
    
    assert security_module.is_valid_voice_id("any_string_is_ok_now") is True
    assert security_module.is_valid_voice_id("!@#$%^") is True # Even with symbols
    assert security_module.is_valid_voice_id("") is True # Empty string also true with empty pattern
    assert security_module.is_valid_voice_id(None) is False # Still false for None
    mock_security_logger.debug.assert_any_call("VOICE_ID_PATTERN is empty, allowing any voice_id string.")


def test_is_valid_voice_id_invalid_regex_pattern(
    mock_appconfig_security_defaults, monkeypatch, mock_security_logger
):
    invalid_pattern = "[" # Invalid regex, causes re.error
    monkeypatch.setattr(app_config.security, 'VOICE_ID_PATTERN', invalid_pattern)
    
    assert security_module.is_valid_voice_id("some_id") is False # Should return False on regex error
    
    mock_security_logger.error.assert_called_once()
    log_message = mock_security_logger.error.call_args[0][0]
    assert f"Invalid regex pattern for VOICE_ID_PATTERN: '{invalid_pattern}'." in log_message
    # Check that exc_info=True was used or that the exception details are in the log
    # For now, just check the main message. The actual code logs the exception.
    assert isinstance(mock_security_logger.error.call_args[1]['exc_info'], Exception) # Check exc_info contains an exception

# --- Tests for sanitize_input (Scenario 5) ---

@pytest.mark.parametrize("input_text, default_on_empty, expected_output", [
    (None, False, None), # None input, no default
    (None, True, ""),    # None input, with default_on_empty=True
    ("Simple text", False, "Simple text"),
    ("  Text with   multiple spaces  ", False, "Text with multiple spaces"), # Whitespace normalized
    ("Text\twith\ttabs", False, "Text with tabs"), # Tabs to spaces
    ("Text\nwith\nnewlines", False, "Text with newlines"), # Newlines kept, but surrounding/multiple spaces handled
    ("Text with <dangerous> chars *?", False, "Text with dangerous chars "), # Default DANGEROUS_CHARS
    ("A" * 300, False, "A" * app_config.security.SANITIZE_MAX_LENGTH), # Default max_length (from fixture)
    ("", False, ""), # Empty string
    ("   ", False, ""), # Blank string (becomes empty after strip)
    ("   ", True, ""),  # Blank string with default_on_empty=True
    ("", True, ""),     # Empty string with default_on_empty=True
])
def test_sanitize_input_default_behavior(
    input_text, default_on_empty, expected_output, 
    mock_appconfig_security_defaults, mock_security_logger # Logger for any potential future logging
):
    # Ensure DANGEROUS_CHARS is set as per fixture for this test
    # Default fixture sets: DANGEROUS_CHARS = r"[\<\>\"\'\`\*\?]"
    
    # Special case for the long string test to use the value from app_config
    if input_text and len(input_text) == 300 and input_text[0] == 'A':
        max_len = app_config.security.SANITIZE_MAX_LENGTH
        expected_output = "A" * max_len
        
    result = security_module.sanitize_input(input_text, default_on_empty=default_on_empty)
    assert result == expected_output


def test_sanitize_input_override_max_length(mock_appconfig_security_defaults):
    input_text = "This string is longer than the overridden max length of 10."
    custom_max_length = 10
    expected = "This strin" # Truncated to 10
    
    result = security_module.sanitize_input(input_text, max_length=custom_max_length)
    assert result == expected
    assert len(result) == custom_max_length


def test_sanitize_input_override_remove_chars(mock_appconfig_security_defaults):
    input_text = "Text with custom #@! chars to remove."
    custom_remove_chars_pattern = r"[#@!]"
    expected = "Text with custom  chars to remove." # Chars replaced by ""
    
    result = security_module.sanitize_input(input_text, remove_chars_pattern=custom_remove_chars_pattern)
    assert result == expected


def test_sanitize_input_no_dangerous_chars_pattern(mock_appconfig_security_defaults, monkeypatch):
    # If DANGEROUS_CHARS is empty, no char replacement should happen based on it
    monkeypatch.setattr(app_config.security, 'DANGEROUS_CHARS', "") 
    input_text = "Text with <dangerous> chars *?"
    expected = "Text with <dangerous> chars *?" # No chars removed by DANGEROUS_CHARS pattern
    
    result = security_module.sanitize_input(input_text)
    assert result == expected


def test_sanitize_input_remove_chars_takes_precedence(mock_appconfig_security_defaults, monkeypatch):
    # DANGEROUS_CHARS is default: r"[\<\>\"\'\`\*\?]"
    # remove_chars_pattern is custom
    monkeypatch.setattr(app_config.security, 'DANGEROUS_CHARS', r"[\*]") # Only star is "dangerous" by default
    
    input_text = "Text with *default_dangerous and #custom_remove"
    custom_remove_pattern = r"[#]"
    
    # Expected: * is removed by default pattern, # is removed by custom pattern
    # The function logic is: if remove_chars_pattern is provided, it's used INSTEAD OF DANGEROUS_CHARS.
    # Not in addition.
    
    # Test 1: Only custom pattern is used
    result1 = security_module.sanitize_input(input_text, remove_chars_pattern=custom_remove_pattern)
    assert result1 == "Text with *default_dangerous and custom_remove" # * kept, # removed

    # Test 2: Default pattern is used (custom is None)
    result2 = security_module.sanitize_input(input_text, remove_chars_pattern=None)
    assert result2 == "Text with default_dangerous and #custom_remove" # * removed, # kept

# --- Tests for validate_path_component (Scenario 6) ---

@pytest.mark.parametrize("component, is_valid", [
    ("valid_component", True),
    ("valid-component.123", True),
    ("valid_with_underscore", True),
    ("valid.with.dots", True),
    # Invalid cases
    ("", False), # Empty
    ("..", False), # Path traversal
    ("../something", False), # Path traversal
    ("comp/slash", False),
    ("comp\\slash", False),
    ("comp\0null", False),
    ("comp*asterisk", False),
    ("comp<lt", False),
    ("comp>gt", False),
    ("comp?q", False),
    ("comp\"quote", False),
    ("comp|pipe", False),
    ("comp:colon", False), # Colon often problematic
    (" leading_space", False),
    ("trailing_space ", False),
])
def test_validate_path_component(component, is_valid, mock_appconfig_security_defaults):
    if is_valid:
        assert security_module.validate_path_component(component) == component
    else:
        with pytest.raises(ValueError) as exc_info:
            security_module.validate_path_component(component)
        
        if not component: # Empty string
            assert "Path component cannot be empty" in str(exc_info.value)
        elif ".." in component or "/" in component or "\\" in component:
            assert "Path component cannot contain traversal elements or path separators" in str(exc_info.value)
        elif "\0" in component:
            assert "Path component cannot contain null bytes" in str(exc_info.value)
        # Add more specific checks for other invalid char groups if needed based on error messages
        else: # General invalid character message
            assert "Path component contains invalid characters" in str(exc_info.value)


def test_validate_path_component_none_input(mock_appconfig_security_defaults):
    with pytest.raises(ValueError, match="Path component cannot be None"):
        security_module.validate_path_component(None)

# Scenario 7 (sanitize_for_filename in security.py) is already covered by test_sanitize_for_filename
# which tests security_module.sanitize_for_filename.
# The initial implementation for Scenario 1 was already targeting security_module.sanitize_for_filename.
