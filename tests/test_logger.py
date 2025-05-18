import os
import sys
import logging
import types
import pytest

from viralStoryGenerator.src.logger import ColorFormatter, EnvironmentFilter, log_startup

class DummyRecord:
    def __init__(self, levelname, msg, environment=None):
        self.levelname = levelname
        self.msg = msg
        if environment:
            self.environment = environment

def test_color_formatter_adds_color(monkeypatch):
    # Force color support
    formatter = ColorFormatter('%(levelname)s: %(message)s', use_colors=True)
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="Hello", args=(), exc_info=None
    )
    record.levelname = "INFO"
    formatted = formatter.format(record)
    assert "\033[32mINFO\033[0m" in formatted  # Green for INFO

def test_color_formatter_environment(monkeypatch):
    formatter = ColorFormatter('%(message)s', use_colors=True)
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="EnvMsg", args=(), exc_info=None
    )
    record.environment = "production"
    formatted = formatter.format(record)
    assert "[PRODUCTION]" in formatted

def test_color_formatter_no_color(monkeypatch):
    formatter = ColorFormatter('%(levelname)s: %(message)s', use_colors=False)
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="NoColor", args=(), exc_info=None
    )
    formatted = formatter.format(record)
    assert "\033" not in formatted

def test_color_formatter_unknown_environment():
    formatter = ColorFormatter('%(message)s', use_colors=True)
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="UnknownEnv", args=(), exc_info=None
    )
    record.environment = "unknown_env"
    formatted = formatter.format(record)
    # Should still include [UNKNOWN_ENV] and color code (default_env)
    assert "[UNKNOWN_ENV]" in formatted

def test_color_formatter_unknown_level():
    formatter = ColorFormatter('%(levelname)s: %(message)s', use_colors=True)
    record = logging.LogRecord(
        name="test", level=99, pathname=__file__, lineno=1,
        msg="UnknownLevel", args=(), exc_info=None
    )
    record.levelname = "NOTALEVEL"
    formatted = formatter.format(record)
    # Should use RESET color for unknown level
    assert "\033[0mNOTALEVEL\033[0m" in formatted

def test_color_formatter_error_level_red():
    formatter = ColorFormatter('%(levelname)s: %(message)s', use_colors=True)
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="ErrorMsg", args=(), exc_info=None
    )
    record.levelname = "ERROR"
    formatted = formatter.format(record)
    # Red color code for ERROR is \033[31m
    assert "\033[31mERROR\033[0m" in formatted

def test_environment_filter_sets_default(monkeypatch):
    filt = EnvironmentFilter(default_environment="testing")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="msg", args=(), exc_info=None
    )
    assert filt.filter(record)
    assert record.environment == "testing"

def test_environment_filter_preserves_existing():
    filt = EnvironmentFilter(default_environment="testing")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="msg", args=(), exc_info=None
    )
    record.environment = "production"
    assert filt.filter(record)
    assert record.environment == "production"

def test_environment_filter_no_default(monkeypatch):
    # Remove ENVIRONMENT from os.environ if present
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    filt = EnvironmentFilter(default_environment=None)
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="msg", args=(), exc_info=None
    )
    assert filt.filter(record)
    # Should fallback to "development"
    assert record.environment == "development"

def test_log_startup_logs(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    log_startup(environment="testing", version="1.2.3", storage_provider="s3")
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Starting Viral Story Generator API v1.2.3" in m for m in messages)
    assert any("Environment for startup: testing" in m for m in messages)
    assert any("Storage provider: s3" in m for m in messages)

def test_log_startup_all_messages_present(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    log_startup(environment="dev", version="9.9.9", storage_provider="azure")
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Starting Viral Story Generator API v9.9.9" in m for m in messages)
    assert any("Environment for startup: dev" in m for m in messages)
    assert any("Storage provider: azure" in m for m in messages)

def test_log_startup_storage_provider_azure(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    from viralStoryGenerator.src.logger import log_startup
    log_startup(environment="dev", version="9.9.9", storage_provider="azure")
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Storage provider: azure" in m for m in messages)

def test_log_startup_defaults(caplog):
    caplog.set_level(logging.INFO)
    from viralStoryGenerator.src.logger import log_startup
    log_startup()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Starting Viral Story Generator API v0.1.2" in m for m in messages)
    assert any("Environment for startup: development" in m for m in messages)
    assert any("Storage provider: local" in m for m in messages)

def test_log_startup_storage_provider_local(caplog):
    caplog.set_level(logging.INFO)
    from viralStoryGenerator.src.logger import log_startup
    log_startup()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Storage provider: local" in m for m in messages)

def test_log_startup_storage_provider_local_message_present(caplog):
    """
    Ensure that 'Storage provider: local' is present in the log output when log_startup is called with defaults.
    """
    caplog.set_level(logging.INFO)
    from viralStoryGenerator.src.logger import log_startup
    log_startup()
    assert any("Storage provider: local" in rec.getMessage() for rec in caplog.records)

def test_log_startup_storage_provider_local_case_insensitive(caplog):
    """
    Ensure that the storage provider message is found regardless of case.
    """
    caplog.set_level(logging.INFO)
    from viralStoryGenerator.src.logger import log_startup
    log_startup()
    assert any("storage provider: local" in rec.getMessage().lower() for rec in caplog.records)

def test_file_handler_added_in_production(monkeypatch, tmp_path):
    """
    Test that a FileHandler is added to the logger when ENVIRONMENT is 'production'.
    """
    import importlib
    import viralStoryGenerator.src.logger as logger_module

    # Patch environment to simulate production
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("USE_COLOR_LOGS", "no")

    # Patch FileHandler to use a temp file, but avoid recursion
    log_file = tmp_path / "app.log"
    original_file_handler = logger_module.logging.FileHandler
    def file_handler_patch(filename, *args, **kwargs):
        return original_file_handler(str(log_file), *args, **kwargs)
    monkeypatch.setattr(logger_module.logging, "FileHandler", file_handler_patch)

    # Remove all handlers before reload to ensure clean state
    base_logger = logger_module.logging.getLogger("viralStoryGenerator")
    for handler in base_logger.handlers[:]:
        base_logger.removeHandler(handler)

    # Reload the logger module to re-trigger handler setup
    importlib.reload(logger_module)

    # Check for FileHandler in handlers
    base_logger = logger_module.logging.getLogger("viralStoryGenerator")
    file_handlers = [h for h in base_logger.handlers if isinstance(h, original_file_handler)]
    assert file_handlers, "FileHandler should be present in production environment"

def test_logger_skips_reconfiguration_when_handlers_exist(caplog):
    """
    Test that the logger skips reconfiguration if handlers already exist.
    """
    import importlib
    import viralStoryGenerator.src.logger as logger_module
    import logging

    base_logger_name = "viralStoryGenerator"
    module_logger_name = "viralStoryGenerator.src.logger"

    # Ensure base_logger (parent) has a handler so the 'else' branch in logger.py is hit upon reload.
    base_logger = logging.getLogger(base_logger_name)

    # Store original handlers and propagate status for restoration
    original_handlers = base_logger.handlers[:]
    original_propagate = base_logger.propagate

    # --- MODIFICATION START ---
    # Clear all existing handlers from base_logger to ensure a clean slate.
    # This removes any pre-existing handlers (like the StreamHandler observed in logs)
    # that might interfere with caplog attaching its own handler.
    base_logger.handlers = []

    # Add a NullHandler. This is crucial for the test's logic:
    # 1. It ensures `base_logger.handlers` is not empty.
    # 2. When `logger_module` is reloaded, it will see that `base_logger` has handlers
    #    and therefore log the "Skipping reconfiguration" message, which this test wants to capture.
    # 3. A NullHandler does not produce output or interfere with other handlers like caplog.
    base_logger.addHandler(logging.NullHandler())
    # --- MODIFICATION END ---

    caplog.set_level(logging.DEBUG, logger=base_logger_name)

    module_logger = logging.getLogger(module_logger_name)
    module_logger.setLevel(logging.DEBUG)
    module_logger.propagate = True

    # Print handlers before reload for comparison
    print(f"Handlers on {base_logger_name} BEFORE reload (should include caplog.handler): {base_logger.handlers}")
    print(f"Caplog handler object: {caplog.handler}")
    is_caplog_handler_present_in_base = caplog.handler in base_logger.handlers
    print(f"Is caplog.handler in base_logger.handlers BEFORE reload? {is_caplog_handler_present_in_base}")
    print(f"Root handlers BEFORE reload: {logging.getLogger().handlers}")


    importlib.reload(logger_module)

    # Print handlers after reload to diagnose
    reloaded_base_logger = logging.getLogger(base_logger_name) # Should be the same object
    print(f"Handlers on {base_logger_name} AFTER reload: {reloaded_base_logger.handlers}")
    print(f"Caplog handler object AFTER reload (should be same): {caplog.handler}")

    is_caplog_handler_present_in_base_after = caplog.handler in reloaded_base_logger.handlers
    print(f"Is caplog.handler in reloaded_base_logger.handlers AFTER reload? {is_caplog_handler_present_in_base_after}")
    print(f"Root handlers AFTER reload: {logging.getLogger().handlers}")
    print(f"Is caplog.handler in root_logger.handlers AFTER reload? {caplog.handler in logging.getLogger().handlers}")
    print(f"Caplog records before assertion: {[(r.name, r.levelno, r.getMessage()) for r in caplog.records]}")


    found = False
    # The message is logged by base_logger, and __name__ in logger.py is module_logger_name
    expected_message_content = f"Logger '{base_logger_name}' already has handlers. Skipping reconfiguration by {module_logger_name}"
    for record in caplog.records:
        if record.name == base_logger_name and \
           expected_message_content in record.getMessage():
            found = True
            break
    assert found, f"Expected log message not found in caplog.records. Records: {[(r.name, r.levelno, r.getMessage()) for r in caplog.records]}. Expected content: '{expected_message_content}'"

    # --- MODIFICATION: Restore original handlers and propagate status ---
    for handler in base_logger.handlers[:]:
        base_logger.removeHandler(handler)
    for handler in original_handlers:
        base_logger.addHandler(handler)
    base_logger.propagate = original_propagate
    # --- END MODIFICATION ---





















