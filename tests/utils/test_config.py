import pytest
import os
import importlib
from unittest.mock import patch, MagicMock

# Import the module and the specific items to test
from viralStoryGenerator.utils import config as config_module
from viralStoryGenerator.utils.config import Config, validate_config_on_startup, ConfigError

# --- Fixtures ---

@pytest.fixture
def mock_config_logger():
    """Fixture to mock the _logger in config.py."""
    with patch('viralStoryGenerator.utils.config._logger') as mock_logger:
        yield mock_logger

@pytest.fixture(autouse=True)
def reset_config_singleton(monkeypatch):
    """
    Ensures that the global `app_config` instance in the config_module 
    is reset/reloaded for tests that modify environ vars and expect re-evaluation.
    This is done by setting it to None, so the next access re-evaluates the class.
    Alternatively, could directly call `importlib.reload(config_module)` in tests.
    For now, this targets the instance. If Config class itself caches at definition,
    module reload is better. The current Config class reads env vars when attributes are accessed
    or during __init__, but the global `app_config` is one instance.
    Reloading the module is cleaner for testing load-time behavior.
    """
    # This fixture doesn't need to do anything if tests explicitly reload.
    # If tests rely on re-instantiating Config or if app_config is a dynamic property,
    # then specific test setups will handle it.
    # For now, we'll rely on importlib.reload in each test that needs it.
    yield


# --- Tests for Config class loading ---

# Scenario 1: Default values
def test_config_loading_default_values(mock_config_logger):
    # Ensure a clean environment for specific keys this test cares about
    # or ensure they are unset if they could affect defaults.
    with patch.dict(os.environ, {}, clear=True): # Clear all env vars for this test scope
        # Reload the config module to ensure Config class is re-evaluated
        # and the global app_config instance is recreated with current (empty) env.
        importlib.reload(config_module)
        
        # Access the reloaded app_config
        reloaded_app_config = config_module.app_config

        # Assert default values
        # General
        assert reloaded_app_config.ENVIRONMENT == "development"
        assert reloaded_app_config.LOG_LEVEL == "DEBUG"
        
        # API
        assert reloaded_app_config.api.HOST == "0.0.0.0"
        assert reloaded_app_config.api.PORT == 8000
        
        # Redis (assuming some defaults are present in the Config class)
        assert reloaded_app_config.redis.HOST == "localhost"
        assert reloaded_app_config.redis.PORT == 6379
        assert reloaded_app_config.redis.USE_SENTINEL is False
        
        # LLM
        assert reloaded_app_config.llm.DEFAULT_TEMPERATURE == 0.7
        assert reloaded_app_config.llm.MODEL == "gpt-4-turbo-preview" # Example default
        
        # Booleans
        assert reloaded_app_config.rag.ENABLED is True # Example default
        assert reloaded_app_config.storyboard.ENABLE_IMAGE_GENERATION is True # Example default
        
        # Lists
        assert reloaded_app_config.http.CORS_ORIGINS == ["*"] # Example default

        # Paths (check if they are made absolute based on a mockable project root if needed)
        # For this test, we assume PROJECT_ROOT has a default or is set by env elsewhere if critical.
        # If PROJECT_ROOT itself comes from env, that's tested in override.
        # Here, we check if a path that *has* a default is correctly formed.
        # Example: LOCAL_STORAGE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
        # We need to know what PROJECT_ROOT defaults to or mock it.
        # For simplicity, let's assume PROJECT_ROOT has a usable default for this test.
        # Or, more robustly, mock PROJECT_ROOT for this default test too.
        with patch.object(config_module.Config, 'PROJECT_ROOT', "/mock/project/root"):
             importlib.reload(config_module) # Reload again with patched PROJECT_ROOT
             reloaded_app_config_with_mock_root = config_module.app_config
             expected_local_storage_path = os.path.abspath("/mock/project/root/data")
             assert reloaded_app_config_with_mock_root.storage.LOCAL_STORAGE_PATH == expected_local_storage_path


# Scenario 2: Environment variable overrides (and type conversions)
def test_config_loading_env_var_overrides(mock_config_logger):
    mock_env_vars = {
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "INFO",
        "API_HOST": "127.0.0.1",
        "API_PORT": "8888", # String, should be int
        "REDIS_PORT": "6380", # String, should be int
        "REDIS_USE_SENTINEL": "true", # String, should be bool
        "S3_BUCKET_NAME": "my-test-bucket",
        "ENABLE_IMAGE_GENERATION": "false", # String, should be bool
        "LLM_TEMPERATURE": "0.95", # String, should be float
        "CORS_ORIGINS": "http://localhost:3000,https://example.com", # CSV String for list
        "LOCAL_STORAGE_PATH": "/custom/storage/path", # Absolute path test
        "LLM_RETRY_ATTEMPTS": "5" # String, should be int
    }
    with patch.dict(os.environ, mock_env_vars, clear=True):
        importlib.reload(config_module)
        reloaded_app_config = config_module.app_config

        assert reloaded_app_config.ENVIRONMENT == "production"
        assert reloaded_app_config.LOG_LEVEL == "INFO"
        assert reloaded_app_config.api.HOST == "127.0.0.1"
        assert reloaded_app_config.api.PORT == 8888
        assert reloaded_app_config.redis.PORT == 6380
        assert reloaded_app_config.redis.USE_SENTINEL is True
        assert reloaded_app_config.storage.S3_BUCKET_NAME == "my-test-bucket"
        assert reloaded_app_config.storyboard.ENABLE_IMAGE_GENERATION is False # This is actually app_config.ENABLE_IMAGE_GENERATION
        assert reloaded_app_config.ENABLE_IMAGE_GENERATION is False # Correcting based on Config structure
        assert reloaded_app_config.llm.DEFAULT_TEMPERATURE == 0.95
        assert reloaded_app_config.http.CORS_ORIGINS == ["http://localhost:3000", "https://example.com"]
        assert reloaded_app_config.storage.LOCAL_STORAGE_PATH == "/custom/storage/path" # Should take the absolute path as is
        assert reloaded_app_config.llm.RETRY_ATTEMPTS == 5


# Scenario 3: Boolean conversion tests
@pytest.mark.parametrize("env_val, expected_bool", [
    ("true", True), ("True", True), ("TRUE", True),
    ("1", True), ("yes", True), ("Yes", True),
    ("false", False), ("False", False), ("FALSE", False),
    ("0", False), ("no", False), ("No", False),
    ("random_string", False), # Other strings are False
    ("", False), # Empty string is False
])
def test_config_loading_boolean_conversion(env_val, expected_bool, mock_config_logger):
    # Test with a boolean config, e.g., RAG_ENABLED
    with patch.dict(os.environ, {"RAG_ENABLED": env_val}, clear=True):
        importlib.reload(config_module)
        reloaded_app_config = config_module.app_config
        assert reloaded_app_config.rag.ENABLED == expected_bool


# Scenario 4: List conversion tests
@pytest.mark.parametrize("env_val, expected_list", [
    ("http://a.com,http://b.com", ["http://a.com", "http://b.com"]),
    ("http://a.com", ["http://a.com"]),
    ("  http://a.com  ,  http://b.com  ", ["http://a.com", "http://b.com"]), # Spaces stripped
    ("", []), # Empty string results in empty list
    ("*,http://localhost", ["*", "http://localhost"]),
])
def test_config_loading_list_conversion(env_val, expected_list, mock_config_logger):
    # Test with a list config, e.g., CORS_ORIGINS
    with patch.dict(os.environ, {"CORS_ORIGINS": env_val}, clear=True):
        importlib.reload(config_module)
        reloaded_app_config = config_module.app_config
        assert reloaded_app_config.http.CORS_ORIGINS == expected_list


# Scenario 5: Path handling (LOCAL_STORAGE_PATH made absolute)
def test_config_loading_path_handling_absolute(mock_config_logger, monkeypatch):
    # This test ensures that if a relative path is given for LOCAL_STORAGE_PATH,
    # it's made absolute relative to PROJECT_ROOT.
    # If an absolute path is given, it's used as is.

    # Case 1: Relative path given for LOCAL_STORAGE_PATH
    with patch.dict(os.environ, {"LOCAL_STORAGE_PATH": "my_data_folder"}, clear=True):
        # Mock PROJECT_ROOT for consistent behavior
        monkeypatch.setattr(config_module.Config, 'PROJECT_ROOT', "/mock/project")
        importlib.reload(config_module)
        reloaded_app_config = config_module.app_config
        expected_path_relative = os.path.abspath("/mock/project/my_data_folder")
        assert reloaded_app_config.storage.LOCAL_STORAGE_PATH == expected_path_relative

    # Case 2: Absolute path given for LOCAL_STORAGE_PATH
    abs_path_env = "/an/absolute/path/to/data"
    with patch.dict(os.environ, {"LOCAL_STORAGE_PATH": abs_path_env}, clear=True):
        monkeypatch.setattr(config_module.Config, 'PROJECT_ROOT', "/another/mock/project") # Should not affect if path is absolute
        importlib.reload(config_module)
        reloaded_app_config = config_module.app_config
        assert reloaded_app_config.storage.LOCAL_STORAGE_PATH == abs_path_env


# --- Tests for validate_config_on_startup ---

# Helper to create a mock config object for validation tests
def create_mock_config_for_validation(monkeypatch, **kwargs):
    # Create a fresh Config instance or mock one.
    # For simplicity, we can monkeypatch the global app_config after a reload.
    # This ensures that the instance being validated is the one we're modifying.
    
    # Start with a clean slate by clearing relevant env vars that might interfere
    # with defaults we are trying to test *against*.
    with patch.dict(os.environ, {}, clear=True):
        importlib.reload(config_module) # Reload to apply defaults
        cfg = config_module.app_config

    # Apply overrides from kwargs
    for key_path, value in kwargs.items():
        parts = key_path.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        monkeypatch.setattr(obj, parts[-1], value)
    return cfg


# 6.1: Valid production config
def test_validate_config_valid_production(mock_config_logger, monkeypatch):
    cfg = create_mock_config_for_validation(monkeypatch,
        ENVIRONMENT="production",
        http={'API_KEY_ENABLED': True, 'API_KEY': "prod_key_set", 'CORS_ORIGINS': ["https://app.example.com"]},
        # Assuming other critical configs like LLM_ENDPOINT are set by default or not critical for this specific valid test
        llm={'ENDPOINT': "http://prod-llm-endpoint", 'MODEL': "prod-model"},
        elevenLabs={'ENABLED': True, 'API_KEY': "eleven_prod_key"},
        dalle={'ENABLED': True, 'API_KEY': "dalle_prod_key"},
        storage={'PROVIDER': "s3", 'S3_BUCKET_NAME': "prod-bucket", 'S3_ACCESS_KEY': "key", 'S3_SECRET_KEY': "secret"},
        rag={'ENABLED': False} # Keep RAG disabled to avoid path checks unless specified
    )
    
    try:
        validate_config_on_startup(cfg)
    except ConfigError as e:
        pytest.fail(f"Valid production config raised ConfigError: {e}")
    
    # Check for absence of critical/error logs, or presence of info logs if any
    # For a valid config, there should be no error/critical logs from validation.
    for call_args in mock_config_logger.critical.call_args_list:
        assert False, f"Critical log found for valid config: {call_args[0][0]}"
    for call_args in mock_config_logger.error.call_args_list:
        assert False, f"Error log found for valid config: {call_args[0][0]}"


# 6.2: Production config with missing API key
def test_validate_config_prod_missing_api_key(mock_config_logger, monkeypatch):
    cfg = create_mock_config_for_validation(monkeypatch,
        ENVIRONMENT="production",
        http={'API_KEY_ENABLED': True, 'API_KEY': None} # API key missing
    )
    with pytest.raises(ConfigError, match="API_KEY must be set when API_KEY_ENABLED is True in production."):
        validate_config_on_startup(cfg)


# 6.3: Production config with wildcard CORS
def test_validate_config_prod_wildcard_cors(mock_config_logger, monkeypatch):
    cfg = create_mock_config_for_validation(monkeypatch,
        ENVIRONMENT="production",
        http={'CORS_ORIGINS': ["*"]} # Wildcard CORS
    )
    validate_config_on_startup(cfg) # Should not raise ConfigError, but log critical
    mock_config_logger.critical.assert_any_call(
        "Running in production with wildcard CORS_ORIGINS ('*'). This is insecure."
    )


# 6.4: Missing essential service configs
@pytest.mark.parametrize("service_config_path, service_key_path, expected_log_fragment", [
    ("elevenLabs.ENABLED", "elevenLabs.API_KEY", "ElevenLabs is enabled but API_KEY is not set."),
    ("dalle.ENABLED", "dalle.API_KEY", "DALL-E image generation is enabled but API_KEY is not set."),
    ("llm.ENDPOINT", None, "LLM_ENDPOINT is not configured."), # Test LLM_ENDPOINT missing
    ("llm.MODEL", None, "LLM_MODEL is not configured."),       # Test LLM_MODEL missing
    ("storage.PROVIDER", "storage.S3_BUCKET_NAME", "S3_BUCKET_NAME must be set when STORAGE_PROVIDER is 's3'."),
    ("storage.PROVIDER", "storage.AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_CONNECTION_STRING must be set when STORAGE_PROVIDER is 'azure'."),
])
def test_validate_config_missing_service_keys(
    mock_config_logger, monkeypatch, service_config_path, service_key_path, expected_log_fragment
):
    # Base config for these tests
    base_settings = {
        "ENVIRONMENT": "development", # Avoid prod-specific errors unless testing them
        "llm.ENDPOINT": "http://default-llm", "llm.MODEL": "default-model", # Ensure these are set by default
        "elevenLabs.ENABLED": False, "dalle.ENABLED": False, "storage.PROVIDER": "local", # Defaults
    }
    
    # Enable the service being tested
    base_settings[service_config_path] = True if service_key_path else "http://default-llm" # Enable service or set provider
    if service_config_path == "storage.PROVIDER": # Special handling for provider
         base_settings[service_config_path] = "s3" if "S3" in service_key_path else "azure"
    
    # Make the critical key for that service None
    if service_key_path: # For API keys
        base_settings[service_key_path] = None
    elif service_config_path == "llm.ENDPOINT": # For LLM_ENDPOINT itself being None
        base_settings["llm.ENDPOINT"] = None
    elif service_config_path == "llm.MODEL":
        base_settings["llm.MODEL"] = None


    cfg = create_mock_config_for_validation(monkeypatch, **base_settings)
    
    # For LLM endpoint/model and S3/Azure, these are critical errors
    if "LLM_" in expected_log_fragment or "S3_" in expected_log_fragment or "AZURE_" in expected_log_fragment:
        with pytest.raises(ConfigError, match=expected_log_fragment):
            validate_config_on_startup(cfg)
    else: # For ElevenLabs/DALL-E API keys, it's a warning
        validate_config_on_startup(cfg)
        mock_config_logger.warning.assert_any_call(expected_log_fragment)


# 6.5: Path validation (AUDIO_STORAGE_PATH outside LOCAL_STORAGE_PATH)
# The ConfigError for this was commented out in source; test for log.
@patch('os.path.abspath') # To control path resolution for the test
def test_validate_config_audio_path_outside_local_storage(mock_abspath, mock_config_logger, monkeypatch):
    # Make abspath return paths as needed for the test condition
    def abspath_side_effect(path):
        if path.startswith("/root/local_storage"): # For LOCAL_STORAGE_PATH
            return path 
        if path.startswith("/outside/audio_storage"): # For AUDIO_STORAGE_PATH
            return path
        return os.path.normpath(path) # Default behavior for other paths
    mock_abspath.side_effect = abspath_side_effect

    cfg = create_mock_config_for_validation(monkeypatch,
        storage={'LOCAL_STORAGE_PATH': "/root/local_storage/data", 
                 'AUDIO_STORAGE_PATH': "/outside/audio_storage"} # AUDIO_STORAGE_PATH is outside
    )
    
    validate_config_on_startup(cfg)
    
    mock_config_logger.error.assert_any_call(
        "AUDIO_STORAGE_PATH is configured outside of LOCAL_STORAGE_PATH. This is not recommended. "
        f"Audio Path: {cfg.storage.AUDIO_STORAGE_PATH}, Local Storage Base: {cfg.storage.LOCAL_STORAGE_PATH}"
    )


# 6.6: RAG path creation
@patch('os.path.exists')
@patch('os.makedirs')
def test_validate_config_rag_path_creation(mock_os_makedirs, mock_os_exists, mock_config_logger, monkeypatch):
    rag_db_path = "/mock_rag_db/vector_db"
    cfg = create_mock_config_for_validation(monkeypatch,
        rag={'ENABLED': True, 'VECTOR_DB_PATH': rag_db_path}
    )
    
    mock_os_exists.return_value = False # Simulate path does not exist

    validate_config_on_startup(cfg)

    mock_os_exists.assert_called_once_with(rag_db_path)
    mock_os_makedirs.assert_called_once_with(rag_db_path)
    mock_config_logger.info.assert_any_call(
        f"RAG vector DB path {rag_db_path} does not exist. Creating..."
    )

@patch('os.path.exists', return_value=True) # Path exists
@patch('os.makedirs') # Should not be called
def test_validate_config_rag_path_already_exists(mock_os_makedirs, mock_os_exists, mock_config_logger, monkeypatch):
    rag_db_path = "/existing_rag_db/vector_db"
    cfg = create_mock_config_for_validation(monkeypatch,
        rag={'ENABLED': True, 'VECTOR_DB_PATH': rag_db_path}
    )
    
    validate_config_on_startup(cfg)

    mock_os_exists.assert_called_once_with(rag_db_path)
    mock_os_makedirs.assert_not_called()


@patch('os.path.exists', return_value=False) # Path does not exist
@patch('os.makedirs', side_effect=OSError("Permission denied creating RAG path")) # makedirs fails
def test_validate_config_rag_path_creation_fails(mock_os_makedirs, mock_os_exists, mock_config_logger, monkeypatch):
    rag_db_path = "/uncreatable_rag_db/vector_db"
    cfg = create_mock_config_for_validation(monkeypatch,
        rag={'ENABLED': True, 'VECTOR_DB_PATH': rag_db_path}
    )
    
    with pytest.raises(ConfigError, match=f"Failed to create RAG vector DB directory: {rag_db_path}. Error: Permission denied creating RAG path"):
        validate_config_on_startup(cfg)
    
    mock_os_makedirs.assert_called_once_with(rag_db_path)
    mock_config_logger.error.assert_any_call(
        f"Failed to create RAG vector DB directory: {rag_db_path}. Error: Permission denied creating RAG path"
    )
