import pytest
import os
import json
import uuid
import shutil
import tempfile
import mimetypes
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, mock_open, AsyncMock # AsyncMock if any async methods are used

# Assuming the module is viralStoryGenerator.utils.storage_manager
from viralStoryGenerator.utils import storage_manager as sm_module
from viralStoryGenerator.utils.config import app_config # For patching config values
# Import specific exceptions if they are caught
from botocore.exceptions import ClientError, NoCredentialsError 
# from azure.core.exceptions import AzureError # Example, adjust if specific Azure errors are caught

# --- Global Mocks & Fixtures ---

MOCK_BASE_DIR = "/tmp/mock_test_storage_base"
MOCK_STORIES_DIR_NAME = "test_stories"
MOCK_LOCAL_STORAGE_PATH = os.path.join(MOCK_BASE_DIR, MOCK_STORIES_DIR_NAME)

@pytest.fixture(autouse=True)
def mock_appconfig_storage_defaults(monkeypatch):
    """Set default app_config values for storage_manager tests."""
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "local") # Default to local for safety
    monkeypatch.setattr(app_config.storage, 'BASE_DIR', MOCK_BASE_DIR)
    monkeypatch.setattr(app_config.storage, 'STORIES_DIR_NAME', MOCK_STORIES_DIR_NAME)
    # LOCAL_STORAGE_PATH is derived, will be effectively MOCK_LOCAL_STORAGE_PATH
    
    # S3 specific
    monkeypatch.setattr(app_config.storage, 'S3_BUCKET_NAME', "test-s3-bucket")
    monkeypatch.setattr(app_config.storage, 'S3_ACCESS_KEY', "test_s3_access_key")
    monkeypatch.setattr(app_config.storage, 'S3_SECRET_KEY', "test_s3_secret_key")
    monkeypatch.setattr(app_config.storage, 'S3_ENDPOINT_URL', None)
    monkeypatch.setattr(app_config.storage, 'S3_REGION', "us-east-1")
    monkeypatch.setattr(app_config.storage, 'S3_PUBLIC_URL_PREFIX', "https://test-s3-bucket.s3.amazonaws.com")

    # Azure specific
    monkeypatch.setattr(app_config.storage, 'AZURE_STORAGE_CONNECTION_STRING', "DefaultEndpointsProtocol=https;AccountName=testacc;AccountKey=testkey;EndpointSuffix=core.windows.net")
    monkeypatch.setattr(app_config.storage, 'AZURE_CONTAINER_NAME', "test-azure-container")
    monkeypatch.setattr(app_config.storage, 'AZURE_PUBLIC_URL_PREFIX', "https://testacc.blob.core.windows.net/test-azure-container")

    monkeypatch.setattr(app_config.storage, 'FILE_RETENTION_DAYS', 7)
    monkeypatch.setattr(app_config.security, 'SANITIZE_MAX_LENGTH', 100) # For filename sanitization

    # Ensure mock local storage path exists for tests that might use it by default
    # Use a temporary directory for actual file operations in local tests if needed,
    # but for unit tests, os functions are usually mocked.
    # This setup is more for integration-like tests if os functions weren't mocked.
    # For pure unit tests, os.makedirs will be mocked.
    
    yield
    
    # Clean up the global storage_manager instance in the module if it's a singleton
    # This ensures each test gets a fresh instance if it reloads or re-initializes.
    if hasattr(sm_module, 'storage_manager_instance'): # Assuming it's named this
        monkeypatch.setattr(sm_module, 'storage_manager_instance', None)
    elif hasattr(sm_module, 'storage_manager'): # Or if it's directly named storage_manager
        # If storage_manager is a direct instance of StorageManager, we need to reset its state
        # or ensure tests re-initialize it by reloading the module.
        # For simplicity with a global instance, tests might need to call _init_provider()
        # or a similar method if available, or reload.
        # Let's assume tests will handle reloading or re-patching for provider changes.
        pass
    
    # Cleanup mock local storage if it was actually created by any test
    if os.path.exists(MOCK_LOCAL_STORAGE_PATH) and "mock_test_storage_base" in MOCK_LOCAL_STORAGE_PATH :
        shutil.rmtree(MOCK_BASE_DIR, ignore_errors=True)


@pytest.fixture
def mock_storage_logger():
    """Fixture to mock the _logger in storage_manager.py."""
    with patch('viralStoryGenerator.utils.storage_manager._logger') as mock_logger:
        yield mock_logger

# --- Mocks for SDKs ---
@pytest.fixture
def mock_boto3_client():
    with patch('boto3.client') as mock_boto_client:
        mock_s3_instance = MagicMock()
        mock_s3_instance.head_bucket = MagicMock()
        # Add other S3 methods as needed: put_object, get_object, delete_object, list_objects_v2
        mock_s3_instance.put_object = MagicMock()
        mock_s3_instance.get_object = MagicMock()
        mock_s3_instance.delete_object = MagicMock()
        mock_s3_instance.list_objects_v2 = MagicMock()
        mock_boto_client.return_value = mock_s3_instance
        yield mock_boto_client, mock_s3_instance

@pytest.fixture
def mock_azure_blob_service_client():
    with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service_client_class:
        mock_bsc_instance = MagicMock() # Instance of BlobServiceClient
        mock_container_client = MagicMock() # Instance of ContainerClient
        
        # from_connection_string returns the BlobServiceClient instance
        mock_blob_service_client_class.from_connection_string.return_value = mock_bsc_instance
        
        # get_container_client returns the ContainerClient instance
        mock_bsc_instance.get_container_client.return_value = mock_container_client
        
        # Mock ContainerClient methods
        mock_container_client.get_container_properties = MagicMock()
        mock_container_client.create_container = MagicMock()
        mock_container_client.upload_blob = MagicMock()
        mock_container_client.download_blob = MagicMock()
        mock_container_client.delete_blob = MagicMock()
        mock_container_client.list_blobs = MagicMock()
        
        yield mock_blob_service_client_class, mock_bsc_instance, mock_container_client


# --- Tests for StorageManager Initialization (Scenario 1) ---

# 1.1: Local provider
@patch('os.makedirs')
@patch('os.path.exists')
@patch('os.path.isdir')
def test_storage_manager_init_local_provider(
    mock_os_isdir, mock_os_exists, mock_os_makedirs,
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "local")
    
    # Simulate directories not existing initially to test creation path
    path_existence_map = {}
    def exists_side_effect(path):
        return path_existence_map.get(path, False)
    def isdir_side_effect(path): # Assume if it "exists" for makedirs, it's a dir after
        return path_existence_map.get(path, False)
    def makedirs_side_effect(path, exist_ok=False):
        path_existence_map[path] = True # Simulate it's created
        return None
        
    mock_os_exists.side_effect = exists_side_effect
    mock_os_isdir.side_effect = isdir_side_effect
    mock_os_makedirs.side_effect = makedirs_side_effect

    # Reload storage_manager module to re-initialize the global storage_manager instance
    importlib.reload(sm_module)
    manager = sm_module.storage_manager # Access the global instance

    assert manager.provider_type == "local"
    assert manager.local_storage_path == MOCK_LOCAL_STORAGE_PATH
    
    # Check if makedirs was called for all necessary paths
    # Based on code: LOCAL_STORAGE_PATH, and then specific subdirs like audio, video etc.
    # For this test, let's assume only LOCAL_STORAGE_PATH is checked/created by __init__ directly.
    # If StorageManager._ensure_dir is called for subdirs, those would also be in calls.
    # The current code's __init__ for local calls self._ensure_dir(self.local_storage_path).
    mock_os_makedirs.assert_any_call(MOCK_LOCAL_STORAGE_PATH, exist_ok=True)
    mock_storage_logger.info.assert_any_call(f"Using local storage provider. Base path: {MOCK_LOCAL_STORAGE_PATH}")


@patch('os.makedirs') # Should not be called if dir exists
@patch('os.path.exists', return_value=True) # Dir already exists
@patch('os.path.isdir', return_value=True)   # It is a directory
def test_storage_manager_init_local_provider_dir_exists(
    mock_os_isdir, mock_os_exists, mock_os_makedirs,
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "local")
    importlib.reload(sm_module)
    manager = sm_module.storage_manager
    assert manager.provider_type == "local"
    mock_os_makedirs.assert_not_called() # Not called because path exists and is a dir


# 1.2: S3 provider
@patch('viralStoryGenerator.utils.storage_manager.BOTO3_AVAILABLE', True)
def test_storage_manager_init_s3_provider_success(
    mock_boto3_client, mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "s3")
    mock_boto_client_class, mock_s3_instance = mock_boto3_client
    
    importlib.reload(sm_module) # Re-initialize with S3 provider
    manager = sm_module.storage_manager

    assert manager.provider_type == "s3"
    assert manager.s3_client is mock_s3_instance
    mock_boto_client_class.assert_called_once_with(
        's3',
        aws_access_key_id=app_config.storage.S3_ACCESS_KEY,
        aws_secret_access_key=app_config.storage.S3_SECRET_KEY,
        endpoint_url=app_config.storage.S3_ENDPOINT_URL,
        region_name=app_config.storage.S3_REGION
    )
    mock_s3_instance.head_bucket.assert_called_once_with(Bucket=app_config.storage.S3_BUCKET_NAME)
    mock_storage_logger.info.assert_any_call(
        f"Using S3 storage provider. Bucket: {app_config.storage.S3_BUCKET_NAME}"
    )

@patch('viralStoryGenerator.utils.storage_manager.BOTO3_AVAILABLE', False) # Simulate boto3 not available
def test_storage_manager_init_s3_provider_boto3_not_available(
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "s3")
    with pytest.raises(ImportError, match="boto3 is not installed. Please install it to use S3 storage."):
        importlib.reload(sm_module) # Re-initialize StorageManager instance
        # Accessing sm_module.storage_manager would trigger the __init__ if it's lazy,
        # or if reload correctly re-runs top-level StorageManager() instantiation.
        # The current StorageManager initializes provider on __init__.
    
    # If __init__ itself raises, the logger might not be called from within StorageManager.
    # However, if the check is done before raising, it might log.
    # Based on current code, it raises directly.

def test_storage_manager_init_s3_provider_missing_config(
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "s3")
    monkeypatch.setattr(app_config.storage, 'S3_BUCKET_NAME', None) # Missing bucket name
    
    with pytest.raises(ValueError, match="S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY must be configured for S3 provider."):
        importlib.reload(sm_module)
        _ = sm_module.storage_manager # Access to trigger init if it's lazy or reloaded

# 1.3: Azure provider
@patch('viralStoryGenerator.utils.storage_manager.AZURE_STORAGE_BLOB_AVAILABLE', True)
def test_storage_manager_init_azure_provider_success_container_exists(
    mock_azure_blob_service_client, mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "azure")
    mock_bsc_class, mock_bsc_instance, mock_container_client = mock_azure_blob_service_client
    
    # Simulate container already exists (get_container_properties does not raise an exception)
    mock_container_client.get_container_properties = MagicMock() 

    importlib.reload(sm_module)
    manager = sm_module.storage_manager

    assert manager.provider_type == "azure"
    assert manager.azure_blob_service_client is mock_bsc_instance
    assert manager.azure_container_client is mock_container_client
    
    mock_bsc_class.from_connection_string.assert_called_once_with(app_config.storage.AZURE_STORAGE_CONNECTION_STRING)
    mock_bsc_instance.get_container_client.assert_called_once_with(app_config.storage.AZURE_CONTAINER_NAME)
    mock_container_client.get_container_properties.assert_called_once() # Checked for existence
    mock_container_client.create_container.assert_not_called() # Not called if exists
    mock_storage_logger.info.assert_any_call(
        f"Using Azure Blob Storage provider. Container: '{app_config.storage.AZURE_CONTAINER_NAME}'"
    )


@patch('viralStoryGenerator.utils.storage_manager.AZURE_STORAGE_BLOB_AVAILABLE', True)
def test_storage_manager_init_azure_provider_success_create_container(
    mock_azure_blob_service_client, mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "azure")
    mock_bsc_class, mock_bsc_instance, mock_container_client = mock_azure_blob_service_client
    
    # Simulate container does not exist: get_container_properties raises an AzureError (e.g., ResourceNotFoundError)
    # from azure.core.exceptions import ResourceNotFoundError (if we want to be specific)
    # For simplicity, using a generic Exception that the code catches.
    # The code catches `Exception as e` and checks for "ContainerNotFound" or 404.
    # Let's simulate ResourceNotFoundError which has a status_code attribute.
    mock_resource_not_found_error = MagicMock()
    mock_resource_not_found_error.status_code = 404 
    # Alternatively, make error message contain "ContainerNotFound"
    # mock_resource_not_found_error = Exception("ContainerNotFound")
    
    mock_container_client.get_container_properties.side_effect = mock_resource_not_found_error

    importlib.reload(sm_module)
    manager = sm_module.storage_manager

    assert manager.provider_type == "azure"
    mock_container_client.create_container.assert_called_once()
    mock_storage_logger.info.assert_any_call(
        f"Azure container '{app_config.storage.AZURE_CONTAINER_NAME}' not found, creating it..."
    )


@patch('viralStoryGenerator.utils.storage_manager.AZURE_STORAGE_BLOB_AVAILABLE', True)
def test_storage_manager_init_azure_provider_connection_error(
    mock_azure_blob_service_client, mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "azure")
    mock_bsc_class, _, _ = mock_azure_blob_service_client # We only need the class mock here
    
    connect_exception = Exception("Azure connection failed")
    mock_bsc_class.from_connection_string.side_effect = connect_exception
    
    with pytest.raises(Exception, match="Azure connection failed"): # Should propagate the error
        importlib.reload(sm_module)
        _ = sm_module.storage_manager
    
    mock_storage_logger.error.assert_any_call(
        f"Failed to connect to Azure Blob Storage: {connect_exception}", exc_info=True
    )


def test_storage_manager_init_azure_provider_missing_config(
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "azure")
    monkeypatch.setattr(app_config.storage, 'AZURE_STORAGE_CONNECTION_STRING', None) # Missing connection string
    
    with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER_NAME must be configured for Azure provider."):
        importlib.reload(sm_module)
        _ = sm_module.storage_manager


@patch('viralStoryGenerator.utils.storage_manager.AZURE_STORAGE_BLOB_AVAILABLE', False) # Simulate Azure SDK not available
def test_storage_manager_init_azure_provider_sdk_not_available(
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    monkeypatch.setattr(app_config.storage, 'PROVIDER', "azure")
    with pytest.raises(ImportError, match="azure-storage-blob is not installed. Please install it to use Azure Blob Storage."):
        importlib.reload(sm_module)
        _ = sm_module.storage_manager

# 1.4: Unsupported provider
@patch('os.makedirs') # For local fallback
@patch('os.path.exists', return_value=True) # Assume local paths exist for fallback
@patch('os.path.isdir', return_value=True)
def test_storage_manager_init_unsupported_provider_falls_back_to_local(
    mock_os_isdir, mock_os_exists, mock_os_makedirs,
    mock_storage_logger, mock_appconfig_storage_defaults, monkeypatch
):
    unsupported_provider = "ftp_storage"
    monkeypatch.setattr(app_config.storage, 'PROVIDER', unsupported_provider)
    
    importlib.reload(sm_module)
    manager = sm_module.storage_manager

    assert manager.provider_type == "local" # Should fall back to local
    mock_storage_logger.warning.assert_any_call(
        f"Unsupported storage provider '{unsupported_provider}'. Falling back to 'local' provider."
    )
    # Check if local storage init logic was called (e.g., makedirs for LOCAL_STORAGE_PATH)
    # If LOCAL_STORAGE_PATH already exists (mock_os_exists=True), makedirs might not be called.
    # Let's ensure it tries to ensure the directory.
    mock_os_makedirs.assert_any_call(MOCK_LOCAL_STORAGE_PATH, exist_ok=True)
