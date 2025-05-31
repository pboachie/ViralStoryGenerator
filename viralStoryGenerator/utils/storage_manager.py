# viralStoryGenerator/utils/storage_manager.py

import datetime
import os
import shutil
import uuid
import time
import tempfile
import mimetypes
import hmac
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO, List, Tuple
from urllib.parse import urljoin, urlparse
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.type_defs import PutObjectOutputTypeDef, GetObjectOutputTypeDef, DeleteObjectOutputTypeDef, HeadBucketOutputTypeDef, DeleteObjectsOutputTypeDef
    from azure.storage.blob import BlobServiceClient as AzureBlobServiceClient, ContentSettings as AzureContentSettings, ContainerClient as AzureContainerClient, BlobClient as AzureBlobClient
    from azure.core.exceptions import ResourceNotFoundError as AzureResourceNotFoundError, AzureError as AzureSDKError
    from botocore.response import StreamingBody

# Import config first
from viralStoryGenerator.utils.config import config as appconfig
import logging
from viralStoryGenerator.utils.security import is_safe_filename, is_file_in_directory

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

# Cloud SDKs are optional dependencies
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False
    boto3 = None # type: ignore
    ClientError = Exception # type: ignore
    NoCredentialsError = Exception # type: ignore
    BotoCoreError = Exception # type: ignore
    # Conditional types for when SDK is not available
    if not TYPE_CHECKING:
        S3Client = None # type: ignore
        PutObjectOutputTypeDef = None # type: ignore
        GetObjectOutputTypeDef = None # type: ignore
        DeleteObjectOutputTypeDef = None # type: ignore
        HeadBucketOutputTypeDef = None # type: ignore
        DeleteObjectsOutputTypeDef = None # type: ignore
        StreamingBody = None # type: ignore

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings, ContainerClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError, AzureError
    _AZURE_SDK_AVAILABLE = True
except ImportError:
    _AZURE_SDK_AVAILABLE = False
    BlobServiceClient = None # type: ignore
    ContentSettings = None # type: ignore
    ResourceNotFoundError = Exception # type: ignore
    AzureError = Exception # type: ignore
    if not TYPE_CHECKING:
        AzureBlobServiceClient = None # type: ignore
        AzureContentSettings = None # type: ignore
        AzureContainerClient = None # type: ignore
        AzureBlobClient = None # type: ignore
        AzureResourceNotFoundError = Exception # type: ignore
        AzureSDKError = Exception # type: ignore


class StorageManager:
    """Handles file storage across local, S3, and Azure providers."""

    def __init__(self):
        """Initialize the storage manager based on configured provider."""
        self.provider: str = appconfig.storage.PROVIDER.lower()
        self.base_url: str = appconfig.http.BASE_URL
        self.s3_client: Optional["S3Client"] = None
        self.azure_blob_service_client: Optional["AzureBlobServiceClient"] = None

        _logger.info(f"Initializing StorageManager with provider: {self.provider}")

        # Initialize the appropriate storage provider
        if self.provider == "local":
            self._init_local_storage()
        elif self.provider == "s3":
            if not _BOTO3_AVAILABLE:
                 _logger.error("S3 storage configured, but 'boto3' library is not installed. pip install boto3")
                 raise ImportError("boto3 library is required for S3 storage.")
            self._init_s3_storage()
        elif self.provider == "azure":
            if not _AZURE_SDK_AVAILABLE:
                 _logger.error("Azure storage configured, but 'azure-storage-blob' library is not installed. pip install azure-storage-blob")
                 raise ImportError("azure-storage-blob library is required for Azure storage.")
            self._init_azure_storage()
        else:
            _logger.warning(f"Unsupported storage provider: {self.provider}. Falling back to 'local'.")
            self.provider = "local"
            self._init_local_storage()

    def _get_storage_dir(self, file_type: str) -> str:
        """Gets the absolute local storage directory for a file type."""
        path_map = {
            "audio": appconfig.storage.AUDIO_STORAGE_PATH,
            "story": appconfig.storage.STORY_STORAGE_PATH,
            "storyboard": appconfig.storage.STORYBOARD_STORAGE_PATH,
            "metadata": appconfig.storage.METADATA_STORAGE_PATH,
            "screenshot": appconfig.storage.SCREENSHOT_STORAGE_PATH,
        }
        storage_dir = path_map.get(file_type)
        if not storage_dir:
            _logger.warning(f"Unknown file_type '{file_type}', using default local storage path.")
            storage_dir = appconfig.storage.LOCAL_STORAGE_PATH

        # Ensure the directory exists (already done in _init_local_storage, but safe to repeat)
        # try:
        #     os.makedirs(storage_dir, exist_ok=True)
        # except OSError as e:
        #      _logger.error(f"Failed to create local storage directory '{storage_dir}': {e}")
        #      # Raise an error as we cannot store files
        #      raise IOError(f"Cannot create storage directory: {storage_dir}") from e
        return storage_dir

    def _get_cloud_key(self, file_type: str, filename: str) -> str:
        """Constructs the object key/blob path for cloud storage."""
        safe_filename = filename.lstrip('/')
        # Basic structure: type/filename
        return f"{file_type}/{safe_filename}"

    def _init_local_storage(self):
        """Initialize local file storage directories."""
        try:
            paths_to_create = [
                appconfig.storage.LOCAL_STORAGE_PATH,
                appconfig.storage.AUDIO_STORAGE_PATH,
                appconfig.storage.STORY_STORAGE_PATH,
                appconfig.storage.STORYBOARD_STORAGE_PATH,
                appconfig.storage.METADATA_STORAGE_PATH,
                appconfig.storage.SCREENSHOT_STORAGE_PATH,
            ]
            for path in paths_to_create:
                 os.makedirs(path, exist_ok=True)
                 _logger.debug(f"Ensured local storage directory exists: {path}")
            _logger.info(f"Initialized local storage provider. Base path: {appconfig.storage.LOCAL_STORAGE_PATH}")
        except OSError as e:
            _logger.critical(f"CRITICAL: Failed to create essential local storage directories: {e}")
            raise IOError("Failed to initialize local storage directories.") from e

    def _init_s3_storage(self):
        """Initialize S3 client and check bucket access."""
        if not _BOTO3_AVAILABLE or boto3 is None:
            _logger.error("S3 storage configured, but 'boto3' library is not installed or failed to import.")
            raise ImportError("boto3 library is required for S3 storage.")

        if not all([appconfig.storage.S3_BUCKET_NAME, appconfig.storage.S3_REGION]):
            _logger.error("S3 provider enabled, but S3_BUCKET_NAME or S3_REGION is not configured.")
            raise ValueError("Missing required S3 configuration (bucket, region).")
        _logger.info(f"Initializing S3 storage provider. Bucket: {appconfig.storage.S3_BUCKET_NAME}, Region: {appconfig.storage.S3_REGION}")

        using_config_creds = bool(appconfig.storage.S3_ACCESS_KEY and appconfig.storage.S3_SECRET_KEY)
        if using_config_creds:
             _logger.info("Using S3 credentials provided in configuration.")

        try:
            session = boto3.Session()
            s3_client_args: Dict[str, Any] = {
                'region_name': appconfig.storage.S3_REGION,
                'endpoint_url': appconfig.storage.S3_ENDPOINT_URL or None
            }
            if using_config_creds:
                 s3_client_args['aws_access_key_id'] = appconfig.storage.S3_ACCESS_KEY
                 s3_client_args['aws_secret_access_key'] = appconfig.storage.S3_SECRET_KEY

            self.s3_client = session.client('s3', **s3_client_args)

            # Check bucket existence and accessibility
            if self.s3_client:
                self.s3_client.head_bucket(Bucket=appconfig.storage.S3_BUCKET_NAME)
                _logger.info(f"Successfully connected to S3 bucket: {appconfig.storage.S3_BUCKET_NAME}")

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            _logger.critical(f"S3 Initialization Failed: Could not access bucket '{appconfig.storage.S3_BUCKET_NAME}'. Error: {e}")
            # Optionally fallback to local, but better to fail if S3 is configured but inaccessible
            # self.provider = "local"; self._init_local_storage()
            raise ConnectionError(f"Failed to initialize S3 storage: {e}") from e
        except Exception as e:
             _logger.critical(f"Unexpected error initializing S3: {e}")
             raise ConnectionError("Unexpected error during S3 initialization.") from e


    def _init_azure_storage(self):
        """Initialize Azure Blob Storage client and check container."""
        if not _AZURE_SDK_AVAILABLE or BlobServiceClient is None:
            _logger.error("Azure storage configured, but 'azure-storage-blob' library is not installed or failed to import.")
            raise ImportError("azure-storage-blob library is required for Azure storage.")

        if not all([appconfig.storage.AZURE_ACCOUNT_NAME, appconfig.storage.AZURE_ACCOUNT_KEY, appconfig.storage.AZURE_CONTAINER_NAME]):
            _logger.error("Azure provider enabled, but AZURE_ACCOUNT_NAME, AZURE_ACCOUNT_KEY, or AZURE_CONTAINER_NAME is not configured.")
            raise ValueError("Missing required Azure configuration.")
        _logger.info(f"Initializing Azure storage provider. Account: {appconfig.storage.AZURE_ACCOUNT_NAME}, Container: {appconfig.storage.AZURE_CONTAINER_NAME}")

        try:
            # Construct connection string securely
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={appconfig.storage.AZURE_ACCOUNT_NAME};"
                f"AccountKey={appconfig.storage.AZURE_ACCOUNT_KEY};"
                f"EndpointSuffix=core.windows.net"
            )
            self.azure_blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_name: str = str(appconfig.storage.AZURE_CONTAINER_NAME)
            container_client: "AzureContainerClient" = self.azure_blob_service_client.get_container_client(container_name)

            # Check if container exists, create if needed
            try:
                container_client.get_container_properties()
                _logger.info(f"Successfully connected to Azure Blob container: {container_name}")
            except ResourceNotFoundError:
                _logger.info(f"Azure container '{container_name}' not found, attempting to create...")
                container_client.create_container()
                _logger.info(f"Successfully created Azure Blob container: {container_name}")
            except AzureError as e:
                 # Catch specific Azure errors during property check/creation
                 _logger.critical(f"Azure Initialization Failed: Could not access or create container '{container_name}'. Error: {e}")
                 raise ConnectionError(f"Failed to initialize Azure storage: {e}") from e

        except AzureError as e:
             # Catch errors during client creation itself
             _logger.critical(f"Azure Initialization Failed: Error creating BlobServiceClient. Error: {e}")
             raise ConnectionError(f"Failed to initialize Azure storage: {e}") from e
        except Exception as e:
            _logger.critical(f"Unexpected error initializing Azure: {e}")
            raise ConnectionError("Unexpected error during Azure initialization.") from e

    def _guess_content_type(self, filename: str) -> str:
        """Guess MIME type from filename, default to octet-stream."""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"

    def _get_validated_local_path(self, file_type: str, filename: str, check_exists: bool = False) -> str:
         """Gets and validates the full local path for a file."""
         if not is_safe_filename(filename):
             _logger.error(f"Attempt to use unsafe filename for local storage: {filename}")
             raise ValueError(f"Invalid or unsafe filename: {filename}")

         storage_dir = self._get_storage_dir(file_type)
         file_path = os.path.abspath(os.path.join(storage_dir, filename))

         if not is_file_in_directory(file_path, storage_dir):
             _logger.critical(f"SECURITY BREACH ATTEMPT: Calculated file path '{file_path}' is outside designated storage directory '{storage_dir}'.")
             raise PermissionError("Calculated file path is outside allowed directory.")

         if check_exists and not os.path.exists(file_path):
             raise FileNotFoundError(f"File does not exist: {file_path}")

         return file_path

    def _get_extension_for_type(self, file_type: str) -> str:
        """Suggests a default file extension based on file type."""
        if file_type == "audio":
            return ".mp3" # todo: determine from content_type if possible
        elif file_type == "story":
            return ".txt"
        elif file_type == "storyboard":
            return ".json"
        elif file_type == "metadata":
            return ".json"
        elif file_type == "screenshot":
            return ".png"
        else:
            return "" # Default to no extension if unknown

    def store_file(self, file_data: Union[str, bytes, BinaryIO],
                  file_type: str,
                  filename: Optional[str] = None,
                  content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Stores file data using the configured provider. Returns dict with info or error.
        Generates a unique filename if none is provided.
        """
        if filename is None:
            ext = mimetypes.guess_extension(content_type or '') or self._get_extension_for_type(file_type)
            filename = f"{uuid.uuid4()}{ext}"
            _logger.debug(f"Generated unique filename: {filename}")
        elif not is_safe_filename(filename):
             _logger.error(f"Provided filename '{filename}' is unsafe.")
             return {"error": "Invalid or unsafe filename provided.", "provider": self.provider}

        start_time = time.time()
        result = {"error": "Provider not implemented or failed", "provider": self.provider}

        try:
            # --- Store based on provider ---
            if self.provider == "local":
                result = self._store_file_local(file_data, file_type, filename, content_type)
            elif self.provider == "s3":
                result = self._store_file_s3(file_data, file_type, filename, content_type)
            elif self.provider == "azure":
                result = self._store_file_azure(file_data, file_type, filename, content_type)
            else:
                 _logger.error(f"store_file called with unsupported provider '{self.provider}'")
                 result = self._store_file_local(file_data, file_type, filename, content_type)

            duration = time.time() - start_time
            if "error" not in result:
                _logger.info(f"Stored '{filename}' ({file_type}) via {self.provider}. Time: {duration:.3f}s. Path/Key: {result.get('file_path')}")
            else:
                _logger.error(f"Failed to store '{filename}' ({file_type}) via {self.provider}. Time: {duration:.3f}s. Error: {result.get('error')}")

        except Exception as e:
             duration = time.time() - start_time
             _logger.exception(f"Unexpected error storing file '{filename}' via {self.provider}. Time: {duration:.3f}s. Error: {e}")
             result = {"error": f"Unexpected error during storage: {e}", "provider": self.provider}

        return result

    def _store_file_local(self, file_data: Union[str, bytes, BinaryIO], file_type: str, filename: str, content_type: Optional[str]) -> Dict[str, Any]:
        """Stores a file in the local filesystem."""
        try:
            file_path = self._get_validated_local_path(file_type, filename)
            _logger.debug(f"Writing local file to: {file_path}")

            # Write the file (handle different data types)
            mode = "wb" if isinstance(file_data, bytes) else "w"
            encoding = None if mode == "wb" else "utf-8" # Use UTF-8 for text

            with open(file_path, mode, encoding=encoding) as f:
                if isinstance(file_data, (str, bytes)):
                    f.write(file_data)
                else:
                    # Ensure reading starts from the beginning if it's a file object
                    if hasattr(file_data, 'seek'): file_data.seek(0)
                    shutil.copyfileobj(file_data, f)

            # Construct relative path for URL generation (relative to storage type base)
            relative_path = f"{file_type}/{filename}"
            file_url = urljoin(self.base_url, f"/static/{relative_path}") # Assumes /static mount point

            return {
                "file_path": relative_path,
                "absolute_path": file_path,
                "filename": filename,
                "url": file_url,
                "content_type": content_type or self._guess_content_type(filename),
                "provider": "local"
            }
        except (IOError, OSError, ValueError, PermissionError) as e:
            _logger.error(f"Failed to store file locally at '{filename}': {e}")
            return {"error": f"Local storage failed: {e}", "provider": "local"}


    def _store_file_s3(self, file_data: Union[str, bytes, BinaryIO], file_type: str, filename: str, content_type: Optional[str]) -> Dict[str, Any]:
        """Stores a file in S3."""
        if not self.s3_client: return {"error": "S3 client not initialized.", "provider": "s3"}

        s3_key: str = self._get_cloud_key(file_type, filename)
        guessed_content_type: str = content_type or self._guess_content_type(filename)
        _logger.debug(f"Uploading to S3. Bucket: {appconfig.storage.S3_BUCKET_NAME}, Key: {s3_key}, ContentType: {guessed_content_type}")

        try:
            extra_args: Dict[str, str] = {"ContentType": guessed_content_type}

            if isinstance(file_data, bytes):
                self.s3_client.put_object(Body=file_data, Bucket=str(appconfig.storage.S3_BUCKET_NAME), Key=s3_key, ContentType=guessed_content_type)
            elif isinstance(file_data, str):
                 self.s3_client.put_object(Body=file_data.encode('utf-8'), Bucket=str(appconfig.storage.S3_BUCKET_NAME), Key=s3_key, ContentType=guessed_content_type)
            else: # BinaryIO
                 if hasattr(file_data, 'seek'): file_data.seek(0)
                 self.s3_client.upload_fileobj(file_data, str(appconfig.storage.S3_BUCKET_NAME), s3_key, ExtraArgs=extra_args)

            file_url: Optional[str] = self.get_file_url(s3_key, file_type)

            return {
                "file_path": s3_key,
                "filename": filename,
                "url": file_url,
                "content_type": guessed_content_type,
                "provider": "s3"
            }
        except (ClientError, BotoCoreError) as e:
            _logger.error(f"Failed to store file in S3 (Key: {s3_key}): {e}")
            return {"error": f"S3 storage failed: {e}", "provider": "s3"}

    def _store_file_azure(self, file_data: Union[str, bytes, BinaryIO], file_type: str, filename: str, content_type: Optional[str]) -> Dict[str, Any]:
        """Stores a file in Azure Blob Storage."""
        if not self.azure_blob_service_client or ContentSettings is None:
            return {"error": "Azure client not initialized or ContentSettings not available.", "provider": "azure"}

        blob_path: str = self._get_cloud_key(file_type, filename)
        guessed_content_type: str = content_type or self._guess_content_type(filename)
        container_name: str = str(appconfig.storage.AZURE_CONTAINER_NAME)
        _logger.debug(f"Uploading to Azure Blob. Container: {container_name}, Blob: {blob_path}, ContentType: {guessed_content_type}")

        try:
            blob_client: "AzureBlobClient" = self.azure_blob_service_client.get_blob_client(container=container_name, blob=blob_path)
            azure_content_settings: "AzureContentSettings" = ContentSettings(content_type=guessed_content_type)

            data_to_upload: Union[bytes, BinaryIO]
            if isinstance(file_data, str):
                 data_to_upload = file_data.encode('utf-8')
            elif isinstance(file_data, bytes):
                 data_to_upload = file_data
            else:
                 if hasattr(file_data, 'seek'): file_data.seek(0)
                 data_to_upload = file_data

            # Upload blob
            blob_client.upload_blob(
                 data_to_upload,
                 overwrite=True,
                 content_settings=azure_content_settings
            )

            # Generate URL
            file_url: Optional[str] = self.get_file_url(blob_path, file_type)

            return {
                "file_path": blob_path,
                "filename": filename,
                "url": file_url,
                "content_type": guessed_content_type,
                "provider": "azure"
            }
        except (AzureError, IOError) as e:
            _logger.error(f"Failed to store file in Azure Blob (Blob: {blob_path}): {e}")
            return {"error": f"Azure storage failed: {e}", "provider": "azure"}


    def get_file_url(self, file_path_or_key: str, file_type: str) -> Optional[str]:
        """Gets the publicly accessible URL for a stored file."""
        filename = os.path.basename(file_path_or_key)

        try:
            if self.provider == "local":
                # Use the relative path directly with the static mount and base URL
                # Assumes file_path_or_key is like "audio/somefile.mp3"
                relative_path = file_path_or_key.lstrip('/')
                return urljoin(self.base_url, f"/static/{relative_path}")

            elif self.provider == "s3":
                 # Construct standard S3 URL or use endpoint URL if provided
                 bucket = appconfig.storage.S3_BUCKET_NAME
                 key = file_path_or_key
                 if appconfig.storage.S3_ENDPOINT_URL:
                     # Handle potential trailing slash in endpoint URL
                     endpoint = appconfig.storage.S3_ENDPOINT_URL.rstrip('/')
                     return f"{endpoint}/{bucket}/{key}"
                 else:
                     region = appconfig.storage.S3_REGION
                     return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
                 # TODO: Consider pre-signed URLs for private content:
                 # return self.s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)

            elif self.provider == "azure":
                 account = appconfig.storage.AZURE_ACCOUNT_NAME
                 container = appconfig.storage.AZURE_CONTAINER_NAME
                 blob_path = file_path_or_key
                 return f"https://{account}.blob.core.windows.net/{container}/{blob_path}"
                 # TODO: Consider SAS tokens for private content

            else:
                _logger.error(f"Cannot get URL for unsupported provider: {self.provider}")
                return None
        except Exception as e:
             _logger.exception(f"Error generating file URL for {file_path_or_key}: {e}")
             return None


    def retrieve_file(self, filename: str, file_type: str,
                    start_byte: Optional[int] = None, end_byte: Optional[int] = None) -> Optional[Union[bytes, BinaryIO, "StreamingBody"]]:
        """Retrieves file content, supporting byte ranges for streaming."""
        _logger.debug(f"Retrieving file '{filename}' ({file_type}), range: {start_byte}-{end_byte}")

        base_filename_for_check = os.path.basename(filename)
        if not is_safe_filename(base_filename_for_check):
            _logger.error(f"Attempt to retrieve file with unsafe base filename component: {base_filename_for_check} (from input: {filename})")
            return None

        try:
            if self.provider == "local":
                file_path: str = self._get_validated_local_path(file_type, base_filename_for_check, check_exists=True)
                if not os.path.exists(file_path):
                    _logger.warning(f"Local file not found: {file_path}")
                    return None

                file_size = os.path.getsize(file_path)
                # Validate range if provided
                range_header, length = self._validate_and_get_range(file_size, start_byte, end_byte)
                if length == 0 and start_byte is not None: return b''

                # Open file and seek for range requests
                file_obj: BinaryIO = open(file_path, "rb")
                if start_byte is not None:
                    file_obj.seek(start_byte)
                return file_obj

            elif self.provider == "s3":
                if not self.s3_client: return None
                s3_key: str = self._get_cloud_key(file_type, base_filename_for_check)
                s3_bucket_name = str(appconfig.storage.S3_BUCKET_NAME)
                params: Dict[str, Any] = {'Bucket': s3_bucket_name, 'Key': s3_key}

                # Add Range header if needed (S3 uses HTTP Range spec)
                if start_byte is not None:
                    range_str = f"bytes={start_byte}-"
                    if end_byte is not None:
                         range_str = f"bytes={start_byte}-{end_byte}"
                    params['Range'] = range_str
                    _logger.debug(f"S3 GetObject Range: {range_str}")

                response: "GetObjectOutputTypeDef" = self.s3_client.get_object(**params)
                return response['Body']

            elif self.provider == "azure":
                if not self.azure_blob_service_client: return None
                blob_path: str = self._get_cloud_key(file_type, base_filename_for_check)
                container_name: str = str(appconfig.storage.AZURE_CONTAINER_NAME)
                blob_client: "AzureBlobClient" = self.azure_blob_service_client.get_blob_client(container=container_name, blob=blob_path)

                offset: int = start_byte if start_byte is not None else 0
                length: Optional[int] = None
                if start_byte is not None and end_byte is not None:
                     length = end_byte - start_byte + 1
                elif start_byte is not None:
                     # Read from start_byte to end if end_byte is None
                     length = None
                else: # Full download
                    offset = 0
                    length = None

                _logger.debug(f"Azure Download Blob Range: offset={offset}, length={length}")
                download_stream: "StreamingBody" = blob_client.download_blob(offset=offset, length=length)
                return download_stream

            else:
                return None # Should not happen

        except (ClientError, AzureError, FileNotFoundError, ValueError, PermissionError) as e:
            _logger.error(f"Failed to retrieve file '{filename}' ({file_type}) from {self.provider}: {e}")
            return None
        except Exception as e:
            _logger.exception(f"Unexpected error retrieving file '{filename}': {e}")
            return None

    def retrieve_file_content_as_json(self, filename: str, file_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves file content and parses it as JSON.
        Returns None if the file is not found, cannot be read, or is not valid JSON.
        """
        _logger.debug(f"Attempting to retrieve and parse file '{filename}' of type '{file_type}' as JSON.")
        file_content_or_stream: Optional[Union[bytes, BinaryIO, "StreamingBody"]] = self.retrieve_file(filename, file_type)

        if file_content_or_stream is None:
            _logger.warning(f"File '{filename}' ({file_type}) not found or failed to retrieve for JSON parsing.")
            return None

        raw_content_str = None
        try:
            if isinstance(file_content_or_stream, bytes):
                raw_content_str = file_content_or_stream.decode('utf-8')
            elif hasattr(file_content_or_stream, 'read'):
                content_bytes: bytes = file_content_or_stream.read()
                if isinstance(content_bytes, bytes):
                    raw_content_str = content_bytes.decode('utf-8')
                else:
                    _logger.error(f"Unexpected content type from stream for '{filename}' ({file_type}): {type(content_bytes)}")
                    if hasattr(file_content_or_stream, 'close'):
                        file_content_or_stream.close()
                    return None
                if hasattr(file_content_or_stream, 'close'):
                    file_content_or_stream.close()
            else:
                _logger.error(f"Retrieved unexpected type for file '{filename}' ({file_type}): {type(file_content_or_stream)}")
                return None

            if raw_content_str is None:
                _logger.error(f"Failed to obtain string content for JSON parsing from '{filename}' ({file_type}).")
                return None

            return json.loads(raw_content_str)

        except UnicodeDecodeError as e:
            _logger.error(f"Unicode decoding error for file '{filename}' ({file_type}): {e}")
            return None
        except json.JSONDecodeError as e:
            _logger.error(f"JSON decoding error for file '{filename}' ({file_type}): {e}. Content snippet: {raw_content_str[:200] if raw_content_str else 'N/A'}")
            return None
        except Exception as e:
            _logger.error(f"Unexpected error parsing file '{filename}' ({file_type}) as JSON: {e}")
            return None

    def _validate_and_get_range(self, file_size: int, start_byte: Optional[int], end_byte: Optional[int]) -> Tuple[Optional[str], int]:
        """Helper to validate byte range and calculate length."""
        if start_byte is None:
            return None, file_size

        start = start_byte
        end = end_byte if end_byte is not None else file_size - 1

        if start < 0 or start >= file_size:
            raise ValueError(f"Invalid start byte {start} for file size {file_size}")
        if end < start or end >= file_size:
             raise ValueError(f"Invalid end byte {end} for start {start} and file size {file_size}")

        range_header = f"bytes {start}-{end}/{file_size}"
        length = end - start + 1
        return range_header, length


    def serve_file(self, filename: str, file_type: str) -> Union[str, Dict[str, Any]]:
        """
        Gets information needed to serve a file (e.g., local path or cloud error).
        DEPRECATED? retrieve_file is likely more useful for API endpoints.
        Kept for potential direct use, but needs careful handling.
        """
        _logger.warning("serve_file method called - consider using retrieve_file for API streaming.")
        if not is_safe_filename(filename):
             _logger.error(f"Serve file rejected unsafe filename: {filename}")
             return {"error": "Invalid or unsafe filename"}

        if self.provider == "local":
             try:
                 # Return validated absolute path for direct serving (e.g., by FileResponse)
                 return self._get_validated_local_path(file_type, filename)
             except (ValueError, PermissionError, FileNotFoundError) as e:
                  _logger.error(f"Cannot serve local file '{filename}': {e}")
                  return {"error": str(e)}

        elif self.provider == "s3" or self.provider == "azure":
             _logger.warning(f"Serving cloud file '{filename}' via 'serve_file' is inefficient. Use 'retrieve_file'.")
             # If absolutely needed
             try:
                  # Create a secure temp file (consider cleanup strategy)
                  temp_suffix = os.path.splitext(filename)[1]
                  with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix, prefix=f"{file_type}_") as temp_f:
                     temp_path = temp_f.name

                  # Retrieve full file content (inefficient for large files)
                  file_stream = self.retrieve_file(filename, file_type)
                  if file_stream is None:
                      os.remove(temp_path) # Clean up empty temp file
                      return {"error": "File not found in cloud storage"}

                  # Write stream to temp file
                  with open(temp_path, "wb") as f_out:
                     if hasattr(file_stream, 'read'): # S3 stream
                         shutil.copyfileobj(file_stream, f_out)
                     elif hasattr(file_stream, 'readall'): # Azure stream
                          f_out.write(file_stream.readall())
                     elif isinstance(file_stream, bytes): # Should not happen without range
                          f_out.write(file_stream)
                     else:
                          raise TypeError("Unsupported stream type from retrieve_file")

                  _logger.debug(f"Downloaded cloud file {filename} to temporary location {temp_path} for serving.")
                  return temp_path
             except Exception as e:
                  _logger.exception(f"Failed to download cloud file {filename} to temp location: {e}")
                  if 'temp_path' in locals() and os.path.exists(temp_path):
                       try: os.remove(temp_path)
                       except OSError: pass
                  return {"error": f"Failed to serve file from {self.provider}: {e}"}
        else:
             return {"error": f"Unsupported provider {self.provider}"}


    def delete_file(self, file_path_or_key: str, file_type: str) -> bool:
        """Deletes a file from the configured storage provider."""
        _logger.warning(f"Attempting to delete file/key '{file_path_or_key}' ({file_type}) from {self.provider} storage.")
        filename: str = os.path.basename(file_path_or_key)

        try:
            if self.provider == "local":
                # Use validated local path for deletion
                abs_path_to_delete: str = self._get_validated_local_path(file_type, filename)
                os.remove(abs_path_to_delete)
                _logger.info(f"Deleted local file: {abs_path_to_delete}")

            elif self.provider == "s3":
                 if not self.s3_client: return False
                 s3_key: str = file_path_or_key
                 self.s3_client.delete_object(Bucket=str(appconfig.storage.S3_BUCKET_NAME), Key=s3_key)
                 _logger.info(f"Deleted S3 object: Bucket={appconfig.storage.S3_BUCKET_NAME}, Key={s3_key}")

            elif self.provider == "azure":
                 if not self.azure_blob_service_client: return False
                 blob_path: str = file_path_or_key
                 container_name: str = str(appconfig.storage.AZURE_CONTAINER_NAME)
                 blob_client: "AzureBlobClient" = self.azure_blob_service_client.get_blob_client(container=container_name, blob=blob_path)
                 blob_client.delete_blob()
                 _logger.info(f"Deleted Azure blob: Container={container_name}, Blob={blob_path}")

            else:
                 _logger.error(f"Delete failed: Unsupported provider {self.provider}")
                 return False

            return True

        except (FileNotFoundError, AzureResourceNotFoundError) as e:
             _logger.warning(f"File/object not found during deletion attempt: {file_path_or_key}. Error: {e}")
             return False
        except (ClientError, AzureSDKError, ValueError, PermissionError, OSError) as e:
             _logger.error(f"Failed to delete file/object '{file_path_or_key}' from {self.provider}: {e}")
             return False
        except Exception as e:
             _logger.exception(f"Unexpected error deleting file/object '{file_path_or_key}': {e}")
             return False


    def cleanup_old_files(self, max_age_days: Optional[int] = None) -> int:
        """Deletes files older than max_age_days based on modification time."""
        retention_days: int = max_age_days if max_age_days is not None else appconfig.storage.FILE_RETENTION_DAYS
        if not isinstance(retention_days, int) or retention_days <= 0:
            _logger.info("File retention cleanup skipped (retention_days <= 0).")
            return 0

        deleted_count = 0
        cutoff_timestamp = time.time() - (retention_days * 24 * 60 * 60)
        cutoff_dt = datetime.datetime.fromtimestamp(cutoff_timestamp, tz=datetime.timezone.utc)
        _logger.info(f"Starting storage cleanup. Provider: {self.provider}. Deleting files older than {retention_days} days (before {cutoff_dt.isoformat()}).")

        file_types_to_clean = ["audio", "story", "storyboard", "metadata", "screenshot"]

        try:
            if self.provider == "local":
                # Iterate through configured storage directories
                for file_type in file_types_to_clean:
                    storage_dir = self._get_storage_dir(file_type)
                    _logger.debug(f"Checking local directory for cleanup: {storage_dir}")
                    if not os.path.isdir(storage_dir): continue

                    for filename in os.listdir(storage_dir):
                         file_path = os.path.join(storage_dir, filename)
                         # Check if it's a file and not a symlink before getting mtime
                         if os.path.isfile(file_path) and not os.path.islink(file_path):
                             try:
                                 file_mtime = os.path.getmtime(file_path)
                                 if file_mtime < cutoff_timestamp:
                                     # Validate path again before deleting
                                     if is_file_in_directory(file_path, storage_dir):
                                         os.remove(file_path)
                                         deleted_count += 1
                                         _logger.debug(f"Deleted old local file: {file_path}")
                                     else:
                                          _logger.warning(f"Skipping deletion of suspicious file outside expected directory: {file_path}")
                             except FileNotFoundError:
                                  continue
                             except OSError as e:
                                  _logger.error(f"Failed to delete old local file {file_path}: {e}")

            elif self.provider == "s3":
                 if not self.s3_client: return 0
                 bucket: str = str(appconfig.storage.S3_BUCKET_NAME)
                 for prefix_base in file_types_to_clean:
                     prefix: str = f"{prefix_base}/"
                     _logger.debug(f"Checking S3 prefix for cleanup: {prefix}")
                     paginator = self.s3_client.get_paginator('list_objects_v2')
                     pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
                     keys_to_delete: List[Dict[str,str]] = []
                     for page in pages:
                         if 'Contents' not in page: continue
                         for obj in page['Contents']:
                             # S3 LastModified is timezone-aware (UTC)
                             if obj['LastModified'] < cutoff_dt:
                                 keys_to_delete.append({'Key': obj['Key']})
                                 _logger.debug(f"Marked old S3 object for deletion: {obj['Key']} (Modified: {obj['LastModified']})")

                     # Batch delete (up to 1000 keys per request)
                     for i in range(0, len(keys_to_delete), 1000):
                         batch = keys_to_delete[i:i+1000]
                         if batch:
                             delete_payload: Dict[str, Any] = {'Objects': batch, 'Quiet': True}
                             response: "DeleteObjectsOutputTypeDef" = self.s3_client.delete_objects(Bucket=bucket, Delete=delete_payload)
                             if 'Errors' in response and response['Errors']:
                                  for error in response['Errors']:
                                       _logger.error(f"Failed to delete S3 object {error['Key']}: {error['Code']} - {error['Message']}")
                             deleted_count += len(batch) - len(response.get('Errors', []))
                             _logger.info(f"Batch deleted {len(batch)} S3 objects (encountered {len(response.get('Errors', []))} errors).")


            elif self.provider == "azure":
                 if not self.azure_blob_service_client: return 0
                 container_name: str = str(appconfig.storage.AZURE_CONTAINER_NAME)
                 container_client: "AzureContainerClient" = self.azure_blob_service_client.get_container_client(container_name)
                 for prefix_base in file_types_to_clean:
                     prefix: str = f"{prefix_base}/"
                     _logger.debug(f"Checking Azure prefix for cleanup: {prefix}")
                     blob_list = container_client.list_blobs(name_starts_with=prefix)
                     blobs_to_delete: List[str] = []
                     for blob in blob_list:
                         if TYPE_CHECKING:
                             assert isinstance(blob, AzureBlobClient)
                         # Azure last_modified is timezone-aware (UTC)
                         if blob.last_modified and blob.last_modified < cutoff_dt:
                             blobs_to_delete.append(blob.name)
                             _logger.debug(f"Marked old Azure blob for deletion: {blob.name} (Modified: {blob.last_modified})")

                     # Delete blobs one by one (Azure SDK doesn't have batch delete like S3)
                     for blob_name in blobs_to_delete:
                         try:
                             blob_client_to_delete: "AzureBlobClient" = container_client.get_blob_client(blob_name)
                             blob_client_to_delete.delete_blob()
                             deleted_count += 1
                         except AzureError as e:
                             _logger.error(f"Failed to delete old Azure blob {blob_name}: {e}")


            _logger.info(f"Storage cleanup complete for provider {self.provider}. Deleted {deleted_count} old files/objects.")
            return deleted_count

        except (ClientError, AzureError, BotoCoreError, OSError) as e:
            _logger.error(f"Error during storage cleanup: {e}")
            return 0
        except Exception as e:
             _logger.exception(f"Unexpected error during storage cleanup: {e}")
             return 0


    async def close(self):
        """Placeholder for closing connections (if needed)."""
        _logger.debug("Closing storage manager (no-op for current implementation).")
        # Add cleanup for SDK clients if necessary (usually not required for boto3/azure)
        return True


# Initialize mimetypes module
mimetypes.init()

# Single shared instance (ensure thread-safety if used across threads without care)
# TODO: Consider using dependency injection (FastAPI Depends) instead of global instance if needed.
storage_manager = StorageManager()