#!/usr/bin/env python
# viralStoryGenerator/utils/storage_manager.py

import os
import shutil
import uuid
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO, List
import mimetypes
from urllib.parse import urljoin

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config

class StorageManager:
    """Storage manager for handling file storage across different providers"""

    def __init__(self):
        """Initialize the storage manager based on configured provider"""
        self.provider = config.storage.PROVIDER.lower()
        self.base_url = config.http.BASE_URL

        # Initialize the appropriate storage provider
        if self.provider == "local":
            self._init_local_storage()
        elif self.provider == "s3":
            self._init_s3_storage()
        elif self.provider == "azure":
            self._init_azure_storage()
        else:
            _logger.warning(f"Unsupported storage provider: {self.provider}. Falling back to local storage.")
            self.provider = "local"
            self._init_local_storage()

    def _init_local_storage(self):
        """Initialize local file storage"""
        # Create base storage directories if they don't exist
        os.makedirs(config.storage.LOCAL_STORAGE_PATH, exist_ok=True)
        os.makedirs(config.storage.AUDIO_STORAGE_PATH, exist_ok=True)
        os.makedirs(config.storage.STORY_STORAGE_PATH, exist_ok=True)
        os.makedirs(config.storage.STORYBOARD_STORAGE_PATH, exist_ok=True)
        _logger.info(f"Initialized local storage at {config.storage.LOCAL_STORAGE_PATH}")

    def _init_s3_storage(self):
        """Initialize S3 storage"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError

            # Initialize S3 client
            session = boto3.session.Session()
            self.s3_client = session.client(
                's3',
                region_name=config.storage.S3_REGION,
                aws_access_key_id=config.storage.S3_ACCESS_KEY,
                aws_secret_access_key=config.storage.S3_SECRET_KEY,
                endpoint_url=config.storage.S3_ENDPOINT_URL or None
            )

            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=config.storage.S3_BUCKET_NAME)
                _logger.info(f"Connected to S3 bucket: {config.storage.S3_BUCKET_NAME}")
            except (ClientError, NoCredentialsError) as e:
                _logger.error(f"S3 bucket not accessible: {e}. Falling back to local storage.")
                self.provider = "local"
                self._init_local_storage()

        except ImportError:
            _logger.error("boto3 library not found. Please install with 'pip install boto3'. Falling back to local storage.")
            self.provider = "local"
            self._init_local_storage()

    def _init_azure_storage(self):
        """Initialize Azure Blob Storage"""
        try:
            from azure.storage.blob import BlobServiceClient

            # Initialize Azure blob client
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={config.storage.AZURE_ACCOUNT_NAME};AccountKey={config.storage.AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
            self.azure_blob_service_client = BlobServiceClient.from_connection_string(connection_string)

            # Check if container exists, create if it doesn't
            container_name = config.storage.AZURE_CONTAINER_NAME
            container_client = self.azure_blob_service_client.get_container_client(container_name)
            try:
                container_properties = container_client.get_container_properties()
                _logger.info(f"Connected to Azure Blob container: {container_name}")
            except Exception as e:
                _logger.info(f"Container {container_name} not found, creating...")
                container_client.create_container()
                _logger.info(f"Created Azure Blob container: {container_name}")

        except ImportError:
            _logger.error("Azure Storage SDK not found. Please install with 'pip install azure-storage-blob'. Falling back to local storage.")
            self.provider = "local"
            self._init_local_storage()
        except Exception as e:
            _logger.error(f"Failed to initialize Azure Blob Storage: {e}. Falling back to local storage.")
            self.provider = "local"
            self._init_local_storage()

    def store_file(self, file_data: Union[str, bytes, BinaryIO],
                  file_type: str,
                  filename: Optional[str] = None,
                  content_type: Optional[str] = None) -> Dict[str, str]:
        """
        Store a file in the configured storage provider

        Args:
            file_data: File content as string, bytes or file-like object
            file_type: Type of file ('audio', 'story', 'storyboard')
            filename: Optional filename (will be generated if not provided)
            content_type: Optional MIME type

        Returns:
            Dict with file info including URL
        """
        if filename is None:
            # Generate a unique filename with uuid
            ext = self._get_extension_for_type(file_type, content_type)
            filename = f"{str(uuid.uuid4())}{ext}"

        # Store based on provider
        if self.provider == "local":
            return self._store_file_local(file_data, file_type, filename, content_type)
        elif self.provider == "s3":
            return self._store_file_s3(file_data, file_type, filename, content_type)
        elif self.provider == "azure":
            return self._store_file_azure(file_data, file_type, filename, content_type)
        else:
            # Should never reach here due to initialization checks
            return self._store_file_local(file_data, file_type, filename, content_type)

    def get_file_url(self, file_path: str, file_type: str) -> str:
        """
        Get URL for a stored file

        Args:
            file_path: Storage path of the file
            file_type: Type of file ('audio', 'story', 'storyboard')

        Returns:
            URL to access the file
        """
        if self.provider == "local":
            # For local storage, form a URL using the base URL
            relative_path = self._get_relative_path(file_path, file_type)
            return urljoin(self.base_url, f"/static/{relative_path}")
        elif self.provider == "s3":
            # For S3, either return a pre-signed URL or direct S3 URL
            if config.storage.S3_ENDPOINT_URL:
                return f"{config.storage.S3_ENDPOINT_URL}/{config.storage.S3_BUCKET_NAME}/{file_path}"
            else:
                return f"https://{config.storage.S3_BUCKET_NAME}.s3.{config.storage.S3_REGION}.amazonaws.com/{file_path}"
        elif self.provider == "azure":
            # For Azure, form the blob URL
            container_name = config.storage.AZURE_CONTAINER_NAME
            return f"https://{config.storage.AZURE_ACCOUNT_NAME}.blob.core.windows.net/{container_name}/{file_path}"
        return ""

    def serve_file(self, file_path: str, file_type: str) -> Union[str, Dict[str, Any]]:
        """
        Get file data for serving directly through the API

        Args:
            file_path: Storage path of the file
            file_type: Type of file ('audio', 'story', 'storyboard')

        Returns:
            File path for local files, or file data for cloud storage
        """
        if self.provider == "local":
            # For local files, return the file path for direct serving
            return self._get_local_path(file_type, os.path.basename(file_path))
        elif self.provider == "s3":
            # Download from S3 to a temp file and return the path
            try:
                # Create the file path key for S3 based on file_type
                s3_key = f"{file_type}/{os.path.basename(file_path)}"

                # Create a temporary file to download to
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1])
                temp_path = temp_file.name
                temp_file.close()  # Close but don't delete

                # Download the file from S3
                self.s3_client.download_file(
                    Bucket=config.storage.S3_BUCKET_NAME,
                    Key=s3_key,
                    Filename=temp_path
                )

                _logger.debug(f"Downloaded S3 file {s3_key} to temporary location {temp_path}")
                return temp_path

            except Exception as e:
                _logger.error(f"Failed to serve file from S3: {e}")
                return {"error": str(e)}
        elif self.provider == "azure":
            # Download from Azure to a temp file and return the path
            try:
                # Create the blob path based on file_type
                blob_path = f"{file_type}/{os.path.basename(file_path)}"
                container_name = config.storage.AZURE_CONTAINER_NAME

                # Create a temporary file to download to
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1])
                temp_path = temp_file.name
                temp_file.close()  # Close but don't delete

                # Get a blob client and download
                blob_client = self.azure_blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_path
                )

                with open(temp_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

                _logger.debug(f"Downloaded Azure blob {blob_path} to temporary location {temp_path}")
                return temp_path

            except Exception as e:
                _logger.error(f"Failed to serve file from Azure: {e}")
                return {"error": str(e)}
        return ""

    def delete_file(self, file_path: str, file_type: str) -> bool:
        """
        Delete a file from storage

        Args:
            file_path: Storage path of the file
            file_type: Type of file ('audio', 'story', 'storyboard')

        Returns:
            True if deletion was successful
        """
        try:
            if self.provider == "local":
                os.remove(file_path)
            elif self.provider == "s3":
                self.s3_client.delete_object(
                    Bucket=config.storage.S3_BUCKET_NAME,
                    Key=file_path
                )
            elif self.provider == "azure":
                container_name = config.storage.AZURE_CONTAINER_NAME
                blob_client = self.azure_blob_service_client.get_blob_client(
                    container=container_name,
                    blob=file_path
                )
                blob_client.delete_blob()
            return True
        except Exception as e:
            _logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def cleanup_old_files(self, max_age_days: int = None) -> int:
        """
        Delete files older than specified age from all storage providers

        Args:
            max_age_days: Maximum file age in days (default from config)

        Returns:
            Number of files deleted
        """
        if max_age_days is None:
            max_age_days = config.storage.FILE_RETENTION_DAYS

        # Skip if retention is set to 0 (keep forever)
        if max_age_days <= 0:
            return 0

        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        cutoff_time = current_time - max_age_seconds

        # Log start of cleanup
        _logger.info(f"Starting file cleanup for files older than {max_age_days} days")

        try:
            # Local storage cleanup
            if self.provider == "local":
                for file_type in ["audio", "story", "storyboard"]:
                    storage_dir = self._get_storage_dir(file_type)
                    if not os.path.exists(storage_dir):
                        continue

                    for filename in os.listdir(storage_dir):
                        file_path = os.path.join(storage_dir, filename)
                        if os.path.isfile(file_path):
                            # Check file modification time
                            file_mtime = os.path.getmtime(file_path)
                            if file_mtime < cutoff_time:
                                try:
                                    os.remove(file_path)
                                    deleted_count += 1
                                    _logger.debug(f"Deleted old file {file_path}")
                                except Exception as e:
                                    _logger.error(f"Failed to delete old file {file_path}: {e}")

            # S3 storage cleanup
            elif self.provider == "s3":
                for prefix in ["audio/", "story/", "storyboard/"]:
                    try:
                        # List objects in the bucket with the given prefix
                        paginator = self.s3_client.get_paginator('list_objects_v2')
                        pages = paginator.paginate(
                            Bucket=config.storage.S3_BUCKET_NAME,
                            Prefix=prefix
                        )

                        for page in pages:
                            if 'Contents' not in page:
                                continue

                            for obj in page['Contents']:
                                # Check object's last modified time
                                if obj['LastModified'].timestamp() < cutoff_time:
                                    try:
                                        self.s3_client.delete_object(
                                            Bucket=config.storage.S3_BUCKET_NAME,
                                            Key=obj['Key']
                                        )
                                        deleted_count += 1
                                        _logger.debug(f"Deleted old S3 object: {obj['Key']}")
                                    except Exception as e:
                                        _logger.error(f"Failed to delete old S3 object {obj['Key']}: {e}")
                    except Exception as e:
                        _logger.error(f"Error listing S3 objects for cleanup with prefix {prefix}: {e}")

            # Azure Blob storage cleanup
            elif self.provider == "azure":
                container_name = config.storage.AZURE_CONTAINER_NAME
                container_client = self.azure_blob_service_client.get_container_client(container_name)

                for prefix in ["audio/", "story/", "storyboard/"]:
                    try:
                        # List all blobs with the given prefix
                        blob_list = container_client.list_blobs(name_starts_with=prefix)

                        for blob in blob_list:
                            # Check blob's last modified time
                            if blob.last_modified.timestamp() < cutoff_time:
                                try:
                                    blob_client = container_client.get_blob_client(blob.name)
                                    blob_client.delete_blob()
                                    deleted_count += 1
                                    _logger.debug(f"Deleted old Azure blob: {blob.name}")
                                except Exception as e:
                                    _logger.error(f"Failed to delete old Azure blob {blob.name}: {e}")
                    except Exception as e:
                        _logger.error(f"Error listing Azure blobs for cleanup with prefix {prefix}: {e}")

            _logger.info(f"File cleanup complete: {deleted_count} files removed")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error during file cleanup: {e}")
            return 0

    def retrieve_file(self, filename: str, file_type: str,
                    start_byte: int = None, end_byte: int = None) -> Optional[Union[bytes, BinaryIO]]:
        """
        Retrieve file data from storage with optional range support for streaming

        Args:
            filename: Name of the file to retrieve
            file_type: Type of file ('audio', 'story', 'storyboard')
            start_byte: Optional starting byte for range requests
            end_byte: Optional ending byte for range requests

        Returns:
            File content as bytes, or None if file not found
            For range requests: file-like object that can be iterated for streaming
        """
        try:
            if self.provider == "local":
                # For local storage, read directly from file system
                file_path = self._get_local_path(file_type, filename)
                if not os.path.exists(file_path):
                    return None

                # If range is specified, return a file object for streaming
                if start_byte is not None:
                    file_obj = open(file_path, "rb")
                    file_obj.seek(start_byte)
                    return file_obj

                # Otherwise return the entire file as bytes
                with open(file_path, "rb") as f:
                    return f.read()

            elif self.provider == "s3":
                # Download from S3 with optional range
                s3_key = f"{file_type}/{filename}"
                try:
                    # Prepare range string if needed
                    range_header = None
                    if start_byte is not None:
                        if end_byte is not None:
                            range_header = f"bytes={start_byte}-{end_byte}"
                        else:
                            range_header = f"bytes={start_byte}-"

                    # Get object with optional range
                    params = {'Bucket': config.storage.S3_BUCKET_NAME, 'Key': s3_key}
                    if range_header:
                        params['Range'] = range_header

                    response = self.s3_client.get_object(**params)

                    # For range requests, return the streaming body for efficient handling
                    if start_byte is not None:
                        return response['Body']

                    # For full requests, read all data
                    return response['Body'].read()

                except Exception as e:
                    _logger.error(f"Error downloading from S3: {e}")
                    return None

            elif self.provider == "azure":
                # Download from Azure Blob with optional range
                blob_path = f"{file_type}/{filename}"
                container_name = config.storage.AZURE_CONTAINER_NAME

                try:
                    blob_client = self.azure_blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob_path
                    )

                    # For range requests
                    if start_byte is not None:
                        if end_byte is not None:
                            download_stream = blob_client.download_blob(
                                offset=start_byte,
                                length=end_byte-start_byte+1
                            )
                        else:
                            download_stream = blob_client.download_blob(
                                offset=start_byte
                            )
                        return download_stream

                    # For full downloads
                    download_stream = blob_client.download_blob()
                    return download_stream.readall()

                except Exception as e:
                    _logger.error(f"Error downloading from Azure: {e}")
                    return None

            return None

        except Exception as e:
            _logger.error(f"Failed to retrieve file {filename}: {e}")
            return None

    def _store_file_local(self, file_data: Union[str, bytes, BinaryIO],
                        file_type: str,
                        filename: str,
                        content_type: Optional[str] = None) -> Dict[str, str]:
        """Store a file in local file system"""
        # Get the appropriate storage directory
        storage_dir = self._get_storage_dir(file_type)
        file_path = os.path.join(storage_dir, filename)

        # Write the file based on the type of data
        try:
            if isinstance(file_data, (str, bytes)):
                # It's string or bytes content
                mode = "wb" if isinstance(file_data, bytes) else "w"
                with open(file_path, mode) as f:
                    f.write(file_data)
            else:
                # It's a file-like object
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file_data, f)

            _logger.debug(f"Stored file at {file_path}")

            # Return file info including URL
            relative_path = self._get_relative_path(file_path, file_type)
            return {
                "file_path": file_path,
                "filename": filename,
                "url": urljoin(self.base_url, f"/static/{relative_path}"),
                "content_type": content_type or self._guess_content_type(filename),
                "provider": "local"
            }
        except Exception as e:
            _logger.error(f"Failed to store file locally: {e}")
            return {
                "error": f"Failed to store file: {str(e)}",
                "provider": "local"
            }

    def _store_file_s3(self, file_data: Union[str, bytes, BinaryIO],
                     file_type: str,
                     filename: str,
                     content_type: Optional[str] = None) -> Dict[str, str]:
        """Store a file in S3"""
        # Determine S3 key (path)
        s3_key = f"{file_type}/{filename}"

        try:
            # Prepare data for upload
            if isinstance(file_data, str):
                file_data = file_data.encode('utf-8')

            # Determine content type if not provided
            if content_type is None:
                content_type = self._guess_content_type(filename)

            # Upload to S3
            if isinstance(file_data, bytes):
                self.s3_client.put_object(
                    Bucket=config.storage.S3_BUCKET_NAME,
                    Key=s3_key,
                    Body=file_data,
                    ContentType=content_type
                )
            else:
                # File-like object
                self.s3_client.upload_fileobj(
                    file_data,
                    config.storage.S3_BUCKET_NAME,
                    s3_key,
                    ExtraArgs={"ContentType": content_type}
                )

            _logger.debug(f"Stored file in S3 at {s3_key}")

            # Generate a URL for the S3 object
            if config.storage.S3_ENDPOINT_URL:
                url = f"{config.storage.S3_ENDPOINT_URL}/{config.storage.S3_BUCKET_NAME}/{s3_key}"
            else:
                url = f"https://{config.storage.S3_BUCKET_NAME}.s3.{config.storage.S3_REGION}.amazonaws.com/{s3_key}"

            return {
                "file_path": s3_key,
                "filename": filename,
                "url": url,
                "content_type": content_type,
                "provider": "s3"
            }
        except Exception as e:
            _logger.error(f"Failed to store file in S3: {e}")
            return {
                "error": f"Failed to store file in S3: {str(e)}",
                "provider": "s3"
            }

    def _store_file_azure(self, file_data: Union[str, bytes, BinaryIO],
                        file_type: str,
                        filename: str,
                        content_type: Optional[str] = None) -> Dict[str, str]:
        """Store a file in Azure Blob Storage"""
        # Determine blob path
        blob_path = f"{file_type}/{filename}"
        container_name = config.storage.AZURE_CONTAINER_NAME

        try:
            # Get a blob client
            blob_client = self.azure_blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_path
            )

            # Determine content type if not provided
            if content_type is None:
                content_type = self._guess_content_type(filename)

            # Upload data
            if isinstance(file_data, str):
                file_data = file_data.encode('utf-8')

            if isinstance(file_data, bytes):
                blob_client.upload_blob(file_data, overwrite=True, content_settings={"content_type": content_type})
            else:
                # File-like object
                blob_client.upload_blob(file_data.read(), overwrite=True, content_settings={"content_type": content_type})

            _logger.debug(f"Stored file in Azure Blob at {blob_path}")

            # Generate URL
            url = f"https://{config.storage.AZURE_ACCOUNT_NAME}.blob.core.windows.net/{container_name}/{blob_path}"

            return {
                "file_path": blob_path,
                "filename": filename,
                "url": url,
                "content_type": content_type,
                "provider": "azure"
            }
        except Exception as e:
            _logger.error(f"Failed to store file in Azure Blob: {e}")
            return {
                "error": f"Failed to store file in Azure Blob: {str(e)}",
                "provider": "azure"
            }

    def _get_storage_dir(self, file_type: str) -> str:
        """Get the appropriate storage directory for the file type"""
        if file_type == "audio":
            return config.storage.AUDIO_STORAGE_PATH
        elif file_type == "story":
            return config.storage.STORY_STORAGE_PATH
        elif file_type == "storyboard":
            return config.storage.STORYBOARD_STORAGE_PATH
        else:
            # Default to base storage path
            return config.storage.LOCAL_STORAGE_PATH

    def _get_local_path(self, file_type: str, filename: str) -> str:
        """Get the full local path for a file"""
        storage_dir = self._get_storage_dir(file_type)
        return os.path.join(storage_dir, filename)

    def _get_relative_path(self, file_path: str, file_type: str) -> str:
        """Get the relative path for a file for URL construction"""
        if file_type == "audio":
            return f"audio/{os.path.basename(file_path)}"
        elif file_type == "story":
            return f"stories/{os.path.basename(file_path)}"
        elif file_type == "storyboard":
            return f"storyboards/{os.path.basename(file_path)}"
        else:
            return os.path.basename(file_path)

    def _get_extension_for_type(self, file_type: str, content_type: Optional[str] = None) -> str:
        """Get the appropriate file extension based on file type"""
        if file_type == "audio":
            return ".mp3"
        elif file_type == "story":
            return ".txt"
        elif file_type == "storyboard":
            return ".json"

        # Try to determine from content type
        if content_type:
            ext = mimetypes.guess_extension(content_type)
            if ext:
                return ext

        # Default
        return ""

    def _guess_content_type(self, filename: str) -> str:
        """Guess the content type based on filename"""
        content_type, _ = mimetypes.guess_type(filename)
        if content_type:
            return content_type

        # Some common mappings
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".mp3":
            return "audio/mpeg"
        elif extension == ".txt":
            return "text/plain"
        elif extension == ".json":
            return "application/json"

        # Default to octet-stream
        return "application/octet-stream"

    async def close(self):
        """Close any open connections or resources"""
        _logger.debug("Closing storage manager resources")
        if self.provider == "s3":
            # Close any S3 connections if needed
            pass
        elif self.provider == "azure":
            # Close any Azure connections if needed
            pass
        # For local storage, nothing special needs to be done
        return True

# Initialize the mimetypes module
mimetypes.init()

# Single shared instance
storage_manager = StorageManager()