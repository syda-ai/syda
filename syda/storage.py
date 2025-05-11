import boto3
import os
from typing import Dict, Optional, Union
from azure.storage.blob import BlobServiceClient
import pandas as pd


class StorageSystem:
    def __init__(self, storage_type: str, **kwargs):
        """
        Initialize storage system
        
        Args:
            storage_type: Type of storage ('s3', 'local', 'azure', 'gcs')
            kwargs: Storage-specific configuration
        """
        self.storage_type = storage_type.lower()
        self.config = kwargs
        
        if self.storage_type == 's3':
            self.client = boto3.client(
                's3',
                aws_access_key_id=kwargs.get('aws_access_key_id'),
                aws_secret_access_key=kwargs.get('aws_secret_access_key'),
                region_name=kwargs.get('region_name', 'us-east-1')
            )
        elif self.storage_type == 'azure':
            self.client = BlobServiceClient.from_connection_string(
                kwargs.get('connection_string')
            )
        
    def read_file(self, path: str) -> bytes:
        """
        Read file from storage
        
        Args:
            path: Path to file
            
        Returns:
            File content as bytes
        """
        if self.storage_type == 's3':
            bucket, key = path.split('/', 1)
            response = self.client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        elif self.storage_type == 'azure':
            container, blob = path.split('/', 1)
            blob_client = self.client.get_blob_client(container, blob)
            return blob_client.download_blob().readall()
        elif self.storage_type == 'local':
            with open(path, 'rb') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def write_file(self, path: str, content: Union[bytes, str], format: str = 'csv'):
        """
        Write file to storage
        
        Args:
            path: Path to write file
            content: File content
            format: File format ('csv', 'json', 'excel')
        """
        if isinstance(content, str):
            content = content.encode()
            
        if self.storage_type == 's3':
            bucket, key = path.split('/', 1)
            self.client.put_object(Bucket=bucket, Key=key, Body=content)
        elif self.storage_type == 'azure':
            container, blob = path.split('/', 1)
            blob_client = self.client.get_blob_client(container, blob)
            blob_client.upload_blob(content, overwrite=True)
        elif self.storage_type == 'local':
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def list_files(self, path: str) -> list:
        """
        List files in directory
        
        Args:
            path: Path to list files from
            
        Returns:
            List of file paths
        """
        if self.storage_type == 's3':
            bucket, prefix = path.split('/', 1)
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [f"{bucket}/{obj['Key']}" for obj in response.get('Contents', [])]
        elif self.storage_type == 'azure':
            container, prefix = path.split('/', 1)
            container_client = self.client.get_container_client(container)
            return [f"{container}/{blob.name}" for blob in container_client.list_blobs(name_starts_with=prefix)]
        elif self.storage_type == 'local':
            return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

class StorageFactory:
    @staticmethod
    def get_storage(storage_type: str, **kwargs) -> StorageSystem:
        """
        Get appropriate storage system
        
        Args:
            storage_type: Type of storage ('s3', 'local', 'azure', 'gcs')
            kwargs: Storage-specific configuration
            
        Returns:
            StorageSystem instance
        """
        return StorageSystem(storage_type, **kwargs)
