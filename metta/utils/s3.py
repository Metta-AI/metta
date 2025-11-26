"""S3 utility functions for checkpoint and replay management.

Provides reusable S3 operations with proper error handling, pagination,
and metadata extraction. Extends metta.utils.file patterns.
"""

import logging
from typing import Any, Optional

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Store:
    """Utility class for S3 operations (checkpoints, replays, etc.)."""

    def __init__(self, s3_client: Any, bucket: str):
        """Initialize S3 store.

        Args:
            s3_client: Boto3 S3 client instance
            bucket: Default S3 bucket name
        """
        self.client = s3_client
        self.bucket = bucket

    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
        page_size: int = 1000,
        filter_extensions: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """List S3 objects with pagination and filtering.

        Extends boto3 pagination with early termination and filtering.

        Args:
            prefix: S3 prefix to list
            max_keys: Maximum total objects to return
            page_size: Objects per page (for pagination efficiency)
            filter_extensions: Optional list of file extensions to filter (e.g., ['.mpt', '.json'])

        Returns:
            List of object metadata dictionaries
        """
        objects = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, MaxKeys=page_size):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                if len(objects) >= max_keys:
                    break

                key = obj["Key"]

                # Apply extension filter if provided
                if filter_extensions:
                    if not any(key.endswith(ext) for ext in filter_extensions):
                        continue

                object_info = {
                    "key": key,
                    "uri": f"s3://{self.bucket}/{key}",
                    "filename": key.split("/")[-1],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "etag": obj["ETag"].strip('"'),
                }

                objects.append(object_info)

            if len(objects) >= max_keys:
                break

        return objects

    def object_exists(self, key: str) -> bool:
        """Check if an S3 object exists.

        Args:
            key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "403", "NoSuchKey"}:
                return False
            raise

    def get_object_metadata(self, key: str) -> Optional[dict[str, Any]]:
        """Get metadata for an S3 object.

        Args:
            key: S3 object key

        Returns:
            Metadata dictionary if object exists, None otherwise
        """
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return {
                "exists": True,
                "key": key,
                "uri": f"s3://{self.bucket}/{key}",
                "size": response["ContentLength"],
                "last_modified": response["LastModified"].isoformat(),
                "etag": response["ETag"].strip('"'),
                "content_type": response.get("ContentType", "application/octet-stream"),
            }
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return {
                    "exists": False,
                    "key": key,
                    "uri": f"s3://{self.bucket}/{key}",
                }
            raise

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for an S3 object.

        Args:
            key: S3 object key
            expires_in: URL expiration time in seconds (default: 3600)

        Returns:
            Presigned URL string
        """
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            logger.warning(f"Failed to generate presigned URL for {key}: {e}")
            raise

    def parse_checkpoint_filename(self, filename: str) -> Optional[dict[str, Any]]:
        """Parse checkpoint filename to extract metadata.

        Handles formats like: 'checkpoint:v123.mpt' or 'checkpoint_v123.mpt'

        Args:
            filename: Checkpoint filename

        Returns:
            Dictionary with parsed metadata (epoch, etc.) or None
        """
        metadata = {}
        if ":v" in filename:
            try:
                epoch_str = filename.split(":v")[1].replace(".mpt", "")
                metadata["epoch"] = int(epoch_str)
            except ValueError:
                pass
        elif "_v" in filename:
            try:
                epoch_str = filename.split("_v")[1].replace(".mpt", "")
                metadata["epoch"] = int(epoch_str)
            except ValueError:
                pass

        return metadata if metadata else None

    def list_checkpoints(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """List checkpoint files in S3 with metadata extraction.

        Args:
            prefix: S3 prefix to list (e.g., "checkpoints/run_name/")
            max_keys: Maximum number of checkpoints to return

        Returns:
            List of checkpoint dictionaries with metadata (epoch, etc.)
        """
        objects = self.list_objects(
            prefix=prefix,
            max_keys=max_keys,
            filter_extensions=[".mpt"],
        )

        checkpoints = []
        for obj in objects:
            checkpoint_info = obj.copy()
            filename = obj["filename"]

            # Parse checkpoint metadata
            metadata = self.parse_checkpoint_filename(filename)
            if metadata:
                checkpoint_info.update(metadata)

            checkpoints.append(checkpoint_info)

        return checkpoints

    def list_replays(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """List replay files in S3.

        Args:
            prefix: S3 prefix to list (e.g., "replays/run_name/")
            max_keys: Maximum number of replays to return

        Returns:
            List of replay file dictionaries
        """
        return self.list_objects(
            prefix=prefix,
            max_keys=max_keys,
            filter_extensions=[".replay", ".json", ".gz"],
        )
