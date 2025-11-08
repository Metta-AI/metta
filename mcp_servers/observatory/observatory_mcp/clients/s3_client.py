"""S3 client wrapper."""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from metta.utils.uri import ParsedURI

logger = logging.getLogger(__name__)


class S3Client:
    """Client wrapper for S3 operations."""

    def __init__(self, profile: Optional[str] = None, bucket: str = "softmax-public"):
        """Initialize the S3 client.

        Args:
            profile: AWS profile name (optional)
            bucket: Default S3 bucket name
        """
        self.profile = profile
        self.bucket = bucket
        self._client = None

        if profile:
            try:
                self._client = boto3.client("s3", profile_name=profile)
                logger.info(f"S3 client initialized (profile={profile}, bucket={bucket})")
            except (NoCredentialsError, ClientError) as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
                self._client = None
        else:
            try:
                self._client = boto3.client("s3")
                logger.info(f"S3 client initialized (bucket={bucket})")
            except (NoCredentialsError, ClientError) as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
                self._client = None

    @property
    def client(self):
        """Get boto3 S3 client."""
        if self._client is None:
            raise RuntimeError("S3 client not initialized. Check AWS credentials.")
        return self._client

    def parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key.

        Args:
            uri: S3 URI (s3://bucket/key or just key)

        Returns:
            Tuple of (bucket, key)
        """
        parsed = ParsedURI.parse(uri)
        if parsed.scheme == "s3":
            return parsed.require_s3()
        return (self.bucket, uri)

