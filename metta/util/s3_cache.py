"""
General S3-backed cache with compression

A flexible caching system that stores compressed objects in S3 using
hash-based keys for efficient retrieval and storage.
"""

import gzip
import hashlib
import logging
import pickle
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3CacheManager:
    """Manages S3-backed caching with compression for expensive object creation."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "cache/",
        compression_level: int = 6,
        aws_region: Optional[str] = None,
    ):
        """
        Initialize the S3 cache manager.

        Args:
            bucket_name: S3 bucket name for cache storage
            prefix: S3 key prefix for cache objects (should end with '/')
            compression_level: Gzip compression level (0-9, higher = more compression)
            aws_region: AWS region for S3 client (uses default if None)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.compression_level = compression_level
        self.s3_client: Optional[BaseClient] = None

        try:
            self.s3_client = boto3.client("s3", region_name=aws_region)
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.s3_available = True
            logger.info(f"S3 cache initialized: s3://{bucket_name}/{self.prefix}")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"S3 unavailable, cache will be disabled: {e}")
            self.s3_available = False
            self.s3_client = None

    def create_key(self, *args, **kwargs) -> str:
        """
        Create a reproducible hash key from arbitrary arguments.

        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            SHA-256 hex digest of the serialized arguments
        """
        hasher = hashlib.sha256()

        for arg in args:
            arg_bytes = self._serialize_for_hash(arg)
            hasher.update(arg_bytes)

        # Hash keyword arguments (sorted for consistency)
        for key in sorted(kwargs.keys()):
            key_bytes = key.encode("utf-8")
            value_bytes = self._serialize_for_hash(kwargs[key])
            hasher.update(key_bytes)
            hasher.update(value_bytes)

        return hasher.hexdigest()

    def _serialize_for_hash(self, obj) -> bytes:
        """
        Serialize an object to bytes for hashing.

        Args:
            obj: Object to serialize

        Returns:
            Serialized bytes representation
        """
        try:
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except (TypeError, pickle.PickleError) as e:
            # Fallback to string representation for non-pickleable objects
            logger.warning(f"Could not pickle object {type(obj)}, using string repr: {e}")
            return str(obj).encode("utf-8")

    def _compress_object(self, obj: Any) -> bytes:
        """
        Serialize and compress an object.

        Args:
            obj: Object to compress

        Returns:
            Compressed bytes
        """
        serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=self.compression_level) as f:
            f.write(serialized)

        return buffer.getvalue()

    def _decompress_object(self, compressed_data: bytes) -> Any:
        """
        Decompress and deserialize an object.

        Args:
            compressed_data: Compressed bytes

        Returns:
            Deserialized object
        """
        with gzip.GzipFile(fileobj=BytesIO(compressed_data), mode="rb") as f:
            serialized = f.read()

        return pickle.loads(serialized)

    def _get_s3_key(self, cache_key: str) -> str:
        """Get the full S3 key for a cache key."""
        return f"{self.prefix}{cache_key}.pkl.gz"

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve an object from cache.

        Args:
            cache_key: The cache key to look up

        Returns:
            The cached object if found, None otherwise
        """
        if not self.s3_available or self.s3_client is None:
            return None

        try:
            s3_key = self._get_s3_key(cache_key)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            compressed_data = response["Body"].read()
            obj = self._decompress_object(compressed_data)
            logger.debug(f"S3 cache hit for key: {cache_key}")
            return obj
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                logger.warning(f"S3 error retrieving {cache_key}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error retrieving from S3 {cache_key}: {e}")

        logger.debug(f"Cache miss for key: {cache_key}")
        return None

    def put(self, cache_key: str, obj: Any) -> bool:
        """
        Store an object in cache.

        Args:
            cache_key: The cache key to store under
            obj: The object to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.s3_available or self.s3_client is None:
            return False

        try:
            compressed_data = self._compress_object(obj)
            s3_key = self._get_s3_key(cache_key)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=compressed_data,
                ContentType="application/octet-stream",
                ContentEncoding="gzip",
            )
            logger.debug(f"Cached object to S3 for key: {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache to S3 for key {cache_key}: {e}")
            return False

    def delete(self, cache_key: str) -> bool:
        """
        Delete an object from cache.

        Args:
            cache_key: The cache key to delete

        Returns:
            True if successfully deleted or object didn't exist, False on error
        """
        if not self.s3_available or self.s3_client is None:
            return False

        try:
            s3_key = self._get_s3_key(cache_key)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.debug(f"Deleted object from S3 for key: {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete from S3 for key {cache_key}: {e}")
            return False

    @contextmanager
    def __call__(self, *args, **kwargs):
        """
        Context manager for cached computation.

        Usage:
            with cache_manager(arg1, arg2, kwarg=value) as cached_result:
                if cached_result is not None:
                    result = cached_result
                else:
                    result = expensive_computation(arg1, arg2, kwarg=value)

        The result will be automatically cached when the context exits.
        """
        cache_key = self.create_key(*args, **kwargs)
        cached_result = self.get(cache_key)

        # Yield the cached result (None if cache miss)
        context = CacheContext(self, cache_key, cached_result)
        try:
            yield context
        finally:
            # Auto-save if a result was set and we had a cache miss
            if context.result is not None and cached_result is None:
                self.put(cache_key, context.result)


class CacheContext:
    """Context object for cached computation."""

    def __init__(self, cache_manager: S3CacheManager, cache_key: str, cached_result: Any):
        self.cache_manager = cache_manager
        self.cache_key = cache_key
        self.cached_result = cached_result
        self.result = None

    @property
    def hit(self) -> bool:
        """True if we had a cache hit."""
        return self.cached_result is not None

    @property
    def miss(self) -> bool:
        """True if we had a cache miss."""
        return self.cached_result is None

    def get(self) -> Any:
        """Get the cached result (None if cache miss)."""
        return self.cached_result

    def set(self, result: Any) -> None:
        """Set the result to be cached."""
        self.result = result
