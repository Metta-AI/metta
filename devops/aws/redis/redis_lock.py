"""
Redis-based distributed lock implementation to replace EFS locks.
"""

import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)


def load_env_file(env_file: str = "redis.env") -> None:
    """Load environment variables from a .env file."""
    env_path = Path(env_file)

    # Try current directory first
    if not env_path.exists():
        # Try parent directories (up to 3 levels)
        for parent in [Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent]:
            potential_path = parent / env_file
            if potential_path.exists():
                env_path = potential_path
                break

    if env_path.exists():
        logger.debug(f"Loading Redis config from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value.strip("\"'")
    else:
        logger.debug(f"No {env_file} found, using environment variables")


class RedisLock:
    """
    A distributed lock implementation using Redis.

    This replaces the file-based EFS lock with a more reliable Redis-based solution.
    Compatible with the same interface as the previous efs_lock.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key: str,
        timeout: int = 300,
        retry_interval: float = 1.0,
        max_retries: int = 60,
    ):
        """
        Initialize a Redis-based distributed lock.

        Args:
            redis_client: Redis client instance
            key: Lock key name
            timeout: Time in seconds after which lock expires
            retry_interval: Time in seconds between retries
            max_retries: Maximum number of retries
        """
        self.redis = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.identifier = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
        self.lock_acquired = False

    def acquire(self) -> bool:
        """Try to acquire the lock."""
        retries = 0

        while retries < self.max_retries:
            try:
                # Try to set the lock with NX (only if not exists) and EX (expiry)
                if self.redis.set(self.key, self.identifier, nx=True, ex=self.timeout):
                    self.lock_acquired = True
                    logger.info(f"Lock acquired: {self.key} by {self.identifier}")
                    return True

                # Check if current lock is expired (shouldn't happen with EX, but just in case)
                current_holder = self.redis.get(self.key)
                if current_holder:
                    logger.debug(f"Lock {self.key} held by: {current_holder.decode('utf-8')}")

            except (RedisError, ConnectionError) as e:
                logger.warning(f"Redis error while acquiring lock: {e}")

            time.sleep(self.retry_interval)
            retries += 1

        logger.error(f"Failed to acquire lock {self.key} after {retries} retries")
        return False

    def release(self) -> bool:
        """Release the lock if we hold it."""
        if not self.lock_acquired:
            return True

        try:
            # Use Lua script to ensure atomic check-and-delete
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """

            result = self.redis.eval(lua_script, 1, self.key, self.identifier)

            if result:
                self.lock_acquired = False
                logger.info(f"Lock released: {self.key}")
                return True
            else:
                logger.warning(f"Lock {self.key} was not held by us when trying to release")
                return False

        except (RedisError, ConnectionError) as e:
            logger.error(f"Failed to release lock {self.key}: {e}")
            return False

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock: {self.key}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def get_redis_client(env_file: str = "redis.env") -> redis.Redis:
    """
    Get Redis client from environment variables or .env file.

    Args:
        env_file: Path to .env file with Redis credentials

    Returns:
        Redis client instance
    """
    # Load from .env file if present
    load_env_file(env_file)

    host = os.environ.get("REDIS_HOST")
    port = int(os.environ.get("REDIS_PORT", 6379))
    password = os.environ.get("REDIS_PASSWORD")

    if not host:
        raise ValueError("REDIS_HOST not found in environment or redis.env file")
    if not password:
        raise ValueError("REDIS_PASSWORD not found in environment or redis.env file")

    logger.info(f"Connecting to Redis at {host}:{port}")

    return redis.Redis(
        host=host,
        port=port,
        password=password,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        retry_on_timeout=True,
        health_check_interval=30,
    )


@contextmanager
def redis_lock(
    path: str,
    timeout: int = 300,
    retry_interval: float = 1.0,
    max_retries: int = 60,
    redis_client: Optional[redis.Redis] = None,
    env_file: str = "redis.env",
):
    """
    Context manager for distributed locking using Redis.

    Drop-in replacement for efs_lock.

    Example:
        with redis_lock("/path/to/resource", timeout=300):
            # Critical section - only one process runs this at a time
            do_something()

    Args:
        path: Resource path/name to lock (will be prefixed with "lock:")
        timeout: Lock expiry time in seconds
        retry_interval: Time between lock attempts
        max_retries: Maximum number of attempts
        redis_client: Optional Redis client (will create one if not provided)
        env_file: Path to .env file with Redis credentials
    """
    if redis_client is None:
        redis_client = get_redis_client(env_file)

    lock = RedisLock(
        redis_client=redis_client, key=path, timeout=timeout, retry_interval=retry_interval, max_retries=max_retries
    )

    with lock:
        yield


# For backward compatibility
def efs_lock(*args, **kwargs):
    """Backward compatibility wrapper - redirects to redis_lock."""
    logger.warning("efs_lock is deprecated, use redis_lock instead")
    return redis_lock(*args, **kwargs)
