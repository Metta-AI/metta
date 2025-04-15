import logging
import os
import platform
import random
import socket
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EfsLock:
    """
    A distributed lock implementation for AWS EFS and other shared filesystems.

    Uses file-based locking with timestamp-based stale lock detection.

    Example usage:
    ```
    with EfsLock("/path/to/lock", timeout=300):
        # This code will only run in one process at a time
        do_something()
    ```
    """

    def __init__(self, path, timeout=300, retry_interval=5, max_retries=60):
        """
        Initialize a distributed lock.

        Args:
            path: Path to the lock file
            timeout: Time in seconds after which a lock is considered stale
            retry_interval: Time in seconds to wait between retries
            max_retries: Maximum number of times to retry acquiring the lock
        """
        self.path = path
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.lock_acquired = False
        self.hostname = socket.gethostname()
        self.system = platform.system()
        logger.debug(f"Initializing EfsLock on {self.system} ({self.hostname}) for path: {path}")

    def __enter__(self):
        """Acquire the lock."""
        self._acquire_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock."""
        self._release_lock()

    def _acquire_lock(self):
        """
        Try to acquire the lock.

        If the lock is already held, wait and retry.
        If the lock is stale, remove it and try again.
        """
        retries = 0

        # Add a small random delay to reduce contention
        delay = random.uniform(0.1, 1.0)
        logger.debug(f"Initial delay before lock attempt: {delay:.2f}s")
        time.sleep(delay)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        logger.debug(f"Starting lock acquisition attempts for {self.path}")

        while retries < self.max_retries:
            try:
                # Try to create the lock file exclusively
                logger.debug(f"Attempt {retries + 1}/{self.max_retries} to acquire lock: {self.path}")
                with open(self.path, "x") as f:
                    # Write current timestamp to the lock file
                    timestamp = time.time()
                    pid = os.getpid()
                    content = f"{timestamp}\n{pid}\n{self.hostname}\n{self.system}\n"
                    f.write(content)
                    logger.debug(f"Lock file content: {content}")

                self.lock_acquired = True
                logger.info(f"Lock acquired: {self.path}")
                return
            except FileExistsError:
                # Lock file exists, check if it's stale
                try:
                    logger.debug(f"Lock file exists, checking if stale: {self.path}")
                    with open(self.path, "r") as f:
                        lines = f.readlines()
                        if len(lines) >= 1:
                            timestamp = float(lines[0].strip())
                            current_time = time.time()
                            age = current_time - timestamp
                            logger.debug(f"Lock age: {age:.2f}s, timeout: {self.timeout}s")

                            # Log more details about the lock
                            if len(lines) >= 3:
                                lock_hostname = lines[2].strip()
                                logger.debug(f"Lock held by: {lock_hostname}")

                            if age > self.timeout:
                                # Lock is stale, try to remove it
                                logger.warning(f"Removing stale lock (age: {age:.2f}s): {self.path}")
                                try:
                                    os.remove(self.path)
                                    logger.debug(f"Successfully removed stale lock: {self.path}")
                                    continue
                                except OSError as e:
                                    logger.warning(f"Failed to remove stale lock: {e}")
                except (ValueError, IOError, OSError) as e:
                    # Invalid lock file or can't read it, try to remove
                    logger.warning(f"Error reading lock file: {e}")
                    try:
                        os.remove(self.path)
                        logger.debug(f"Removed invalid lock file: {self.path}")
                        continue
                    except OSError as e:
                        logger.warning(f"Failed to remove invalid lock file: {e}")

                # Wait and retry
                logger.debug(f"Waiting {self.retry_interval}s for lock: {self.path}")
                time.sleep(self.retry_interval)
                retries += 1

        logger.error(f"Failed to acquire lock after {retries} retries: {self.path}")
        raise TimeoutError(f"Failed to acquire lock after {retries} retries: {self.path}")

    def _release_lock(self):
        """Release the lock if we hold it."""
        if self.lock_acquired:
            try:
                logger.debug(f"Releasing lock: {self.path}")
                os.remove(self.path)
                self.lock_acquired = False
                logger.info(f"Lock released: {self.path}")
            except OSError as e:
                logger.warning(f"Failed to release lock {self.path}: {e}")


@contextmanager
def efs_lock(path, timeout=300, retry_interval=5, max_retries=60):
    """
    A context manager for distributed locking on EFS.

    This is a convenience wrapper around the EfsLock class.

    Example usage:
    ```
    with efs_lock("/path/to/lock", timeout=300):
        # This code will only run in one process at a time
        do_something()
    ```

    Args:
        path: Path to the lock file
        timeout: Time in seconds after which a lock is considered stale
        retry_interval: Time in seconds to wait between retries
        max_retries: Maximum number of times to retry acquiring the lock
    """
    # For local development on Mac, use a shorter timeout and retry interval
    if platform.system() == "Darwin":
        logger.debug("Running on macOS, adjusting lock parameters")
        if timeout > 60:
            timeout = 60
        if retry_interval > 2:
            retry_interval = 2

    logger.debug(f"Creating lock with timeout={timeout}s, retry_interval={retry_interval}s, max_retries={max_retries}")
    lock = EfsLock(path, timeout, retry_interval, max_retries)
    try:
        lock._acquire_lock()
        yield
    finally:
        lock._release_lock()
