"""Retry logic for Asana API calls with exponential backoff."""

import asyncio
import logging
import random
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class AsanaAPIError(Exception):
    """Base exception for Asana API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception):
        super().__init__(message)
        self.last_exception = last_exception


def is_retryable_error(error: Exception) -> tuple[bool, Optional[int]]:
    """
    Determine if an error is retryable and return retry delay.

    Returns:
        Tuple of (is_retryable, retry_after_seconds)
    """
    error_str = str(error).lower()

    if "429" in error_str or "rate limit" in error_str:
        retry_after = _extract_retry_after(error)
        return True, retry_after

    if any(code in error_str for code in ["500", "502", "503", "504", "timeout", "connection"]):
        return True, None

    if any(code in error_str for code in ["400", "401", "403", "404"]):
        return False, None

    return True, None


def _extract_retry_after(error: Exception) -> Optional[int]:
    """Extract Retry-After value from error if available."""
    error_str = str(error)
    if "retry-after" in error_str.lower():
        try:
            import re

            match = re.search(r"retry-after[:\s]+(\d+)", error_str, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
    return None


async def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 4,
    initial_delay_ms: float = 500.0,
    max_delay_ms: float = 10000.0,
    exp_base: float = 2.0,
    operation_name: str = "operation",
) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Supports both sync and async functions.

    Args:
        func: Function to execute (sync or async)
        max_retries: Maximum number of retry attempts (default: 4)
        initial_delay_ms: Initial delay in milliseconds (default: 500ms)
        max_delay_ms: Maximum delay in milliseconds (default: 10000ms)
        exp_base: Exponential base for backoff (default: 2.0)
        operation_name: Name of operation for logging

    Returns:
        Result of func()

    Raises:
        RetryExhausted: If all retries are exhausted
    """
    last_exception = None
    is_async = asyncio.iscoroutinefunction(func)

    for attempt in range(max_retries):
        try:
            if is_async:
                return await func()
            else:
                return func()
        except Exception as e:
            last_exception = e
            is_retryable, retry_after = is_retryable_error(e)

            if not is_retryable:
                logger.warning(f"{operation_name} failed with non-retryable error: {e}")
                raise

            if attempt < max_retries - 1:
                if retry_after:
                    delay_seconds = retry_after
                else:
                    delay_ms = min(initial_delay_ms * (exp_base**attempt), max_delay_ms)
                    jitter_ms = random.uniform(0, delay_ms * 0.1)
                    delay_seconds = (delay_ms + jitter_ms) / 1000.0

                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay_seconds:.2f}s"
                )
                await asyncio.sleep(delay_seconds)
            else:
                logger.error(f"{operation_name} failed after {max_retries} attempts: {e}")
                raise RetryExhausted(f"{operation_name} failed after {max_retries} attempts", e) from e

    raise RetryExhausted(f"{operation_name} exhausted retries", last_exception or Exception("Unknown error"))
