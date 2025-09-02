import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_function(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    error_prefix: str = "Function failed",
    logger: logging.Logger | None = None,
) -> T:
    """
    Execute a function with retry logic.

    This is a non-decorator version that can be used inline with lambda functions.

    Args:
        func: Function to execute (should return the result or raise an exception)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry on
        error_prefix: Prefix for error messages
        logger: Logger instance for logging retry attempts

    Returns:
        The result of the successful function call

    Raises:
        The last exception if all retries fail

    Example:
        result = retry_function(
            lambda: requests.get(url),
            max_retries=3,
            error_prefix="Failed to fetch data"
        )
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == 0:
                if logger:
                    logger.warning(f"{error_prefix}: {e}")
            else:
                if logger:
                    logger.warning(f"{error_prefix} (retry {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                if logger:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                if logger:
                    logger.error(f"{error_prefix} after {max_retries} retries")

    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError(f"{error_prefix}: All retries failed")


def retry_on_exception(
    max_retries: int = 3,
    retry_delay: float = 5.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on exception.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        exceptions: Tuple of exception types to catch and retry on
        logger: Logger instance for logging retry attempts

    Returns:
        Decorated function that implements retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use retry_function internally to avoid code duplication
            return retry_function(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                retry_delay=retry_delay,
                exceptions=exceptions,
                error_prefix=f"{func.__name__} failed",
                logger=logger,
            )

        return wrapper

    return decorator
