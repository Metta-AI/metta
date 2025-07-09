import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


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
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if logger:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        if logger:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        if logger:
                            logger.error(f"{func.__name__} failed after {max_retries} attempts")

            # If we get here, all retries failed
            if last_exception is not None:
                raise last_exception
            else:
                # This should never happen, but just in case
                raise RuntimeError(f"{func.__name__} failed without capturing exception")

        return wrapper

    return decorator
