import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def calculate_backoff_delay(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
) -> float:
    """Calculate delay in seconds with exponential backoff and optional jitter."""
    delay = min(initial_delay * (backoff_factor**attempt), max_delay)
    delay = random.uniform(0, delay)
    return delay


def retry_function(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    error_prefix: str = "Function failed",
    logger: Optional[logging.Logger] = None,
) -> T:
    """Execute a function with retry logic using exponential backoff."""
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e

            # Log the failure
            if logger:
                if attempt == 0:
                    logger.warning(f"{error_prefix}: {e}")
                else:
                    logger.warning(f"{error_prefix} (retry {attempt}/{max_retries}): {e}")

            # If not the last attempt, sleep before retrying
            if attempt < max_retries:
                delay = calculate_backoff_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    backoff_factor=backoff_factor,
                )

                if logger:
                    logger.info(f"Retrying in {delay:.2f} seconds...")

                time.sleep(delay)
            else:
                if logger:
                    logger.error(f"{error_prefix} after {max_retries} retries")

    # If we get here, all retries failed
    if last_exception is not None:
        raise last_exception
    else:
        # This should never happen, but just in case
        raise RuntimeError(f"{error_prefix}: All retries failed without exception")


def retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on exception with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use retry_function internally to avoid code duplication
            return retry_function(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
                error_prefix=f"{func.__name__} failed",
                logger=logger,
            )

        return wrapper

    return decorator
