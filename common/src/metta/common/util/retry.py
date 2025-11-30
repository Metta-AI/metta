from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


def retry_function(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute a function with retry logic using exponential backoff.

    Args:
        func: A callable with no arguments to execute.
        max_retries: Maximum number of retry attempts after the initial call.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier for exponential backoff (used as exp_base).
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        The return value of the function.

    Raises:
        The last exception raised if all retries fail.
    """
    retryer = retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential_jitter(initial=initial_delay, max=max_delay, exp_base=backoff_factor),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
    try:
        return retryer(func)()
    except RetryError:
        # This shouldn't happen with reraise=True, but handle it just in case
        raise


def retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on exception with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts after the initial call.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier for exponential backoff (used as exp_base).
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        A decorator that wraps functions with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_function(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
            )

        return wrapper

    return decorator
