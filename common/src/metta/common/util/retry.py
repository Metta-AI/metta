import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def calculate_backoff_delay(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(initial_delay * (backoff_factor**attempt), max_delay)
    return random.uniform(0, delay) if jitter else delay


def retry_function(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute a function with retry logic using exponential backoff."""
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e

            # If not the last attempt, sleep before retrying
            if attempt < max_retries:
                delay = calculate_backoff_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    backoff_factor=backoff_factor,
                )
                time.sleep(delay)

    # If we get here, all retries failed
    assert last_exception is not None  # Should always have an exception here
    raise last_exception


def retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on exception with exponential backoff."""

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
