"""
Retry utilities for environment initialization.
"""

import time
import functools
import logging
from typing import TypeVar, Callable, Any, Optional, Type, Tuple, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


def exponential_backoff_retry(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 0.1)
        max_delay: Maximum delay between retries in seconds (default: 2.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exceptions to catch and retry (default: all exceptions)
    
    Returns:
        Decorated function that implements retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )
            
            # If we get here, all retries have failed
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")
        
        return wrapper
    return decorator


# Convenience decorator with default settings for environment initialization
env_init_retry = exponential_backoff_retry(
    max_attempts=3,
    initial_delay=0.1,
    max_delay=2.0,
    exceptions=(RuntimeError, ConnectionError, TimeoutError)
)
