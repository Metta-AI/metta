"""Route timing and logging utilities for performance monitoring."""

import logging
import time
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException, Request

# Logger for route performance
route_logger = logging.getLogger("route_performance")
route_logger.setLevel(logging.WARNING)

# Threshold for slow route warnings (2 seconds)
SLOW_ROUTE_THRESHOLD_SECONDS = 2.0


def timed_http_handler(func):
    """
    Exception handling wrapper for timed_route.
    """
    timed_func = timed_route(func.__name__)(func)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await timed_func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            operation = func.__name__.replace("_", " ")
            route_logger.debug(f"Failed to {operation}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to {operation}: {str(e)}") from e

    return wrapper


def timed_route(route_name: str = ""):
    """
    Decorator that logs the execution time of FastAPI routes.

    Logs all route executions with their timing. If a route takes longer than
    SLOW_ROUTE_THRESHOLD_SECONDS, logs a warning.

    Args:
        route_name: Optional custom name for the route (defaults to function name)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            name = route_name or func.__name__

            # Extract request info if available
            request_info = ""
            for arg in args:
                if isinstance(arg, Request):
                    request_info = f" {arg.method} {arg.url.path}"
                    break

            route_logger.debug(f"ROUTE START: {name}{request_info}")

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                route_logger.debug(f"ROUTE COMPLETE: {name} in {execution_time:.3f}s")

                # Log slow routes
                if execution_time > SLOW_ROUTE_THRESHOLD_SECONDS:
                    route_logger.warning(f"SLOW ROUTE ({execution_time:.3f}s): {name}{request_info}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                route_logger.debug(f"ROUTE FAILED: {name} after {execution_time:.3f}s - {e}")
                raise

        return wrapper

    return decorator
