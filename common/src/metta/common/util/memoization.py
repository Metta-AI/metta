import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")

T = TypeVar("T")


def memoize(max_age: int) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        cache: dict[tuple[Any, ...], tuple[T, float]] = {}

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()

            if key in cache:
                value, timestamp = cache[key]
                if current_time - timestamp < max_age:
                    return value

            result = await func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result

        return wrapper

    return decorator
