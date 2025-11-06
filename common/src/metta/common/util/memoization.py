import collections.abc
import functools
import time
import typing

P = typing.ParamSpec("P")

T = typing.TypeVar("T")


def memoize(
    max_age: float,
) -> collections.abc.Callable[
    [collections.abc.Callable[P, collections.abc.Awaitable[T]]],
    collections.abc.Callable[P, collections.abc.Awaitable[T]],
]:
    def decorator(
        func: collections.abc.Callable[P, collections.abc.Awaitable[T]],
    ) -> collections.abc.Callable[P, collections.abc.Awaitable[T]]:
        cache: dict[tuple[typing.Any, ...], tuple[T, float]] = {}

        @functools.wraps(func)
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
