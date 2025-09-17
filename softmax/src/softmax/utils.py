import time
from functools import wraps


def memoize(max_age: int = 60):
    def decorator(func):
        cache = {}
        cache_time = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hash((args, tuple(sorted(kwargs.items()))))
            current_time = time.time()

            if key in cache and current_time - cache_time[key] < max_age:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = current_time
            return result

        return wrapper

    return decorator
