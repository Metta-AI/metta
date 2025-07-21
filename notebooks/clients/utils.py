import functools
import time


def memoize_with_expiry(ttl_seconds):
    def decorator(func):
        cache = {}  # Stores (result, timestamp)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = functools._make_key(args, kwargs, typed=False)  # Helper for creating cache key

            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
                else:
                    # Cache expired, remove and recompute
                    del cache[key]

            # Compute and cache new result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        return wrapper

    return decorator
