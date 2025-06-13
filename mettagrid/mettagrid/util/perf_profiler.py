import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")


class PerfProfiler:
    def __init__(self) -> None:
        self.stats: Dict[str, float] = defaultdict(float)

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stats[name] += time.perf_counter() - start

    def reset(self) -> None:
        self.stats.clear()


def profiled(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to profile a method using ``PerfProfiler`` on ``self``."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            profiler: PerfProfiler = self._profiler
            with profiler.time(name):
                result = func(self, *args, **kwargs)
            return result

        return wrapper

    return decorator
