from collections import defaultdict
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
K = TypeVar("K")


def group_by(collection: list[T], key_fn: Callable[[T], K]) -> defaultdict[K, list[T]]:
    grouped = defaultdict(list)
    for item in collection:
        grouped[key_fn(item)].append(item)
    return grouped


def remove_none_values(d: dict[K, T | None]) -> dict[K, T]:
    return {k: v for k, v in d.items() if v is not None}


def remove_none_keys(d: dict[K | None, T]) -> dict[K, T]:
    return {k: v for k, v in d.items() if k is not None}


def find_first(collection: Iterable[T], predicate: Callable[[T], bool]) -> T | None:
    return next((item for item in collection if predicate(item)), None)
