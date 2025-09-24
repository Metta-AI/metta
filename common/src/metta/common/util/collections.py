from collections import Counter, defaultdict
from typing import Callable, Hashable, Iterable, List, Sequence, TypeVar

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


def is_unique(collection: Sequence[T]) -> bool:
    return len(collection) == len(set(collection))


def remove_falsey(collection: Iterable[T | None]) -> list[T]:
    return [item for item in collection if item]


def duplicates(iterable: Iterable[Hashable]) -> List[Hashable]:
    """Returns a list of items that appear more than once in the iterable."""
    counts = Counter(iterable)
    return [item for item, count in counts.items() if count > 1]
