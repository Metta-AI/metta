import collections
import typing

T = typing.TypeVar("T")
K = typing.TypeVar("K")


def group_by(collection: list[T], key_fn: typing.Callable[[T], K]) -> collections.defaultdict[K, list[T]]:
    grouped = collections.defaultdict(list)
    for item in collection:
        grouped[key_fn(item)].append(item)
    return grouped


def remove_none_values(d: dict[K, T | None]) -> dict[K, T]:
    return {k: v for k, v in d.items() if v is not None}


def remove_none_keys(d: dict[K | None, T]) -> dict[K, T]:
    return {k: v for k, v in d.items() if k is not None}


def find_first(collection: typing.Iterable[T], predicate: typing.Callable[[T], bool]) -> T | None:
    return next((item for item in collection if predicate(item)), None)


def is_unique(collection: typing.Sequence[T]) -> bool:
    return len(collection) == len(set(collection))


def remove_falsey(collection: typing.Iterable[T | None]) -> list[T]:
    return [item for item in collection if item]


def duplicates(iterable: typing.Iterable[typing.Hashable]) -> typing.List[typing.Hashable]:
    """Returns a list of items that appear more than once in the iterable."""
    counts = collections.Counter(iterable)
    return [item for item, count in counts.items() if count > 1]
