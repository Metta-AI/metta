"""Utilities for temporarily silencing selected warning categories."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Type, TypeVar

WarningType = Type[Warning]
_T = TypeVar("_T")


@contextmanager
def silence_warnings(*, categories: Iterable[WarningType] | None = None) -> Iterator[None]:
    """Context manager that suppresses the provided warning categories.

    Parameters
    ----------
    categories:
        Iterable of warning classes to ignore for the duration of the context.
        Defaults to the base :class:`Warning`, silencing all warnings.
    """

    selected = tuple(categories) if categories else (Warning,)

    with warnings.catch_warnings():
        for category in selected:
            warnings.simplefilter("ignore", category=category)
        yield


def silence_warnings_function(
    func: Callable[..., _T],
    *,
    categories: Iterable[WarningType] | None = None,
) -> Callable[..., _T]:
    """Decorator helper that silences warnings while ``func`` executes."""

    def wrapper(*args, **kwargs):
        with silence_warnings(categories=categories):
            return func(*args, **kwargs)

    return wrapper


__all__ = ["silence_warnings", "silence_warnings_function", "WarningType"]
