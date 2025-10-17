"""Helper for safely stubbing Triton when it is not installed.

This allows modules that annotate kernels with ``@triton.jit`` to be imported
even on systems where Triton is unavailable.  The stub mirrors the attributes
that our code touches and raises a helpful error if any Triton-only function is
invoked at runtime.
"""

from __future__ import annotations

import sys
import types
from functools import wraps
from typing import Any, Callable

_RUNTIME_ERROR = RuntimeError(
    "Triton is not installed. Install the Triton package or set "
    "CORTEX_DISABLE_TRITON=1 to silence Triton-dependent paths."
)


def _jit_stub(*jit_args: Any, **jit_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that raises if the Triton kernel is executed."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def _missing(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
            raise _RUNTIME_ERROR

        return _missing

    return decorator


def _autotune_stub(*autotune_args: Any, **autotune_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Autotune decorator is a no-op when Triton is missing."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return decorator


def _cdiv_stub(x: int, y: int) -> int:
    """Ceiling integer division used to compute Triton grid sizes."""
    if y == 0:
        raise ZeroDivisionError("division by zero")  # pragma: no cover - guard
    return -(-x // y)


class _ConfigStub:
    """Lightweight stand-in for ``triton.Config`` used in kernel specs."""

    def __init__(self, kwargs: dict[str, Any] | None = None, **meta: Any) -> None:
        self.kwargs = kwargs or {}
        self.meta = meta


class _TLStub(types.ModuleType):
    """Stub of ``triton.language`` providing lazy attributes that raise on use."""

    constexpr = int  # used for annotations; any sentinel works

    def __getattr__(self, name: str) -> Callable[..., Any]:  # pragma: no cover - defensive
        def _missing(*args: Any, **kwargs: Any) -> Any:
            raise _RUNTIME_ERROR

        return _missing


class _OutOfResources(RuntimeError):
    """Stub exception mirroring ``triton.OutOfResources``."""

    pass


def install_triton_stub() -> None:
    """Install stub modules for ``triton`` and ``triton.language``."""
    triton_module = types.ModuleType("triton")
    triton_module.jit = _jit_stub
    triton_module.autotune = _autotune_stub
    triton_module.cdiv = _cdiv_stub
    triton_module.Config = _ConfigStub
    triton_module.OutOfResources = _OutOfResources

    tl_module = _TLStub("triton.language")

    sys.modules.setdefault("triton", triton_module)
    sys.modules.setdefault("triton.language", tl_module)


__all__ = ["install_triton_stub"]
