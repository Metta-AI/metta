"""Token registry and decorator-based registration.

This module provides a simple registry for mapping token symbols to
block configuration builders. Registration is by exact token string,
so both base tokens (e.g., "X") and caret variants (e.g., "X^") can
be registered with the same decorator.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from cortex.config import BlockConfig

_BUILDERS: Dict[str, Callable[[], BlockConfig]] = {}


def register_token(symbol: str) -> Callable[[Callable[[], BlockConfig]], Callable[[], BlockConfig]]:
    """Decorator to register a builder for an exact token symbol.

    Example:
        @register_token("M")
        def build_m() -> BlockConfig: ...

        @register_token("M^")
        def build_m_axon() -> BlockConfig: ...
    """

    def _decorator(fn: Callable[[], BlockConfig]) -> Callable[[], BlockConfig]:
        _BUILDERS[symbol] = fn
        return fn

    return _decorator


def block_config_for_token(token: str) -> Optional[BlockConfig]:
    """Return a BlockConfig for a registered token or None if unknown.

    The token may optionally end with "^" to request an axon-enabled variant
    if one is registered for its base symbol.
    """

    # Exact match first (supports registering "X^" directly)
    builder = _BUILDERS.get(token)
    if builder is not None:
        return builder()
    # Fallback: if caret requested but only base is registered, use base
    if token.endswith("^"):
        base = token[:-1]
        builder = _BUILDERS.get(base)
        if builder is not None:
            return builder()
    return None


def get_single_char_symbols() -> Iterable[str]:
    """Return base symbols for registered single-character tokens.

    The returned set includes a base symbol if either the base or its
    caret variant is registered.
    """

    bases: List[str] = []
    seen = set()
    for k in _BUILDERS.keys():
        base = k.rstrip("^")
        if len(base) == 1 and base not in seen:
            bases.append(base)
            seen.add(base)
    return tuple(bases)


def can_use_caret(base: str) -> bool:
    """Return True if a caret variant (e.g., "X^") is registered for `base`."""

    return f"{base}^" in _BUILDERS


__all__ = [
    "register_token",
    "block_config_for_token",
    "get_single_char_symbols",
    "can_use_caret",
]


def get_registered_tokens() -> Tuple[str, ...]:
    """Return all registered token symbols (exact strings)."""
    return tuple(_BUILDERS.keys())
