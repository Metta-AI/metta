"""Utilities for configuring PyTorch scaled dot-product attention backends."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Optional

import torch


def build_sdpa_context(
    *,
    prefer_flash: bool = True,
    prefer_mem_efficient: bool = True,
    prefer_math: bool = True,
    set_priority: bool = True,
) -> Optional[contextlib.AbstractContextManager]:
    """Return a context manager that configures SDPA backends, if possible.

    The helper prefers the modern ``torch.nn.attention.sdpa_kernel`` API while
    gracefully falling back to legacy ``torch.backends.cuda`` toggles when
    running on older PyTorch releases.
    """

    context = _modern_sdpa_context(
        prefer_flash=prefer_flash,
        prefer_mem_efficient=prefer_mem_efficient,
        prefer_math=prefer_math,
        set_priority=set_priority,
    )
    if context is not None:
        return context

    return _legacy_sdpa_context(
        enable_flash=prefer_flash,
        enable_mem_efficient=prefer_mem_efficient,
        enable_math=prefer_math,
    )


def _modern_sdpa_context(
    *,
    prefer_flash: bool,
    prefer_mem_efficient: bool,
    prefer_math: bool,
    set_priority: bool,
) -> Optional[contextlib.AbstractContextManager]:
    nn_attention = getattr(torch.nn, "attention", None)
    if nn_attention is None:
        return None

    sdpa_kernel = getattr(nn_attention, "sdpa_kernel", None)
    if not callable(sdpa_kernel):
        return None

    backend_cls = getattr(nn_attention, "SDPBackend", None)
    if backend_cls is None:
        return None

    backends: list = []
    if prefer_flash and hasattr(backend_cls, "FLASH_ATTENTION"):
        backends.append(backend_cls.FLASH_ATTENTION)
    if prefer_mem_efficient and hasattr(backend_cls, "EFFICIENT_ATTENTION"):
        backends.append(backend_cls.EFFICIENT_ATTENTION)
    if prefer_math and hasattr(backend_cls, "MATH"):
        backends.append(backend_cls.MATH)

    if not backends:
        return None

    try:
        return sdpa_kernel(backends, set_priority=set_priority)
    except RuntimeError:
        return None


def _legacy_sdpa_context(
    *,
    enable_flash: bool,
    enable_mem_efficient: bool,
    enable_math: bool,
) -> Optional[contextlib.AbstractContextManager]:
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return None

    toggles: list[tuple] = []
    for enable_name, query_name, desired in (
        ("enable_flash_sdp", "flash_sdp_enabled", enable_flash),
        ("enable_mem_efficient_sdp", "mem_efficient_sdp_enabled", enable_mem_efficient),
        ("enable_math_sdp", "math_sdp_enabled", enable_math),
    ):
        enable_fn = getattr(cuda_backends, enable_name, None)
        if not callable(enable_fn):
            continue
        query_fn = getattr(cuda_backends, query_name, None)
        previous: Optional[bool]
        if callable(query_fn):
            try:
                previous = bool(query_fn())
            except TypeError:
                previous = None
        else:
            previous = None
        toggles.append((enable_fn, previous, bool(desired)))

    if not toggles:
        return None

    @contextmanager
    def _context():
        for enable_fn, _, desired_value in toggles:
            enable_fn(desired_value)
        try:
            yield
        finally:
            for enable_fn, previous_value, _ in reversed(toggles):
                if previous_value is not None:
                    enable_fn(previous_value)

    return _context()


__all__ = ["build_sdpa_context"]
