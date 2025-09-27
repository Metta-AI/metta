"""Torch AMP compatibility helpers for vendored Mamba."""

from functools import partial
from typing import Callable

import torch


def _custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool) -> Callable:
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)

    return decorator


if hasattr(torch.amp, "custom_fwd"):  # type: ignore[attr-defined]
    _deprecated = True
    from torch.amp import custom_fwd, custom_bwd  # type: ignore[attr-defined]
else:  # pragma: no cover - older torch
    _deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = _custom_amp_decorator(custom_fwd, _deprecated)
custom_bwd = _custom_amp_decorator(custom_bwd, _deprecated)

__all__ = ["custom_fwd", "custom_bwd"]
