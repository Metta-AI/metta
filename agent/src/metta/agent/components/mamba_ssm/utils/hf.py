"""HuggingFace checkpoint helpers for the vendored Mamba implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import torch

try:  # pragma: no cover - dependency guard
    from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
    from transformers.utils.hub import cached_file
except ImportError:  # pragma: no cover - informative fallback
    CONFIG_NAME = WEIGHTS_NAME = None
    cached_file = None


def _require_transformers() -> None:
    if cached_file is None:
        raise ImportError(
            "transformers is required for load_config_hf/load_state_dict_hf. "
            "Install it via `pip install transformers` to enable checkpoint loading."
        )


def load_config_hf(model_name: str) -> Dict[str, Any]:
    """Load a HuggingFace config json for ``model_name``."""

    _require_transformers()
    resolved = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict_hf(
    model_name: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load a checkpoint state dict from HuggingFace for ``model_name``.

    Args:
        model_name: HuggingFace repo/model identifier.
        device: Optional device to place tensors on. Defaults to CPU.
        dtype: Optional dtype conversion applied after loading.
    """

    _require_transformers()

    map_location: Any
    if dtype not in (None, torch.float32):
        map_location = "cpu"
    else:
        map_location = device or "cpu"

    resolved = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    state_dict = torch.load(resolved, map_location=map_location)

    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

    if device is not None and device != map_location:
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}

    return state_dict
