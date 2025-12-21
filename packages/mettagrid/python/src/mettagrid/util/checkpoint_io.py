from __future__ import annotations

from typing import Mapping

import torch


def prepare_state_dict_for_save(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Prepare state dict for safetensors: detach, move to CPU, handle shared storage."""
    result: dict[str, torch.Tensor] = {}
    seen_storage: dict[int, str] = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State dict entry '{key}' is not a torch.Tensor")

        value = tensor.detach().cpu()
        data_ptr = value.data_ptr()

        if data_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage[data_ptr] = key

        result[key] = value

    return result
