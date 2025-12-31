from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from mettagrid.base_config import Config


class NodeConfig(Config):
    enabled: bool = Field(default=True)


def analyze_loss_alignment(
    shared_data,
    name1: str,
    name2: str,
    params: list[Tensor],
    tracker: dict[str, list[float]],
) -> None:
    key1 = f"{name1}_loss_vec"
    key2 = f"{name2}_loss_vec"

    if key1 not in shared_data or key2 not in shared_data:
        return

    loss_vec1 = shared_data[key1]
    loss_vec2 = shared_data[key2]

    vec1_flat = loss_vec1.flatten()
    vec2_flat = loss_vec2.flatten()

    if vec1_flat.shape != vec2_flat.shape:
        return

    with torch.no_grad():
        loss_cos = F.cosine_similarity(vec1_flat, vec2_flat, dim=0)
        tracker[f"{name1}_{name2}_loss_cos"].append(float(loss_cos.item()))

        loss_diff_var = (vec1_flat - vec2_flat).var()
        tracker[f"{name1}_{name2}_loss_diff_var"].append(float(loss_diff_var.item()))

    params_with_grad = [p for p in params if p.requires_grad]
    if not params_with_grad:
        return

    loss_scalar1 = loss_vec1.mean()
    loss_scalar2 = loss_vec2.mean()

    grads1 = torch.autograd.grad(
        loss_scalar1, params_with_grad, retain_graph=True, create_graph=False, allow_unused=True
    )
    grads2 = torch.autograd.grad(
        loss_scalar2, params_with_grad, retain_graph=True, create_graph=False, allow_unused=True
    )

    def _flatten_grads(grads, params):
        flat_list = []
        for g, p in zip(grads, params, strict=True):
            if g is None:
                flat_list.append(torch.zeros_like(p).flatten())
            else:
                flat_list.append(g.flatten())
        return torch.cat(flat_list)

    grad1_flat = _flatten_grads(grads1, params_with_grad).detach()
    grad2_flat = _flatten_grads(grads2, params_with_grad).detach()

    grad_cos = F.cosine_similarity(grad1_flat, grad2_flat, dim=0)
    tracker[f"{name1}_{name2}_grad_cos"].append(float(grad_cos.item()))

    grad_diff_var = (grad1_flat - grad2_flat).var()
    tracker[f"{name1}_{name2}_grad_diff_var"].append(float(grad_diff_var.item()))


@dataclass(slots=True)
class NodeState:
    policy: Any
    trainer_cfg: Any
    env: Any
    device: torch.device
    node_name: str
    cfg: NodeConfig
    replay: Any | None = None
    loss_tracker: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _zero_tensor: Tensor | None = None
    _state_attrs: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._zero_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.register_state_attr("loss_tracker")

    def _zero(self) -> Tensor:
        assert self._zero_tensor is not None
        return self._zero_tensor

    def stats(self) -> dict[str, float]:
        prefix = f"graph/{self.node_name}/"
        return {f"{prefix}{k}": (sum(v) / len(v) if v else 0.0) for k, v in self.loss_tracker.items()}

    def zero_loss_tracker(self) -> None:
        self.loss_tracker.clear()

    def register_state_attr(self, *names: str) -> None:
        for name in names:
            if not hasattr(self, name):
                raise AttributeError(f"Node state has no attribute '{name}' to register for state tracking")
            self._state_attrs.add(name)

    def state_dict(self) -> OrderedDict[str, Any]:
        state = OrderedDict()
        for name in sorted(self._state_attrs):
            value = getattr(self, name)
            state[name] = self._clone_state_value(value)
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], *, strict: bool = True) -> tuple[list[str], list[str]]:
        missing_keys: list[str] = [name for name in self._state_attrs if name not in state_dict]
        unexpected_keys: list[str] = [name for name in state_dict.keys() if name not in self._state_attrs]

        for name in self._state_attrs - set(missing_keys):
            self._restore_state_value(name, state_dict[name])

        if strict and (missing_keys or unexpected_keys):
            missing_msg = f"Missing keys: {missing_keys}" if missing_keys else ""
            unexpected_msg = f"Unexpected keys: {unexpected_keys}" if unexpected_keys else ""
            separator = "; " if missing_msg and unexpected_msg else ""
            raise RuntimeError(f"Error loading node state dict: {missing_msg}{separator}{unexpected_msg}")

        return missing_keys, unexpected_keys

    def _clone_state_value(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().clone().cpu()
        if isinstance(value, Mapping):
            return {k: self._clone_state_value(v) for k, v in value.items()}
        if isinstance(value, defaultdict):
            return {k: copy.deepcopy(v) for k, v in value.items()}
        if hasattr(value, "clone") and callable(value.clone):
            return value.clone()
        return copy.deepcopy(value)

    def _restore_state_value(self, name: str, stored_value: Any) -> None:
        current = getattr(self, name, None)

        if isinstance(current, Tensor):
            tensor = stored_value if isinstance(stored_value, Tensor) else torch.as_tensor(stored_value)
            setattr(self, name, tensor.to(device=current.device, dtype=current.dtype))
            return

        if isinstance(current, defaultdict):
            rebuilt = defaultdict(current.default_factory)
            for key, value in (stored_value or {}).items():
                rebuilt[key] = copy.deepcopy(value)
            setattr(self, name, rebuilt)
            return

        if isinstance(current, dict):
            setattr(self, name, {k: copy.deepcopy(v) for k, v in (stored_value or {}).items()})
            return

        if isinstance(stored_value, Tensor):
            setattr(self, name, stored_value.to(device=self.device))
            return

        setattr(self, name, copy.deepcopy(stored_value))
