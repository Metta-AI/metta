"""Utilities for analyzing policy network structure."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torch.utils.hooks import RemovableHandle

from metta.agent.policy_auto_builder import PolicyAutoBuilder


def compute_dormant_neuron_stats(policy: nn.Module, *, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Compute dormant neuron statistics for a policy.

    A neuron is considered dormant when the mean absolute weight for its outgoing connections
    falls below the supplied threshold. Results are returned as a dictionary compatible with
    wandb logging (keys are already namespaced).
    """

    if not isinstance(policy, nn.Module):
        return {}

    total_neurons = 0
    dormant_neurons = 0
    stats: Dict[str, float] = {}

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue

        weight_tensor = torch.nan_to_num(weight.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        # Use mean absolute weight per output unit to measure activity.
        activity = weight_tensor.abs().mean(dim=1)
        if activity.numel() == 0:
            continue

        layer_total = int(activity.numel())
        layer_dormant = int((activity < threshold).sum().item())
        if layer_total == 0:
            continue

        total_neurons += layer_total
        dormant_neurons += layer_dormant

        sanitized = name.replace(".", "/") if name else "root"
        stats[f"weights/dormant_neurons/{sanitized}"] = layer_dormant / layer_total

    if total_neurons == 0:
        return {}

    dormant_fraction = dormant_neurons / total_neurons
    stats["weights/dormant_neurons/total_neurons"] = float(total_neurons)
    stats["weights/dormant_neurons/dormant_neurons"] = float(dormant_neurons)
    stats["weights/dormant_neurons/fraction"] = dormant_fraction
    stats["weights/dormant_neurons/active_fraction"] = 1.0 - dormant_fraction

    return stats


class _ReLUActivationTracker:
    """Collects activation statistics for registered components."""

    def __init__(self) -> None:
        self._active: Dict[str, list[int]] = {}
        self._totals: Dict[str, list[int]] = {}

    def is_tracking(self, component: str) -> bool:
        return component in self._active

    def update(self, component: str, tensor: torch.Tensor) -> None:
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            flat = tensor.unsqueeze(0)
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])

        active_counts = (flat > 0).sum(dim=0).tolist()
        total = int(flat.shape[0])

        active = self._active.setdefault(component, [0] * len(active_counts))
        totals = self._totals.setdefault(component, [0] * len(active_counts))

        if len(active) < len(active_counts):
            active.extend([0] * (len(active_counts) - len(active)))
            totals.extend([0] * (len(active_counts) - len(totals)))

        for idx, value in enumerate(active_counts):
            active[idx] += int(value)
            totals[idx] += total

    def mark_component(self, component: str) -> None:
        self._active.setdefault(component, [])
        self._totals.setdefault(component, [])

    def metrics(self, *, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for component, active in self._active.items():
            totals = self._totals[component]
            for idx, (act, tot) in enumerate(zip(active, totals, strict=False)):
                if tot == 0:
                    continue
                metrics[f"activations/relu/{component}/{idx}"] = act / tot
        if reset:
            self._active.clear()
            self._totals.clear()
        return metrics


_DEFAULT_RELU_COMPONENTS: Dict[str, Any] = {"actor_mlp": "actor_hidden"}


def attach_relu_activation_hooks(
    policy: PolicyAutoBuilder,
    *,
    components: Optional[Mapping[str, Any]] = None,
) -> list[tuple[str, Callable[[PolicyAutoBuilder, str, nn.Module], RemovableHandle], str]]:
    """
    Build hook specifications for tracking ReLU activations on the supplied policy.

    Returns a list of ``(component_name, hook_factory, hook_type)`` tuples that can be passed to
    ``PolicyAutoBuilder.register_component_hook_rule``.
    """

    if not isinstance(policy, PolicyAutoBuilder):
        return []

    tracker: _ReLUActivationTracker | None = getattr(policy, "_relu_activation_tracker", None)
    if tracker is None:
        tracker = _ReLUActivationTracker()
        policy._relu_activation_tracker = tracker  # type: ignore[attr-defined]

    component_map = components if components is not None else _DEFAULT_RELU_COMPONENTS

    specs: list[tuple[str, Callable[[PolicyAutoBuilder, str, nn.Module], RemovableHandle], str]] = []

    for component_name, spec in component_map.items():
        if tracker.is_tracking(component_name):
            continue
        component = policy.components.get(component_name)
        if component is None:
            continue

        extractor = _build_extractor(spec)
        if extractor is None:
            continue

        def hook_factory(
            _policy: PolicyAutoBuilder,
            name: str,
            module: nn.Module,
            *,
            extractor_fn: Callable[[Any], Optional[torch.Tensor]] = extractor,
        ) -> torch.utils.hooks.RemovableHandle:
            def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
                tensor = extractor_fn(output)
                if tensor is None:
                    return
                tracker.update(name, tensor.detach())

            return module.register_forward_hook(hook)

        tracker.mark_component(component_name)
        specs.append((component_name, hook_factory, "forward"))

    return specs

def get_relu_activation_metrics(policy: Any, *, reset: bool = False) -> Dict[str, float]:
    tracker: _ReLUActivationTracker | None = getattr(policy, "_relu_activation_tracker", None)
    if tracker is None:
        return {}
    return tracker.metrics(reset=reset)


def _build_extractor(spec: Any) -> Optional[Callable[[Any], Optional[torch.Tensor]]]:
    if callable(spec):
        return spec

    if isinstance(spec, str):
        key = spec

        def extractor(output: Any) -> Optional[torch.Tensor]:
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, TensorDictBase):
                try:
                    tensor = output.get(key)
                except KeyError:
                    return None
                return tensor if isinstance(tensor, torch.Tensor) else None
            if isinstance(output, dict):
                tensor = output.get(key)
                return tensor if isinstance(tensor, torch.Tensor) else None
            return None

        return extractor

    return None


def compute_saturated_activation_stats(
    policy: nn.Module,
    *,
    activation: str,
    derivative_threshold: float = 1e-3,
) -> Dict[str, float]:
    """
    Compute saturated neuron statistics for activations with bounded derivatives.

    This helper targets architectures where a linear layer feeds into either a tanh or
    sigmoid activation. A neuron is flagged as saturated when the magnitude of the
    activation derivative (evaluated at its bias) falls below ``derivative_threshold``.
    """

    if not isinstance(policy, nn.Module):
        return {}

    supported = {"tanh", "sigmoid"}
    if activation not in supported:
        raise ValueError(f"Unsupported activation '{activation}'. Expected one of {sorted(supported)}.")

    total_neurons = 0
    saturated_neurons = 0
    stats: Dict[str, float] = {}

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        out_features = getattr(module, "out_features", None)
        if out_features is None:
            continue

        bias = getattr(module, "bias", None)
        if bias is None:
            device = module.weight.device if hasattr(module, "weight") else torch.device("cpu")
            bias_tensor = torch.zeros(out_features, device=device)
        else:
            if bias.ndim != 1 or bias.numel() != out_features:
                continue
            bias_tensor = bias.detach()

        bias_tensor = torch.nan_to_num(bias_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if activation == "tanh":
            derivative = 1.0 - torch.tanh(bias_tensor).pow(2)
        else:  # activation == "sigmoid"
            sigma = torch.sigmoid(bias_tensor)
            derivative = sigma * (1.0 - sigma)

        if derivative.numel() == 0:
            continue

        layer_total = int(derivative.numel())
        layer_saturated = int((derivative.abs() < derivative_threshold).sum().item())
        if layer_total == 0:
            continue

        total_neurons += layer_total
        saturated_neurons += layer_saturated

        sanitized = name.replace(".", "/") if name else "root"
        stats[f"activations/{activation}/saturated_fraction/{sanitized}"] = layer_saturated / layer_total

    if total_neurons == 0:
        return {}

    saturated_fraction = saturated_neurons / total_neurons
    prefix = f"activations/{activation}/saturation"
    stats[f"{prefix}/total_neurons"] = float(total_neurons)
    stats[f"{prefix}/saturated_neurons"] = float(saturated_neurons)
    stats[f"{prefix}/fraction"] = saturated_fraction
    stats[f"{prefix}/active_fraction"] = 1.0 - saturated_fraction

    return stats
