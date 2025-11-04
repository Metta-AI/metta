"""Utilities for analyzing policy network structure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from metta.tools.train import PostHookBuilder


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


class ReLUActivationState:
    """State container tracking ReLU activation frequencies per component."""

    def __init__(self) -> None:
        self.active: dict[str, list[int]] = {}
        self.totals: dict[str, list[int]] = {}

    def has_component(self, component_name: str) -> bool:
        return component_name in self.active

    def register_component(self, component_name: str) -> None:
        self.active.setdefault(component_name, [])
        self.totals.setdefault(component_name, [])

    def update(self, component_name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            flat = tensor.unsqueeze(0)
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])

        counts = (flat > 0).sum(dim=0).tolist()
        if not counts:
            return
        total = int(flat.shape[0])

        active = self.active.setdefault(component_name, [0] * len(counts))
        totals = self.totals.setdefault(component_name, [0] * len(counts))

        if len(active) < len(counts):
            active.extend([0] * (len(counts) - len(active)))
            totals.extend([0] * (len(counts) - len(totals)))

        for idx, value in enumerate(counts):
            if idx >= len(active):
                active.append(0)
                totals.append(0)
            active[idx] += int(value)
            totals[idx] += total

    def to_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for component, active_values in self.active.items():
            component_totals = self.totals.get(component, [])
            for idx, (act, tot) in enumerate(zip(active_values, component_totals, strict=False)):
                if tot:
                    metrics[f"activations/relu/{component}/{idx}"] = act / tot
        return metrics

    def reset(self) -> None:
        for component, values in self.active.items():
            self.active[component] = [0] * len(values)
        for component, values in self.totals.items():
            self.totals[component] = [0] * len(values)


def attach_relu_activation_hooks(
    *,
    extractor: Any = "actor_hidden",
) -> "PostHookBuilder":
    """Create a builder that prepares ReLU activation tracking for a given component."""

    def builder(component_name: str, trainer: Any) -> Optional[Callable[..., None]]:
        state = getattr(trainer, "_relu_activation_state", None)
        if not isinstance(state, ReLUActivationState):
            state = ReLUActivationState()
            trainer._relu_activation_state = state  # type: ignore[attr-defined]
            context = getattr(trainer, "context", None)
            if context is not None:
                context.relu_activation_state = state

        if state.has_component(component_name):
            return None

        extractor_fn = _build_extractor(extractor)
        if extractor_fn is None:
            return None

        tracker = state
        tracker.register_component(component_name)

        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            tensor = extractor_fn(output)
            if tensor is None:
                return
            tracker.update(component_name, tensor.detach())

        return hook

    return builder


def get_relu_activation_metrics(state: Any, *, reset: bool = False) -> Dict[str, float]:
    if not isinstance(state, ReLUActivationState):
        return {}
    metrics = state.to_metrics()
    if reset:
        state.reset()
    return metrics


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
