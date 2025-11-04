"""Utilities for analyzing policy network structure."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import torch
import torch.nn as nn

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


def attach_relu_activation_hooks(
    policy: PolicyAutoBuilder,
    *,
    components: Iterable[str] = ("actor_mlp",),
) -> None:
    """
    Attach forward hooks that track ReLU activation rates for the specified components.

    The hooks collect per-neuron activation counts under ``policy._hooks``; callers can
    read the accumulated data via ``policy._hooks.forward_handles`` or expose a custom accessor.
    """

    tracker: dict[str, List[int]] = {}
    totals: dict[str, List[int]] = {}

    def hook_factory(policy_obj: PolicyAutoBuilder, name: str, module: nn.Module) -> torch.utils.hooks.RemovableHandle:
        if not isinstance(module, nn.Sequential):  # for MLP, module is TensorDictSequential
            pass

        def forward_hook(_, __, output: torch.Tensor) -> None:
            tensor = output.detach()
            if tensor.ndim == 0:
                return
            if tensor.ndim == 1:
                flat = tensor.unsqueeze(0)
            else:
                flat = tensor.reshape(-1, tensor.shape[-1])

            active_counts = (flat > 0).sum(dim=0)
            total = flat.shape[0]

            act = tracker.setdefault(name, [0] * active_counts.numel())
            tot = totals.setdefault(name, [0] * active_counts.numel())

            for idx, value in enumerate(active_counts.tolist()):
                act[idx] += value
                tot[idx] += total

        return module.register_forward_hook(forward_hook)

    for component_name in components:
        policy.register_component_hook_rule(
            component_name=component_name,
            hook_factory=hook_factory,
            hook_type="forward",
        )


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
