"""Utilities for analyzing policy network structure."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


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
