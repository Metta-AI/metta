from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from metta.rl.model_analysis import compute_dormant_neuron_stats, compute_saturated_activation_stats


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 3, bias=False)
        self.layer2 = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.layer1.weight.zero_()
            self.layer2.weight.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))


def test_compute_dormant_neuron_stats_counts_layers() -> None:
    policy = _TinyPolicy()
    stats = compute_dormant_neuron_stats(policy, threshold=1e-8)

    assert stats["weights/dormant_neurons/total_neurons"] == 5.0
    assert stats["weights/dormant_neurons/dormant_neurons"] == 4.0
    assert stats["weights/dormant_neurons/fraction"] == 0.8
    assert stats["weights/dormant_neurons/active_fraction"] == pytest.approx(0.2)
    # Ensure layer-specific entries exist and use sanitized names
    assert stats["weights/dormant_neurons/layer1"] == 1.0
    assert stats["weights/dormant_neurons/layer2"] == 0.5


class _BiasOnlyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.tensor([0.0, 5.0, -6.0]))


def test_compute_saturated_activation_stats_tanh() -> None:
    policy = _BiasOnlyPolicy()
    stats = compute_saturated_activation_stats(policy, activation="tanh", derivative_threshold=1e-2)

    assert stats["activations/tanh/saturation/total_neurons"] == 3.0
    assert stats["activations/tanh/saturation/saturated_neurons"] == 2.0
    assert stats["activations/tanh/saturation/fraction"] == pytest.approx(2.0 / 3.0)
    assert stats["activations/tanh/saturation/active_fraction"] == pytest.approx(1.0 / 3.0)
    assert stats["activations/tanh/saturated_fraction/linear"] == pytest.approx(2.0 / 3.0)


def test_compute_saturated_activation_stats_sigmoid() -> None:
    policy = _BiasOnlyPolicy()
    stats = compute_saturated_activation_stats(policy, activation="sigmoid", derivative_threshold=5e-2)

    assert stats["activations/sigmoid/saturation/total_neurons"] == 3.0
    assert stats["activations/sigmoid/saturation/saturated_neurons"] == 2.0
    assert stats["activations/sigmoid/saturation/fraction"] == pytest.approx(2.0 / 3.0)
    assert stats["activations/sigmoid/saturation/active_fraction"] == pytest.approx(1.0 / 3.0)
