from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from metta.rl.model_analysis import compute_dormant_neuron_stats


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
