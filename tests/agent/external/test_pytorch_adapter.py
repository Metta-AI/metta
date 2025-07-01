"""Unit tests for the PyTorch adapter for external policies."""

import torch
import torch.nn as nn

from metta.agent.external.pytorch_adapter import PytorchAdapter
from metta.agent.policy_state import PolicyState


class MockPolicy(nn.Module):
    """Mock policy for testing."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.actor = nn.ModuleList([nn.Linear(hidden_size, 9), nn.Linear(hidden_size, 10)])
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, obs, state=None):
        batch_size = obs.shape[0]
        # Mock logits for actions
        logits = torch.randn(batch_size, 19)  # 9 + 10 action logits
        value = torch.randn(batch_size, 1)
        return logits, value


class TestPytorchAdapter:
    """Test cases for PytorchAdapter."""

    def test_standard_policy_forward(self):
        """Test forward pass with standard policy."""
        policy = MockPolicy()
        adapter = PytorchAdapter(policy)

        # Create mock inputs
        batch_size = 4
        obs = torch.randn(batch_size, 22, 11, 11)
        state = PolicyState(batch_size=batch_size)

        # Forward pass
        actions, logprob, entropy, value, logits = adapter(obs, state)

        # Validate outputs
        assert actions.shape == (batch_size, 2)  # action_type, action_param
        assert logprob.shape == (batch_size,)
        assert entropy.shape == ()
        assert value.shape == (batch_size, 1)
        assert logits.shape == (batch_size, 19)
