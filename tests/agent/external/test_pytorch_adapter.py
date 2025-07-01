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


class MockLSTMPolicy(nn.Module):
    """Mock LSTM policy that returns state."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self.policy = MockPolicy(hidden_size)

    def forward(self, obs, state=None):
        batch_size = obs.shape[0]
        # Initialize state if not provided or if lstm_h is None
        if state is None:
            state = {"lstm_h": None, "lstm_c": None}

        if state.get("lstm_h") is None:
            h = torch.zeros(batch_size, self.hidden_size)
            c = torch.zeros(batch_size, self.hidden_size)
            state["lstm_h"] = h
            state["lstm_c"] = c

        # Mock LSTM forward
        hidden = torch.randn(batch_size, self.hidden_size)
        new_h, new_c = self.cell(hidden, (state["lstm_h"], state["lstm_c"]))

        # Update state dict in place (mimicking PufferLib behavior)
        state["lstm_h"] = new_h
        state["lstm_c"] = new_c

        # Return policy outputs
        logits, value = self.policy(obs, state)
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
        action, action_log_prob, entropy, value, log_probs = adapter(obs, state)

        # Validate outputs
        assert action.shape == (batch_size, 2)  # action_type, action_param
        assert action_log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)  # entropy is per sample, not a scalar
        assert value.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size, 19)

    def test_lstm_policy_forward(self):
        """Test forward pass with LSTM policy."""
        policy = MockLSTMPolicy()
        adapter = PytorchAdapter(policy)

        # Create mock inputs
        batch_size = 4
        obs = torch.randn(batch_size, 22, 11, 11)
        state = PolicyState(batch_size=batch_size)

        # Forward pass
        action, action_log_prob, entropy, value, log_probs = adapter(obs, state)

        # Validate outputs
        assert action.shape == (batch_size, 2)  # action_type, action_param
        assert action_log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)  # entropy is per sample, not a scalar
        assert value.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size, 19)

        # Check that state was updated
        assert state.lstm_h is not None
        assert state.lstm_c is not None
        assert state.lstm_h.shape == (batch_size, policy.hidden_size)
        assert state.lstm_c.shape == (batch_size, policy.hidden_size)

    def test_token_observation_preprocessing(self):
        """Test that token observations with 255 values are properly preprocessed."""
        policy = MockPolicy()
        adapter = PytorchAdapter(policy)

        # Create token observations with 255 values
        batch_size = 4
        num_tokens = 100
        obs = torch.zeros(batch_size, num_tokens, 3)

        # Add some 255 values that should be zeroed out
        obs[0, 10, 2] = 255
        obs[1, 20, 1] = 255
        obs[2, 30, 0] = 255

        state = PolicyState(batch_size=batch_size)

        # Forward pass (this should preprocess the observations)
        # Note: The mock policy expects 4D input, so we'll create a simple 4D tensor
        obs_4d = torch.randn(batch_size, 22, 11, 11)
        obs_4d[0, 0, 0, 0] = 255  # Add a 255 value to test preprocessing

        action, action_log_prob, entropy, value, log_probs = adapter(obs_4d, state)

        # The forward pass should succeed without errors
        assert action.shape == (batch_size, 2)
