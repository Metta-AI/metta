"""Unit tests for LSTM policy implementation."""

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from cogames.policy.lstm import LSTMPolicyNet


class MockEnv:
    """Mock environment for testing."""

    def __init__(self):
        self.single_observation_space = Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
        self.single_action_space = Discrete(8)


def test_forward_return_signature():
    """Test that forward_eval returns exactly 2 values (not 3).

    PufferLib expects forward_eval to return only (logits, values), not (logits, values, state).
    The state is managed externally via in-place dict updates.
    """
    env = MockEnv()
    net = LSTMPolicyNet(env)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # This should return exactly 2 values (logits, values)
    # NOT 3 values (logits, values, new_state) which was the old API
    result = net.forward_eval(obs, None)
    assert len(result) == 2, f"Expected 2 values, got {len(result)}"

    logits, values = result
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, MockEnv().single_action_space.n)
    assert values.shape == (4, 1)  # Batch of 4, single value per obs


def test_forward_with_dict_state():
    """Test that forward_eval works with dict state and updates it in-place.

    PufferLib passes state as a dict with shape (batch_size, num_layers, hidden_size)
    and expects it to be updated in-place with the same shape.
    """
    env = MockEnv()
    net = LSTMPolicyNet(env)
    batch_size = 4
    obs = torch.randint(0, 256, (batch_size, 7, 7, 3))

    # Test with dict state (PufferLib format: batch_size, num_layers, hidden_size)
    state = {
        "lstm_h": torch.zeros(batch_size, 1, net.hidden_size),
        "lstm_c": torch.zeros(batch_size, 1, net.hidden_size),
    }

    # Save initial state for comparison
    initial_h = state["lstm_h"].clone()

    logits, values = net.forward_eval(obs, state)

    # Check that state was updated in-place
    assert "lstm_h" in state, "state dict should still have lstm_h key"
    assert "lstm_c" in state, "state dict should still have lstm_c key"
    assert state["lstm_h"].shape == (batch_size, 1, net.hidden_size)
    assert state["lstm_c"].shape == (batch_size, 1, net.hidden_size)

    # Check that state was actually updated (not same as initial)
    assert not torch.allclose(state["lstm_h"], initial_h), "State should be updated"


def test_forward_with_empty_dict_state():
    """Test that forward_eval works with an empty dict (no initial state).

    Empty dict means no LSTM state, so the dict should remain empty.
    """
    env = MockEnv()
    net = LSTMPolicyNet(env)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # Empty dict - should be treated as no state
    state = {}

    logits, values = net.forward_eval(obs, state)

    # Empty dict should stay empty (no state to update)
    assert len(state) == 0, "Empty dict should remain empty"


def test_forward_with_none_state():
    """Test that forward_eval works with None state (no LSTM state)."""
    env = MockEnv()
    net = LSTMPolicyNet(env)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # None state means no LSTM state
    logits, values = net.forward_eval(obs, None)

    # Should still work and return valid outputs
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, MockEnv().single_action_space.n)
    assert values.shape == (4, 1)


def test_forward_method_matches_forward_eval():
    """Test that forward() method returns same signature as forward_eval()."""
    env = MockEnv()
    net = LSTMPolicyNet(env)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # Both should return (logits, values)
    result1 = net.forward(obs, None)
    result2 = net.forward_eval(obs, None)

    assert len(result1) == len(result2) == 2


def test_training_noise_changes_logits():
    env = MockEnv()

    torch.manual_seed(0)
    baseline = LSTMPolicyNet(env)
    torch.manual_seed(0)
    noisy = LSTMPolicyNet(env, noise_std=0.25)
    noisy.load_state_dict(baseline.state_dict())

    obs = torch.randint(0, 256, (2, 7, 7, 3))

    baseline.train()
    noisy.train()

    torch.manual_seed(42)
    baseline_logits, _ = baseline.forward_eval(obs, None)
    torch.manual_seed(42)
    noisy_logits, _ = noisy.forward_eval(obs, None)

    assert not torch.allclose(noisy_logits, baseline_logits), "Noise should perturb training logits"


def test_eval_noise_disabled_by_default():
    env = MockEnv()

    torch.manual_seed(123)
    baseline = LSTMPolicyNet(env)
    torch.manual_seed(123)
    noisy = LSTMPolicyNet(env, noise_std=0.5)
    noisy.load_state_dict(baseline.state_dict())

    obs = torch.randint(0, 256, (3, 7, 7, 3))

    baseline.eval()
    noisy.eval()

    torch.manual_seed(99)
    baseline_logits, _ = baseline.forward_eval(obs, None)
    torch.manual_seed(99)
    noisy_logits, _ = noisy.forward_eval(obs, None)

    assert torch.allclose(noisy_logits, baseline_logits), "Evaluation should be noise-free by default"


def test_eval_noise_flag_enables_noise():
    env = MockEnv()

    torch.manual_seed(7)
    baseline = LSTMPolicyNet(env)
    torch.manual_seed(7)
    noisy_eval = LSTMPolicyNet(env, noise_std=0.5, noise_during_eval=True)
    noisy_eval.load_state_dict(baseline.state_dict())

    obs = torch.randint(0, 256, (3, 7, 7, 3))

    baseline.eval()
    noisy_eval.eval()

    torch.manual_seed(5)
    baseline_logits, _ = baseline.forward_eval(obs, None)
    torch.manual_seed(5)
    noisy_logits, _ = noisy_eval.forward_eval(obs, None)

    assert not torch.allclose(noisy_logits, baseline_logits), "Noise flag should perturb eval logits"
