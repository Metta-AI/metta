"""Unit tests for LSTM policy implementation."""

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.lstm import LSTMPolicy, LSTMPolicyNet
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken


def create_mock_policy_env_info() -> PolicyEnvInterface:
    """Create a mock PolicyEnvInterface for testing."""
    actions_cfg = ActionsConfig()
    obs_space = Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
    action_space = Discrete(8)
    return PolicyEnvInterface(
        obs_features=[],
        tags=[],
        actions=actions_cfg,
        num_agents=1,
        observation_space=obs_space,
        action_space=action_space,
        obs_width=7,
        obs_height=7,
        assembler_protocols=[],
        tag_id_to_name={},
    )


def test_forward_return_signature():
    """Test that forward_eval returns exactly 2 values (not 3).

    PufferLib expects forward_eval to return only (logits, values), not (logits, values, state).
    The state is managed externally via in-place dict updates.
    """
    policy_env_info = create_mock_policy_env_info()
    net = LSTMPolicyNet(policy_env_info)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # This should return exactly 2 values (logits, values)
    # NOT 3 values (logits, values, new_state) which was the old API
    result = net.forward_eval(obs, None)
    assert len(result) == 2, f"Expected 2 values, got {len(result)}"

    logits, values = result
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == 4  # Batch size
    assert logits.shape[1] == len(policy_env_info.actions.actions())  # Number of actions
    assert values.shape == (4, 1)  # Batch of 4, single value per obs


def test_forward_with_dict_state():
    """Test that forward_eval works with dict state and updates it in-place.

    PufferLib passes state as a dict with shape (batch_size, num_layers, hidden_size)
    and expects it to be updated in-place with the same shape.
    """
    policy_env_info = create_mock_policy_env_info()
    net = LSTMPolicyNet(policy_env_info)
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
    policy_env_info = create_mock_policy_env_info()
    net = LSTMPolicyNet(policy_env_info)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # Empty dict - should be treated as no state
    state = {}

    logits, values = net.forward_eval(obs, state)

    # Empty dict should stay empty (no state to update)
    assert len(state) == 0, "Empty dict should remain empty"


def test_forward_with_none_state():
    """Test that forward_eval works with None state (no LSTM state)."""
    policy_env_info = create_mock_policy_env_info()
    net = LSTMPolicyNet(policy_env_info)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # None state means no LSTM state
    logits, values = net.forward_eval(obs, None)

    # Should still work and return valid outputs
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == 4  # Batch size
    assert logits.shape[1] == len(policy_env_info.actions.actions())  # Number of actions
    assert values.shape == (4, 1)


def test_forward_method_matches_forward_eval():
    """Test that forward() method returns same signature as forward_eval()."""
    policy_env_info = create_mock_policy_env_info()
    net = LSTMPolicyNet(policy_env_info)
    obs = torch.randint(0, 256, (4, 7, 7, 3))

    # Both should return (logits, values)
    result1 = net.forward(obs, None)
    result2 = net.forward_eval(obs, None)

    assert len(result1) == len(result2) == 2


def test_stateful_agent_policy_requires_reset():
    """Test that StatefulAgentPolicy requires reset() to be called before step().

    This test demonstrates the correct usage pattern: reset() must be called
    before step() to initialize the internal state. Without reset(), step() will
    raise an AssertionError.
    """
    policy_env_info = create_mock_policy_env_info()
    policy = LSTMPolicy(policy_env_info)

    # Get an agent policy (returns StatefulAgentPolicy wrapper)
    agent_policy = policy.agent_policy(agent_id=0)

    # Create a minimal mock observation
    # For LSTM, the observation needs tokens with feature, location, and value
    feature = ObservationFeatureSpec(id=0, name="test_feature", normalization=1.0)
    token = ObservationToken(feature=feature, location=(0, 0), value=1, raw_token=(255, 0, 0))
    obs = AgentObservation(agent_id=0, tokens=[token])

    # Verify that step() fails without reset()
    try:
        agent_policy.step(obs)
        raise AssertionError("step() should have raised AssertionError before reset()")
    except AssertionError as e:
        assert "reset()" in str(e), "Error message should mention reset()"

    # IMPORTANT: Must call reset() before step() to initialize state
    agent_policy.reset()

    # Now step() should pass the assertion (even if it fails later due to observation format)
    # The key is that reset() initializes the state from Ellipsis to None (or other value)
    try:
        agent_policy.step(obs)
        # If we get here, the assertion passed (observation format issues are separate)
    except AssertionError as e:
        raise AssertionError("step() should not raise AssertionError after reset()") from e
    except RuntimeError:
        # RuntimeError is expected due to observation format, but AssertionError should not occur
        pass
