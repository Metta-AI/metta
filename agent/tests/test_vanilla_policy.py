"""Test that MettaAgent requires specific methods, which can be provided by PyTorchAgentMixin."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin
from metta.rl.system_config import SystemConfig


class VanillaTorchPolicy(torch.nn.Module):
    """A simple vanilla torch.nn.Module policy for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)

    def forward(self, td: TensorDict, state=None, action: torch.Tensor | None = None) -> TensorDict:
        """Simple forward pass that returns mock actions."""
        # Get batch size
        if "env_obs" in td:
            batch_size = td["env_obs"].shape[0]
        else:
            batch_size = 1

        # Return mock actions
        td["actions"] = torch.zeros((batch_size, 2), dtype=torch.long)
        td["act_log_prob"] = torch.zeros((batch_size,))
        td["values"] = torch.zeros((batch_size,))
        return td


class VanillaPolicyWithMixin(PyTorchAgentMixin, VanillaTorchPolicy):
    """A vanilla policy enhanced with the PyTorchAgentMixin."""

    def __init__(self):
        super().__init__()
        # Initialize the mixin
        self.init_mixin(clip_range=0)

    def reset_memory(self):
        """Minimal memory reset implementation."""
        pass

    def get_memory(self) -> dict:
        """Minimal memory getter implementation."""
        return {}


def test_vanilla_policy_fails_without_mixin():
    """Test that a vanilla torch.nn.Module policy fails without the mixin."""

    # Create minimal environment mock
    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
            self.obs_width = 10
            self.obs_height = 10
            self.single_action_space = gym.spaces.Discrete(10)
            self.feature_normalizations = {}

    # Create configs
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = DictConfig({"clip_range": 0})

    # Create MettaAgent with vanilla policy (no mixin)
    vanilla_policy = VanillaTorchPolicy()
    agent = MettaAgent(MinimalEnv(), system_cfg, agent_cfg, policy=vanilla_policy)

    # Test that initialization FAILS without required methods
    features = {
        "health": {"id": 1, "type": "scalar", "normalization": 100.0},
        "energy": {"id": 2, "type": "scalar", "normalization": 50.0},
    }
    action_names = ["move", "attack"]
    action_max_params = [3, 1]

    # This should fail because vanilla policy doesn't have initialize_to_environment
    with pytest.raises(AttributeError, match="initialize_to_environment"):
        agent.initialize_to_environment(features, action_names, action_max_params, "cpu")


def test_metta_agent_with_mixin_policy():
    """Test that MettaAgent works correctly with a policy that has the mixin."""

    # Create minimal environment mock
    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
            self.obs_width = 10
            self.obs_height = 10
            self.single_action_space = gym.spaces.Discrete(10)
            self.feature_normalizations = {}

    # Create configs
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = DictConfig({"clip_range": 0})

    # Create MettaAgent with mixin-enhanced policy
    mixin_policy = VanillaPolicyWithMixin()
    agent = MettaAgent(MinimalEnv(), system_cfg, agent_cfg, policy=mixin_policy)

    # Test that initialization works
    features = {
        "health": {"id": 1, "type": "scalar", "normalization": 100.0},
        "energy": {"id": 2, "type": "scalar", "normalization": 50.0},
    }
    action_names = ["move", "attack"]
    action_max_params = [3, 1]

    # This should work now that the policy has the mixin
    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Verify that MettaAgent stored the features and actions
    assert agent.original_feature_mapping == {"health": 1, "energy": 2}
    assert agent.action_names == ["move", "attack"]
    assert agent.action_max_params == [3, 1]

    # Test forward pass
    td = TensorDict({"env_obs": torch.zeros((4, 10, 10, 3), dtype=torch.uint8)})
    output = agent(td)

    # Verify output has expected keys
    assert "actions" in output
    assert output["actions"].shape == (4, 2)

    # Test memory methods (mixin provides minimal implementation)
    agent.reset_memory()  # Should not error
    memory = agent.get_memory()  # Should return empty dict
    assert memory == {}

    metrics = agent.compute_weight_metrics()  # Should return list via mixin
    assert isinstance(metrics, list)

    # Test that re-initialization works without errors (detailed remapping tests are in test_feature_remapping.py)
    new_features = {
        "health": {"id": 5, "type": "scalar", "normalization": 100.0},  # Different ID
        "energy": {"id": 7, "type": "scalar", "normalization": 50.0},  # Different ID
        "mana": {"id": 10, "type": "scalar", "normalization": 30.0},  # New feature
    }

    # Re-initialize in eval mode - should work without errors
    agent.eval()
    agent.initialize_to_environment(new_features, action_names, action_max_params, "cpu", is_training=False)

    # Verify basic functionality still works after re-initialization
    td = TensorDict({"env_obs": torch.zeros((4, 10, 10, 3), dtype=torch.uint8)})
    output = agent(td)
    assert "actions" in output
    assert output["actions"].shape == (4, 2)


def test_training_mode_learns_new_features():
    """Test that in training mode, new features are learned and added to the mapping."""

    # Create minimal environment mock
    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
            self.obs_width = 10
            self.obs_height = 10
            self.single_action_space = gym.spaces.Discrete(10)
            self.feature_normalizations = {}

    # Create configs
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = DictConfig({"clip_range": 0})

    # Create MettaAgent with mixin-enhanced policy
    mixin_policy = VanillaPolicyWithMixin()
    agent = MettaAgent(MinimalEnv(), system_cfg, agent_cfg, policy=mixin_policy)

    # Initial features
    features = {
        "health": {"id": 1, "type": "scalar", "normalization": 100.0},
        "energy": {"id": 2, "type": "scalar", "normalization": 50.0},
    }
    action_names = ["move", "attack"]
    action_max_params = [3, 1]

    # Initialize with initial features
    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Verify original mapping stored
    assert agent.original_feature_mapping == {"health": 1, "energy": 2}

    # New features with different IDs and a new feature
    new_features = {
        "health": {"id": 5, "type": "scalar", "normalization": 100.0},  # Different ID
        "energy": {"id": 7, "type": "scalar", "normalization": 50.0},  # Different ID
        "mana": {"id": 10, "type": "scalar", "normalization": 30.0},  # New feature
    }

    # Re-initialize in TRAINING mode (default)
    agent.train()
    agent.initialize_to_environment(new_features, action_names, action_max_params, "cpu", is_training=True)

    # Verify core training mode behavior: new features should be learned
    assert "mana" in agent.original_feature_mapping  # New feature learned
    assert agent.original_feature_mapping["mana"] == 10

    # Verify agent still functions correctly after training mode re-initialization
    td = TensorDict({"env_obs": torch.zeros((4, 10, 10, 3), dtype=torch.uint8)})
    output = agent(td)
    assert "actions" in output
    assert output["actions"].shape == (4, 2)


def test_mixin_provides_all_required_methods():
    """Test that the mixin provides all methods required by MettaAgent."""

    # Create a more complex policy that uses mixin features
    class PolicyUsingMixinFeatures(PyTorchAgentMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(100, 50)
            self.linear2 = torch.nn.Linear(50, 10)
            self.init_mixin(clip_range=1.0)  # Enable weight clipping

        def forward(self, td: TensorDict, state=None, action=None) -> TensorDict:
            batch_size = td.get("env_obs", torch.zeros(1, 10, 10, 3)).shape[0]
            td["actions"] = torch.zeros((batch_size, 2), dtype=torch.long)
            td["act_log_prob"] = torch.zeros((batch_size,))
            td["values"] = torch.zeros((batch_size,))
            return td

        def reset_memory(self):
            pass

        def get_memory(self) -> dict:
            return {}

    # Create minimal environment
    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
            self.obs_width = 10
            self.obs_height = 10
            self.single_action_space = gym.spaces.Discrete(10)
            self.feature_normalizations = {}

    system_cfg = SystemConfig(device="cpu")
    agent_cfg = DictConfig({"clip_range": 1.0})

    policy = PolicyUsingMixinFeatures()
    agent = MettaAgent(MinimalEnv(), system_cfg, agent_cfg, policy=policy)

    # Initialize environment
    features = {"test": {"id": 1, "type": "scalar"}}
    agent.initialize_to_environment(features, ["action"], [1], "cpu")

    # Test action conversion methods work
    if hasattr(policy, "_convert_action_to_logit_index"):
        # These require the tensors set by initialize_to_environment
        action = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        indices = policy._convert_action_to_logit_index(action)
        assert indices.shape[0] == 2
