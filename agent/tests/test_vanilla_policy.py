"""Test that MettaAgent works with vanilla torch.nn.Module policies."""

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent
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


def test_metta_agent_with_vanilla_policy():
    """Test that MettaAgent works correctly with a vanilla torch.nn.Module policy."""

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

    # Create MettaAgent with vanilla policy using AgentConfig pattern
    from metta.agent.agent_config import AgentConfig

    vanilla_policy = VanillaTorchPolicy()
    # Use string agent config name for testing
    agent_cfg_name = "vanilla_policy_test"
    config = AgentConfig(env=MinimalEnv(), system_cfg=system_cfg, agent_cfg=agent_cfg_name, policy=vanilla_policy)
    agent = MettaAgent(config)

    # Test that initialization works
    features = {
        "health": {"id": 1, "type": "scalar", "normalization": 100.0},
        "energy": {"id": 2, "type": "scalar", "normalization": 50.0},
    }
    action_names = ["move", "attack"]
    action_max_params = [3, 1]

    # This should work without errors even though vanilla policy doesn't have these methods
    agent.initialize_to_environment(features, action_names, action_max_params, torch.device("cpu"))

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

    # Test memory methods (vanilla policy doesn't have them, should handle gracefully)
    agent.reset_memory()  # Should not error
    memory = agent.get_memory()  # Should return empty dict
    assert memory == {}

    # Test feature remapping on re-initialization
    new_features = {
        "health": {"id": 5, "type": "scalar", "normalization": 100.0},  # Different ID
        "energy": {"id": 7, "type": "scalar", "normalization": 50.0},  # Different ID
        "mana": {"id": 10, "type": "scalar", "normalization": 30.0},  # New feature
    }

    # Re-initialize in eval mode
    agent.eval()
    agent.initialize_to_environment(new_features, action_names, action_max_params, torch.device("cpu"))

    # Check that remapping was created (even though vanilla policy can't use it)
    assert agent.feature_id_remap[5] == 1  # health remapped
    assert agent.feature_id_remap[7] == 2  # energy remapped
    assert agent.feature_id_remap[10] == 255  # mana mapped to UNKNOWN in eval mode


def test_metta_agent_fallback_methods():
    """Test that MettaAgent provides sensible fallbacks for missing policy methods."""

    # Create a truly minimal policy with just forward
    class MinimalPolicy(torch.nn.Module):
        def forward(self, td, state=None, action=None):
            td["actions"] = torch.zeros((1, 2), dtype=torch.long)
            return td

    # Create minimal environment
    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
            self.obs_width = 10
            self.obs_height = 10
            self.single_action_space = gym.spaces.Discrete(10)
            self.feature_normalizations = {}

    system_cfg = SystemConfig(device="cpu")
    agent_cfg = DictConfig({})

    # Create agent using AgentConfig pattern
    from metta.agent.agent_config import AgentConfig

    policy = MinimalPolicy()
    # Use string agent config name for testing
    agent_cfg_name = "minimal_policy_test"
    config = AgentConfig(env=MinimalEnv(), system_cfg=system_cfg, agent_cfg=agent_cfg_name, policy=policy)
    agent = MettaAgent(config)

    # Test that all these methods work without errors
    features = {"test": {"id": 1, "type": "scalar"}}
    agent.initialize_to_environment(features, ["action"], [1], torch.device("cpu"))

    # These should all work gracefully even though the policy doesn't have these methods
    agent.reset_memory()
    memory = agent.get_memory()
    assert memory == {}

    # Policy doesn't have clip_weights, should handle gracefully
    agent.clip_weights()

    # Policy doesn't have l2_init_loss, should return 0
    loss = agent.l2_init_loss()
    assert loss == torch.tensor(0.0)

    # Policy doesn't have update_l2_init_weight_copy, should not error
    agent.update_l2_init_weight_copy()

    # Policy doesn't have compute_weight_metrics, should return empty list
    metrics = agent.compute_weight_metrics()
    assert metrics == []
