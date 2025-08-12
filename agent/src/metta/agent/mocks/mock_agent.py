import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent
from metta.rl.system_config import SystemConfig


class MockPolicy(torch.nn.Module):
    """
    A simple policy that always does nothing. Used for tests and to run play without requiring a real policy.

    This mock policy supports feature remapping for testing purposes while maintaining
    minimal functionality for simulation runs.
    """

    def __init__(self):
        super().__init__()
        # Initialize required attributes that policies might need
        self.components_with_memory = []
        self.components = torch.nn.ModuleDict()  # Use ModuleDict for proper nn.Module handling
        self.device = "cpu"
        self.cum_action_max_params = None
        self.action_index_tensor = None

    def forward(self, td: TensorDict, state=None, action: torch.Tensor | None = None) -> TensorDict:
        """
        Mock forward pass - always returns "do nothing" actions.

        This is a minimal implementation that satisfies the simulation's requirements:
        - Takes a TensorDict with "env_obs"
        - Adds "actions" key with shape [num_agents, 2] for [action_type, action_param]
        - Returns the modified TensorDict

        Args:
            td: TensorDict containing at least "env_obs"
            state: Optional state (ignored in mock)
            action: Optional action tensor (ignored in mock - only used in training)

        Returns:
            TensorDict with mock actions added
        """
        # Get batch size from env_obs if it exists, otherwise default to 1
        if "env_obs" in td:
            env_obs = td["env_obs"]
            num_agents = env_obs.shape[0]
        else:
            # Some tests might not provide env_obs, default to batch size 1
            num_agents = td.batch_size[0] if td.batch_size else 1

        # Create "do nothing" actions (action_type=0, action_param=0)
        # These are the minimal valid actions that won't cause errors
        actions = torch.zeros((num_agents, 2), dtype=torch.long)

        # Add required outputs to the TensorDict
        # The simulation expects at least the "actions" key
        td["actions"] = actions

        # These are optional but might be expected by some code paths:
        # td["act_log_prob"] = torch.zeros((num_agents,))
        # td["values"] = torch.zeros((num_agents,))

        return td

    def update_feature_remapping(self, remap_tensor: torch.Tensor):
        """Update feature remapping in observation component."""
        if "_obs_" in self.components:
            obs_component = self.components["_obs_"]
            if hasattr(obs_component, "update_feature_remapping"):
                obs_component.update_feature_remapping(remap_tensor)

    def update_normalization_factors(self, features: dict[str, dict], original_feature_mapping: dict[str, int] | None):
        """Update normalization factors - mock implementation."""
        pass

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings - mock implementation."""
        pass

    def reset_memory(self):
        """Mock implementation - no memory to reset."""
        pass

    def get_memory(self):
        """Mock implementation - returns empty memory dict."""
        return {}


class MockAgent(MettaAgent):
    """
    A wrapper that creates a MettaAgent with a MockPolicy.

    This class is used for compatibility with existing tests that expect a MockAgent.
    It creates a MettaAgent instance with a MockPolicy as its policy.
    """

    def __init__(self):
        # Create a minimal environment mock
        import gymnasium as gym
        import numpy as np

        class MinimalEnv:
            def __init__(self):
                # Use proper gym space for single_observation_space
                self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
                self.obs_width = 10
                self.obs_height = 10
                self.single_action_space = gym.spaces.Discrete(10)  # Simple discrete action space
                self.feature_normalizations = {}

        # Create minimal configs
        system_cfg = SystemConfig(device="cpu")
        agent_cfg = DictConfig({"clip_range": 0})

        # Initialize MettaAgent with a MockPolicy
        mock_policy = MockPolicy()
        super().__init__(MinimalEnv(), system_cfg, agent_cfg, policy=mock_policy)

    @property
    def components(self):
        """Provide access to policy's components for backward compatibility."""
        return self.policy.components
