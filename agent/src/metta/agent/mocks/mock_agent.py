import torch
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent


class MockAgent(MettaAgent):
    """
    An agent that always does nothing. Used for tests and to run play without requiring a policy.

    This mock agent supports feature remapping for testing purposes while maintaining
    minimal functionality for simulation runs.
    """

    def __init__(self):
        # Don't call parent __init__ as it requires many parameters we don't have
        # Instead, manually initialize as nn.Module and set required attributes
        torch.nn.Module.__init__(self)

        # Initialize required attributes that MettaAgent expects
        self.components_with_memory = []
        self.components = torch.nn.ModuleDict()  # Use ModuleDict for proper nn.Module handling
        self.device = "cpu"

    def activate_actions(self, action_names, action_max_params, device):
        """Store action configuration for testing."""
        self.action_names = action_names
        self.action_max_params = action_max_params
        self.device = device

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        """
        Mock forward pass - always returns "do nothing" actions.

        This is a minimal implementation that satisfies the simulation's requirements:
        - Takes a TensorDict with "env_obs"
        - Adds "actions" key with shape [num_agents, 2] for [action_type, action_param]
        - Returns the modified TensorDict

        Args:
            td: TensorDict containing at least "env_obs"
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

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """
        Initialize the agent to work with a specific environment.

        For MockAgent, this sets up feature remapping support while maintaining
        minimal functionality.

        Note: is_training parameter is deprecated and ignored.
        """
        # Store action configuration
        self.activate_actions(action_names, action_max_params, device)

        # Initialize observations to support feature remapping
        self.activate_observations(features, device)

    def reset_memory(self):
        """Mock implementation - no memory to reset."""
        pass

    def get_memory(self):
        """Mock implementation - returns empty memory dict."""
        return {}
