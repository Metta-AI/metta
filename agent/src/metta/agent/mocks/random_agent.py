import torch
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent


class RandomAgent(MettaAgent):
    """
    A minimal agent that samples random valid actions from the environment's
    action space after initialize_to_environment() is called.
    """

    def __init__(self):
        # Avoid full MettaAgent init; set up minimal nn.Module state
        torch.nn.Module.__init__(self)
        self.device = "cpu"
        self.components = torch.nn.ModuleDict()
        self.action_names: list[str] = []
        self.action_max_params: list[int] = []

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        self.device = device
        self.action_names = action_names
        self.action_max_params = action_max_params

    def reset_memory(self):
        pass

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        # Determine batch size
        if "env_obs" in td:
            num_agents = td["env_obs"].shape[0]
        else:
            num_agents = td.batch_size[0] if td.batch_size else 1

        # Sample random action type and arg per agent
        num_actions = len(self.action_max_params) if self.action_max_params else 1
        action_types = torch.randint(low=0, high=max(1, num_actions), size=(num_agents,), dtype=torch.long)

        # For action arguments, use the max range regardless of action for simplicity
        max_arg = max(self.action_max_params) if self.action_max_params else 1
        action_args = torch.randint(low=0, high=max(1, max_arg), size=(num_agents,), dtype=torch.long)

        actions = torch.stack([action_types, action_args], dim=-1)
        td["actions"] = actions
        return td
