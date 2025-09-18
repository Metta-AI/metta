"""Policy helpers and wrappers."""

import torch
from tensordict import TensorDict

from metta.agent.components.obs_shim import ObsShimBox, ObsShimTokens
from metta.agent.policy_architecture import PolicyArchitecture
from metta.agent.policy_base import Policy
from metta.rl.training.training_environment import EnvironmentMetaData


class ExternalPolicyWrapper(Policy):
    """
    For wrapping generic policies, aleiviating the need to conform to Metta's internal agent interface reqs.

    Expectations of the policy is that it takes a tensor of observations and returns a tensor of actions that matches
    the action space. That's to say that these policies will be used in evaluation, not in training.

    Policies that wish to be trained in metta should instead inherit from Policy and implement an agent experience spec,
    return the tensors needed for losses (ie values, entropy, and others depending on the loss), and the other methods
    if necessary.
    """

    def __init__(self, policy: nn.Module, env_metadata: EnvironmentMetaData, box_obs: bool = True):
        self.policy = policy
        if box_obs:
            self.obs_shaper = ObsShimBox(env=env_metadata, in_key="env_obs", out_key="obs")
        else:
            self.obs_shaper = ObsShimTokens(env=env_metadata, in_key="env_obs", out_key="obs")

    def forward(self, td: TensorDict) -> TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

    def get_agent_experience_spec(self):
        pass

    def initialize_to_environment(self, env_metadata: EnvironmentMetaData, device: torch.device):
        pass

    @property
    def device(self) -> torch.device:
        return self.policy.device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self):
        pass
