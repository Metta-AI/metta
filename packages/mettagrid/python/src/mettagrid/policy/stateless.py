import logging

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action as MettaGridAction
from mettagrid.simulator import AgentObservation as MettaGridObservation

logger = logging.getLogger("mettagrid.policy.stateless_policy")


class StatelessPolicyNet(torch.nn.Module):
    """Stateless feedforward policy network."""

    def __init__(self, actions_cfg: ActionsConfig, obs_shape: tuple):
        super().__init__()
        self.hidden_size = 128
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(
                torch.nn.Linear(np.prod(obs_shape).item(), self.hidden_size),
            ),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )

        self.num_actions = len(actions_cfg.actions())

        self.action_head = torch.nn.Linear(self.hidden_size, self.num_actions)
        self.value_head = torch.nn.Linear(self.hidden_size, 1)

    def forward_eval(self, observations, state=None):
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1).float() / 255.0
        hidden = self.net(observations)
        logits = self.action_head(hidden)

        values = self.value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class StatelessAgentPolicyImpl(AgentPolicy):
    """Per-agent policy that uses the shared feedforward network."""

    def __init__(self, net: StatelessPolicyNet, device: torch.device, num_actions: int):
        self._net = net
        self._device = device
        self._num_actions = num_actions

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get action for this agent."""
        # Convert single observation to batch of 1 for network forward pass
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_action = dist.sample().cpu().item()
            return dtype_actions.type(sampled_action)


class StatelessPolicy(TrainablePolicy):
    """Stateless feedforward policy."""

    short_names = ["stateless"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: torch.device | str | None = None):
        super().__init__(policy_env_info)
        actions_cfg = policy_env_info.actions
        obs_shape = policy_env_info.observation_space.shape
        self._net = StatelessPolicyNet(actions_cfg, obs_shape)
        if device is not None:
            self._net = self._net.to(torch.device(device))
        self.num_actions = len(actions_cfg.actions())

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create a Policy instance for a specific agent."""
        current_device = next(self._net.parameters()).device
        return StatelessAgentPolicyImpl(self._net, current_device, self.num_actions)

    def is_recurrent(self) -> bool:
        return False

    def load_policy_data(self, checkpoint_path: str) -> None:
        device = next(self._net.parameters()).device
        state_dict = torch.load(checkpoint_path, map_location=device)
        self._net.load_state_dict(state_dict)
        self._net = self._net.to(device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
