import logging

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.interfaces import AgentPolicy, TrainablePolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions

logger = logging.getLogger("cogames.policies.simple_policy")


class SimplePolicyNet(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.hidden_size = 128
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(
                torch.nn.Linear(np.prod(env.single_observation_space.shape), self.hidden_size)
            ),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )

        self.num_actions = int(env.single_action_space.n)

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


class SimpleAgentPolicyImpl(AgentPolicy):
    """Per-agent policy that uses the shared feedforward network."""

    def __init__(self, net: SimplePolicyNet, device: torch.device, num_actions: int):
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


class SimplePolicy(TrainablePolicy):
    """Simple feedforward policy."""

    def __init__(self, env: MettaGridEnv, device: torch.device):
        super().__init__()
        self._net = SimplePolicyNet(env).to(device)
        self._device = device
        self.num_actions = int(env.single_action_space.n)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create a Policy instance for a specific agent."""
        return SimpleAgentPolicyImpl(self._net, self._device, self.num_actions)

    def load_policy_data(self, checkpoint_path: str) -> None:
        self._net.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._net = self._net.to(self._device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
