import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy import TrainablePolicy
from mettagrid import MettaGridEnv

logger = logging.getLogger("cogames.examples.simple_policy")


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

        self.action_nvec = tuple(env.single_action_space.nvec)
        num_actions = sum(self.action_nvec)
        self.decoder = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, num_actions), std=0.01)

        self.action_head = torch.nn.Linear(self.hidden_size, sum(self.action_nvec))
        self.value_head = torch.nn.Linear(self.hidden_size, 1)

    def forward_eval(self, observations, state=None):
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1).float() / 255.0
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        logits = logits.split(self.action_nvec, dim=1)

        values = self.value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class SimplePolicy(TrainablePolicy):
    def __init__(self, env: MettaGridEnv, device: torch.device):
        super().__init__()
        self._net = SimplePolicyNet(env).to(device)
        self._device = device
        self.action_nvec = tuple(env.single_action_space.nvec)

    def network(self) -> nn.Module:
        return self._net

    def step(self, agent_id: int, agent_obs: Any) -> Any:
        """Get action for a single agent given its observation.

        Args:
            agent_id: The ID of the agent (unused in this policy)
            agent_obs: The observation for this specific agent

        Returns:
            The action for this agent to take
        """
        # Convert single observation to batch of 1 for network forward pass
        obs_tensor = torch.tensor(agent_obs, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)

            # Sample action from the logits
            actions = []
            for logit in logits:
                dist = torch.distributions.Categorical(logits=logit)
                actions.append(dist.sample().item())

            return np.array(actions, dtype=np.int32)

    def reset(self) -> None:
        pass

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self._net.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._net = self._net.to(self._device)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
