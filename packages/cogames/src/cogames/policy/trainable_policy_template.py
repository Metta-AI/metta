"""
Trainable Policy Template for the CoGames environment.

This template provides a minimal trainable neural network policy that can be used with
`cogames train`. It demonstrates the key interfaces required for training:

- MultiAgentPolicy: The main policy class that manages multiple agents
- AgentPolicy: Per-agent decision-making interface
- network(): Returns the nn.Module for training
- load_policy_data() / save_policy_data(): Checkpoint serialization

This implementation is closely based on mettagrid/policy/stateless.py, simplified for
clarity and without the pufferlib dependency.

To use this template:
1. Modify MyNetwork to implement your desired architecture
2. Run: cogames train -m easy_hearts -p class=my_trainable_policy.MyTrainablePolicy
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class MyNetwork(nn.Module):
    """A simple feedforward network for demonstration.

    Replace this with your own architecture (CNN, Transformer, etc.).
    The network receives observations and outputs action logits and value estimates.

    Important: PufferLib training requires forward_eval() to return (logits, values).
    """

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(int(np.prod(obs_shape)), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward_eval(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for evaluation. Called by PufferLib during rollout collection.

        Args:
            x: Observation tensor of shape (batch, *obs_shape). Values are uint8 [0, 255].
               See mettagrid/docs/observations.md for observation format details.
            state: RNN state dict (unused for stateless policies, but required by PufferLib)

        Returns:
            logits: Action logits of shape (batch, num_actions)
            values: Value estimates of shape (batch, 1)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1).float() / 255.0
        hidden = self.encoder(x)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training. Called by PufferLib during policy gradient updates.

        For stateless policies, this is identical to forward_eval. Stateful policies may
        differ (e.g., handling hidden state differently during backprop).
        """
        return self.forward_eval(x, state)


class MyAgentPolicy(AgentPolicy):
    """Per-agent policy that uses the shared network for inference."""

    def __init__(self, policy_env_info: PolicyEnvInterface, network: "MyNetwork", device: torch.device):
        super().__init__(policy_env_info)
        self._network = network
        self._action_names = policy_env_info.action_names
        self._device = device

    def step(self, obs: AgentObservation) -> Action:
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0).float()

        # Run inference and take argmax action
        with torch.no_grad():
            self._network.eval()
            logits, _ = self._network.forward_eval(obs_tensor)
            action_idx = logits.argmax(dim=-1).item()

        try:
            return Action(name=self._action_names[int(action_idx)])
        except IndexError:
            # Fallback if network outputs invalid action index
            return Action(name="noop") if "noop" in self._action_names else Action(name=self._action_names[0])


class MyTrainablePolicy(MultiAgentPolicy):
    """A trainable policy that can be used with `cogames train`.

    This policy:
    - Creates a neural network based on the environment's observation/action spaces
    - Provides per-agent policies via agent_policy()
    - Exposes the network for training via network()
    - Supports checkpoint save/load via load_policy_data() and save_policy_data()
    """

    # Uncomment to register a shorthand name (e.g., `cogames train -p class=my_trainable`)
    # short_names = ["my_trainable"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        hidden_size: int = 128,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        super().__init__(policy_env_info, **kwargs)

        obs_shape = policy_env_info.observation_space.shape
        num_actions = len(policy_env_info.action_names)
        self._network = MyNetwork(obs_shape, num_actions, hidden_size)

        if device is not None:
            self._network = self._network.to(torch.device(device))

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Return an AgentPolicy instance for the given agent."""
        current_device = next(self._network.parameters()).device
        return MyAgentPolicy(self._policy_env_info, self._network, current_device)

    def network(self) -> nn.Module:
        """Return the neural network module for training."""
        return self._network

    def is_recurrent(self) -> bool:
        """Return False for stateless policies."""
        return False

    def load_policy_data(self, path: str) -> None:
        """Load network weights from a checkpoint file."""
        device = next(self._network.parameters()).device
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self._network.load_state_dict(state_dict)
        self._network = self._network.to(device)

    def save_policy_data(self, path: str) -> None:
        """Save network weights to a checkpoint file."""
        torch.save(self._network.state_dict(), path)
