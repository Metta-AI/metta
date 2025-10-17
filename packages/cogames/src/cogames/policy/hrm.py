"""Hierarchical Reasoning Model (HRM) policy implementation.

The HRM policy factors decision making into two levels:

1. A *manager* that produces soft attention over a fixed set of strategic
   branches. Each branch can be interpreted as a coarse reasoning template.
2. A set of *experts* (one per branch) that transform the shared embedding
   into action logits. The manager attention is used to blend the experts'
   contributions into the final policy logits.

This keeps the policy compatible with the existing feedforward training loop
while yielding interpretable per-branch weights that downstream tooling can log
or visualise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.interfaces import AgentPolicy, TrainablePolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions


def _normalise_observations(observations: torch.Tensor) -> torch.Tensor:
    """Normalise raw uint8 observations to floats in [0, 1]."""
    if observations.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return observations.float() / 255.0
    if observations.max() > 1.0:
        return observations.float() / 255.0
    return observations.float()


class HRMPolicyNet(torch.nn.Module):
    """Two-level hierarchical reasoning policy network."""

    def __init__(
        self,
        env: MettaGridEnv,
        *,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_branches: int = 4,
    ) -> None:
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        self.num_actions = int(env.single_action_space.n)
        self.num_branches = num_branches

        self._encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(obs_dim, embedding_dim)),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(embedding_dim, embedding_dim)),
            nn.GELU(),
        )

        self._manager = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_branches),
        )

        self._experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.num_actions),
                )
                for _ in range(num_branches)
            ]
        )

        self._value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return blended action logits and value estimates."""
        del state  # HRM is feedforward-only
        batch = observations.shape[0]
        flat_obs = observations.view(batch, -1)
        encoded = self._encoder(_normalise_observations(flat_obs))

        manager_logits = self._manager(encoded)
        expert_weights = torch.softmax(manager_logits, dim=-1)

        expert_logits = torch.stack([expert(encoded) for expert in self._experts], dim=1)
        combined_logits = torch.sum(expert_logits * expert_weights.unsqueeze(-1), dim=1)

        values = self._value_head(encoded)
        return combined_logits, values

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Torch forward hook delegates to forward_eval."""
        return self.forward_eval(observations, state)

    def reasoning_weights(self, observations: torch.Tensor) -> torch.Tensor:
        """Expose manager attention weights for interpretability tooling."""
        batch = observations.shape[0]
        flat_obs = observations.view(batch, -1)
        encoded = self._encoder(_normalise_observations(flat_obs))
        return torch.softmax(self._manager(encoded), dim=-1)


class HRMAgentPolicyImpl(AgentPolicy):
    """Per-agent wrapper around the shared HRM network."""

    def __init__(self, net: HRMPolicyNet, device: torch.device):
        self._net = net
        self._device = device

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0)
        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample().cpu().item()
            return dtype_actions.type(action_idx)


class HRMPolicy(TrainablePolicy):
    """Trainable policy exposing hierarchical reasoning weights."""

    def __init__(self, env: MettaGridEnv, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self._net = HRMPolicyNet(env).to(device)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        del agent_id
        return HRMAgentPolicyImpl(self._net, self._device)

    def policy_diagnostics(self, observations: torch.Tensor) -> torch.Tensor:
        """Return manager attention weights for a batch of observations."""
        self._net.eval()
        with torch.no_grad():
            return self._net.reasoning_weights(observations.to(self._device))
