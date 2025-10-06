import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, TrainablePolicy
from cogames.policy.utils import ActionLayout
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation

logger = logging.getLogger("cogames.policies.simple_policy")


class SimplePolicyNet(torch.nn.Module):
    """Feed-forward baseline for discrete action spaces."""

    def __init__(self, env: MettaGridEnv) -> None:
        super().__init__()
        self.hidden_size = 128
        obs_size = int(np.prod(env.single_observation_space.shape))
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, self.hidden_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )

        self._layout = ActionLayout.from_env(env)

        self.action_head = torch.nn.Linear(self.hidden_size, self._layout.total_actions)
        self.value_head = torch.nn.Linear(self.hidden_size, 1)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.shape[0]
        obs_flat = observations.view(batch_size, -1).float() / 255.0
        hidden = self.net(obs_flat)
        logits = self.action_head(hidden)

        if state is not None and "action" in state and state["action"] is not None:
            action_tensor = torch.as_tensor(state["action"], device=logits.device)
            flat = self._layout.encode_torch(action_tensor.view(-1, 2)).view(action_tensor.shape[:-1])
            state["action"] = flat.to(state["action"].device, dtype=state["action"].dtype)

        values = self.value_head(hidden)
        return logits, values

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_eval(observations, state)


class SimpleAgentPolicyImpl(AgentPolicy):
    """Per-agent policy wrapper sharing the feed-forward network."""

    def __init__(self, net: SimplePolicyNet, device: torch.device, layout: ActionLayout) -> None:
        self._net = net
        self._device = device
        self._layout = layout

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            flat_idx = dist.sample().cpu().numpy().astype(np.int64)

        pair = self._layout.decode_numpy(flat_idx).astype(np.int32)
        return pair[0] if pair.ndim > 1 else pair


class SimplePolicy(TrainablePolicy):
    """Simple feedforward policy."""

    def __init__(self, env: MettaGridEnv, device: torch.device) -> None:
        super().__init__()
        self._net = SimplePolicyNet(env).to(device)
        self._device = device
        self._layout = self._net._layout
        self.action_layout_max_args = self._layout.max_args

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return SimpleAgentPolicyImpl(self._net, self._device, self._layout)

    def load_policy_data(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self._net.load_state_dict(state_dict)
        self._net = self._net.to(self._device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
