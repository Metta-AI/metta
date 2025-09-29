import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy import TrainablePolicy
from mettagrid import MettaGridEnv

logger = logging.getLogger("cogames.examples.stateful_policy")


class StatefulPolicyNet(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.hidden_size = 128
        obs_size = int(np.prod(env.single_observation_space.shape))
        self._obs_shape = tuple(env.single_observation_space.shape)
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, self.hidden_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )

        self.rnn = torch.nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

        self.action_nvec = tuple(env.single_action_space.nvec)

        self.action_head = torch.nn.Linear(self.hidden_size, sum(self.action_nvec))
        self.value_head = torch.nn.Linear(self.hidden_size, 1)

    def initial_state(self, batch: int, device: torch.device) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros(batch, self.hidden_size, device=device)
        return {"lstm_h": zeros, "lstm_c": zeros}

    def _flatten_observations(self, observations: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        obs = observations.float() / 255.0
        obs_dims = len(self._obs_shape)
        if obs.dim() == obs_dims + 1:
            batch = obs.shape[0]
            time = 1
            flat = obs.view(batch, -1)
            return flat, batch, time
        if obs.dim() == obs_dims + 2:
            batch, time = obs.shape[:2]
            flat = obs.view(batch * time, -1)
            return flat, batch, time
        msg = f"Unsupported observation shape: {tuple(obs.shape)}"
        raise ValueError(msg)

    def _run_network(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        flat, batch, time = self._flatten_observations(observations)
        hidden = self.net(flat)
        hidden_seq = hidden.view(batch, time, self.hidden_size)

        lstm_state = None
        if state is not None:
            h = state.get("lstm_h")
            c = state.get("lstm_c")
            if h is not None and c is not None:
                lstm_state = (h.unsqueeze(0), c.unsqueeze(0))

        output, lstm_out = self.rnn(hidden_seq, lstm_state)
        h_out, c_out = lstm_out
        if state is not None:
            state["lstm_h"] = h_out.squeeze(0).detach()
            state["lstm_c"] = c_out.squeeze(0).detach()

        output_flat = output.reshape(batch * time, self.hidden_size)
        logits = self.action_head(output_flat).split(self.action_nvec, dim=1)
        values = self.value_head(output_flat).squeeze(-1)

        if time > 1:
            values = values.view(batch, time)
        else:
            values = values.view(batch)

        return logits, values

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if state is None:
            device = observations.device
            state = self.initial_state(observations.shape[0], device)
        logits, values = self._run_network(observations, state)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if state is None:
            device = observations.device
            batch = observations.shape[0]
            state = self.initial_state(batch, device)
        logits, values = self._run_network(observations, state)
        return logits, values


class StatefulPolicy(TrainablePolicy):
    def __init__(self, env: MettaGridEnv, device: torch.device):
        super().__init__()
        self._net = StatefulPolicyNet(env).to(device)
        self._device = device
        self.action_nvec = tuple(env.single_action_space.nvec)
        self._state: Optional[Dict[str, torch.Tensor]] = None

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
        if self._state is None:
            self._state = self._net.initial_state(batch=1, device=self._device)

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor, self._state)

            actions = []
            for logit in logits:
                dist = torch.distributions.Categorical(logits=logit)
                actions.append(dist.sample().item())

            return np.array(actions, dtype=np.int32)

    def reset(self) -> None:
        self._state = None

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self._net.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._net = self._net.to(self._device)
        self._state = None

    def save_checkpoint(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
