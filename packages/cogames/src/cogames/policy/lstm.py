import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, StatefulAgentPolicy, TrainablePolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation

logger = logging.getLogger("cogames.policies.lstm_policy")


StateDict = Dict[str, torch.Tensor]
LSTMState = Tuple[torch.Tensor, torch.Tensor]


def _ensure_three_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Return tensor shaped as (layers, batch, hidden)."""

    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).unsqueeze(1)
    if tensor.dim() == 0:
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return tensor


def _state_from_container(
    state: Optional[Union[LSTMState, StateDict]],
) -> Optional[LSTMState]:
    """Normalize state containers to a tuple with layers-first tensors."""

    if state is None:
        return None

    if isinstance(state, tuple):
        hidden, cell = state
        return _ensure_three_dims(hidden), _ensure_three_dims(cell)

    if isinstance(state, dict):
        hidden = state.get("lstm_h")
        cell = state.get("lstm_c")
        if hidden is None or cell is None:
            return None
        hidden_layers = _ensure_three_dims(hidden).transpose(0, 1)
        cell_layers = _ensure_three_dims(cell).transpose(0, 1)
        return hidden_layers, cell_layers

    msg = f"Unsupported LSTM state container type: {type(state)!r}"
    raise TypeError(msg)


def _write_state_to_dict(target: StateDict, state: LSTMState) -> None:
    """Store layers-first state tuple into batch-first dict (detached)."""

    hidden, cell = state
    target["lstm_h"] = hidden.transpose(0, 1).contiguous().detach()
    target["lstm_c"] = cell.transpose(0, 1).contiguous().detach()


class LSTMPolicyNet(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        # Public: Required by PufferLib for RNN state management
        self.hidden_size = 128
        self._obs_size = int(np.prod(env.single_observation_space.shape))

        self._net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(self._obs_size, self.hidden_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )

        self._rnn = torch.nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

        self._action_nvec = tuple(env.single_action_space.nvec)

        self._action_head = torch.nn.Linear(self.hidden_size, sum(self._action_nvec))
        self._value_head = torch.nn.Linear(self.hidden_size, 1)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        observations = observations.float()
        if observations.max() > 1.0:
            observations = observations / 255.0

        batch_size = observations.shape[0]
        obs_flat = observations.view(batch_size, -1)
        if obs_flat.shape[1] == self._obs_size:
            bptt_horizon = 1
            obs_steps = obs_flat
        else:
            if obs_flat.shape[1] % self._obs_size != 0:
                msg = (
                    "Observation tensor cannot be reshaped into expected input size. "
                    f"Received flattened size {obs_flat.shape[1]} for expected {self._obs_size}."
                )
                raise ValueError(msg)
            bptt_horizon = obs_flat.shape[1] // self._obs_size
            obs_steps = obs_flat.view(batch_size * bptt_horizon, self._obs_size)

        hidden = self._net(obs_steps)
        hidden = hidden.view(batch_size, bptt_horizon, self.hidden_size)

        rnn_state = _state_from_container(state)
        hidden, new_state = self._rnn(hidden, rnn_state)

        if isinstance(state, dict) and new_state is not None:
            _write_state_to_dict(state, new_state)

        hidden = hidden.view(batch_size * bptt_horizon, self.hidden_size)
        logits = self._action_head(hidden)
        logits = logits.split(self._action_nvec, dim=1)

        values = self._value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.forward_eval(observations, state)


class LSTMAgentPolicy(StatefulAgentPolicy[LSTMState]):
    """Per-agent policy that uses the shared LSTM network."""

    def __init__(self, net: LSTMPolicyNet, device: torch.device, action_nvec: tuple):
        self._net = net
        self._device = device
        self._action_nvec = action_nvec

    def agent_state(self) -> Optional[LSTMState]:
        """Get initial state for a new agent.

        For LSTM, we return None and let the network initialize the state on first forward pass.
        """
        return None

    def step_with_state(
        self,
        obs: Union[MettaGridObservation, torch.Tensor],
        state: Optional[LSTMState],
    ) -> Tuple[MettaGridAction, Optional[LSTMState]]:
        """Get action and update state for this agent."""
        # Convert single observation to batch of 1 for network forward pass
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self._device).unsqueeze(0) if obs.dim() < 2 else obs.to(self._device)
        else:
            obs_tensor = torch.tensor(obs, device=self._device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            # For inference, hold state in batch-first dict so forward_eval can reuse it
            state_dict: Dict[str, torch.Tensor] = {}
            if state is not None:
                _write_state_to_dict(state_dict, state)

            # Debug: check observation
            if torch.isnan(obs_tensor).any():
                logger.error(f"NaN in observation! obs shape: {obs_tensor.shape}, obs: {obs_tensor}")
            if torch.isinf(obs_tensor).any():
                logger.error(f"Inf in observation! obs shape: {obs_tensor.shape}")

            logits, _ = self._net.forward_eval(obs_tensor, state_dict)

            # Debug: check logits
            if any(torch.isnan(logit).any() for logit in logits):
                logger.error(
                    f"NaN in logits! obs shape: {obs_tensor.shape}, obs min/max: {obs_tensor.min()}/{obs_tensor.max()}"
                )
                logger.error(f"Logits: {[logit for logit in logits]}")
                # Check network parameters
                for name, param in self._net.named_parameters():
                    if torch.isnan(param).any():
                        logger.error(f"NaN in parameter {name}")

            # Extract the new state from the dict
            new_state = _state_from_container(state_dict)

            # Sample action from the logits
            actions = []
            for logit in logits:
                dist = torch.distributions.Categorical(logits=logit)
                actions.append(dist.sample().item())

            return np.array(actions, dtype=np.int32), new_state


class LSTMPolicy(TrainablePolicy):
    """LSTM-based policy that creates StatefulPolicy wrappers for each agent."""

    def __init__(self, env: MettaGridEnv, device: torch.device):
        super().__init__()
        self._net = LSTMPolicyNet(env).to(device)
        self._device = device
        self._action_nvec = tuple(env.single_action_space.nvec)
        self._agent_policy = LSTMAgentPolicy(self._net, device, self._action_nvec)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create a StatefulPolicy wrapper for a specific agent."""
        return StatefulAgentPolicy(self._agent_policy, agent_id)

    def load_policy_data(self, checkpoint_path: str) -> None:
        self._net.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._net = self._net.to(self._device)
        # Update the agent policy's reference to the network
        self._agent_policy._net = self._net

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
