import logging
from typing import Dict, Optional, Tuple, Union

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


def _normalize_tuple_state(state: Optional[LSTMState]) -> Optional[LSTMState]:
    """Return LSTM state as (layers, batch, hidden) tuple."""

    if state is None:
        return None

    hidden, cell = state
    return _ensure_three_dims(hidden), _ensure_three_dims(cell)


def _state_from_dict(state: StateDict) -> Optional[LSTMState]:
    """Convert batch-first state dict into layers-first tuple."""

    if not state:
        return None

    hidden = state.get("lstm_h")
    cell = state.get("lstm_c")
    if hidden is None or cell is None:
        return None

    hidden_layers = _ensure_three_dims(hidden).transpose(0, 1).contiguous()
    cell_layers = _ensure_three_dims(cell).transpose(0, 1).contiguous()
    return hidden_layers, cell_layers


def _update_state_dict(target: StateDict, state: Optional[LSTMState]) -> None:
    """Populate ``target`` dict with batch-first state tensors."""

    target.clear()
    if state is None:
        return

    normalized = _normalize_tuple_state(state)
    if normalized is None:
        return

    hidden, cell = normalized
    target["lstm_h"] = hidden.transpose(0, 1).contiguous().detach()
    target["lstm_c"] = cell.transpose(0, 1).contiguous().detach()


def _unpack_state(
    state: Optional[Union[LSTMState, StateDict]],
) -> Tuple[Optional[LSTMState], Optional[StateDict]]:
    """Return normalized tuple state and optional dict reference."""

    if state is None:
        return None, None

    if isinstance(state, tuple):
        return _normalize_tuple_state(state), None

    if isinstance(state, dict):
        return _state_from_dict(state), state

    msg = f"Unsupported LSTM state container type: {type(state)!r}"
    raise TypeError(msg)


class LSTMPolicyNet(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        # Public: Required by PufferLib for RNN state management
        self.hidden_size = 128
        self._obs_shape = tuple(env.single_observation_space.shape)
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

    def _forward_internal(
        self,
        observations: torch.Tensor,
        state: Optional[LSTMState],
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, Optional[LSTMState]]:
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

        rnn_state = _normalize_tuple_state(state)
        hidden, new_state = self._rnn(hidden, rnn_state)

        hidden = hidden.view(batch_size * bptt_horizon, self.hidden_size)
        logits = self._action_head(hidden)
        logits = logits.split(self._action_nvec, dim=1)
        values = self._value_head(hidden)
        return logits, values, new_state

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Union[LSTMState, StateDict]] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        tuple_state, dict_state = _unpack_state(state)
        logits, values, new_state = self._forward_internal(observations, tuple_state)
        if dict_state is not None:
            _update_state_dict(dict_state, new_state)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Union[LSTMState, StateDict]] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        return self.forward_eval(observations, state)

    def forward_agent(
        self,
        observations: torch.Tensor,
        state: Optional[LSTMState],
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, Optional[LSTMState]]:
        logits, values, new_state = self._forward_internal(observations, state)
        if new_state is not None:
            new_state = tuple(component.detach() for component in new_state)
        return logits, values, new_state


class LSTMAgentPolicy(StatefulAgentPolicy[LSTMState]):
    """Per-agent policy that uses the shared LSTM network."""

    def __init__(self, net: LSTMPolicyNet, device: torch.device, action_nvec: tuple):
        self._net = net
        self._device = device
        self._action_nvec = action_nvec
        self._obs_shape = getattr(net, "_obs_shape", None)

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
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self._device)
        else:
            obs_tensor = torch.as_tensor(obs, device=self._device, dtype=torch.float32)

        expected_dims = len(self._obs_shape) if self._obs_shape is not None else None
        if expected_dims is not None and obs_tensor.dim() == expected_dims:
            obs_tensor = obs_tensor.unsqueeze(0)
        elif obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        elif expected_dims is None and obs_tensor.dim() < 2:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            self._net.eval()

            if torch.isnan(obs_tensor).any():
                logger.error("NaN in observation! obs shape: %s", obs_tensor.shape)
            if torch.isinf(obs_tensor).any():
                logger.error("Inf in observation! obs shape: %s", obs_tensor.shape)

            logits, _, new_state = self._net.forward_agent(obs_tensor, state)

            if any(torch.isnan(logit).any() for logit in logits):
                logger.error(
                    "NaN in logits! obs shape: %s, obs min/max: %s/%s",
                    obs_tensor.shape,
                    obs_tensor.min(),
                    obs_tensor.max(),
                )
                for name, param in self._net.named_parameters():
                    if torch.isnan(param).any():
                        logger.error("NaN in parameter %s", name)

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
        super().load_policy_data(checkpoint_path)
        self._net = self._net.to(self._device)
        self._agent_policy._net = self._net
