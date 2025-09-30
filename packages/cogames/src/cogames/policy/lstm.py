import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, StatefulAgentPolicy, TrainablePolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation

logger = logging.getLogger("cogames.policies.lstm_policy")


@dataclass
class _RnnState:
    """Utility for converting between PyTorch and external LSTM state formats."""

    hidden: torch.Tensor  # (num_layers, batch, hidden_size)
    cell: torch.Tensor  # (num_layers, batch, hidden_size)

    @staticmethod
    def _expand_to_three_dims(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure state tensors have shape (layers, batch, hidden)."""
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).unsqueeze(1)
        if tensor.dim() == 0:
            return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return tensor

    @classmethod
    def from_container(
        cls,
        state: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
    ) -> Optional["_RnnState"]:
        """Convert user-provided state container to layers-first tensors."""

        if state is None:
            return None

        if isinstance(state, dict):
            hidden = state.get("lstm_h")
            cell = state.get("lstm_c")
            if hidden is None or cell is None:
                return None
            hidden_expanded = cls._expand_to_three_dims(hidden)
            cell_expanded = cls._expand_to_three_dims(cell)
            # Dict storage is batch-first -> transpose to layers-first
            if hidden_expanded.dim() == 3:
                hidden_expanded = hidden_expanded.transpose(0, 1)
                cell_expanded = cell_expanded.transpose(0, 1)
            return cls(hidden=hidden_expanded, cell=cell_expanded)

        if isinstance(state, tuple):
            hidden, cell = state
            hidden_expanded = cls._expand_to_three_dims(hidden)
            cell_expanded = cls._expand_to_three_dims(cell)
            return cls(hidden=hidden_expanded, cell=cell_expanded)

        msg = f"Unsupported LSTM state container type: {type(state)!r}"
        raise TypeError(msg)

    def assign_to_dict(self, target: Dict[str, torch.Tensor]) -> None:
        """Write the current state into a mutable dict in batch-first format."""

        hidden_store = self.hidden.transpose(0, 1).contiguous()
        cell_store = self.cell.transpose(0, 1).contiguous()
        target["lstm_h"] = hidden_store.detach()
        target["lstm_c"] = cell_store.detach()

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the state as a tuple in layers-first orientation."""

        return self.hidden.detach(), self.cell.detach()


@dataclass
class AgentLSTMState:
    """State container used by per-agent policies (batch-first orientation)."""

    hidden: torch.Tensor  # (batch=1, num_layers, hidden_size)
    cell: torch.Tensor  # (batch=1, num_layers, hidden_size)

    def to_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lstm_h": self.hidden, "lstm_c": self.cell}

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> "AgentLSTMState":
        hidden = state_dict["lstm_h"].detach()
        cell = state_dict["lstm_c"].detach()
        return cls(hidden=hidden, cell=cell)


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
        obs_flat = observations.reshape(batch_size, -1)
        features, remainder = divmod(obs_flat.shape[1], self._obs_size)
        if remainder != 0:
            msg = (
                "Observation tensor cannot be reshaped into expected input size. "
                f"Received flattened size {obs_flat.shape[1]} for expected {self._obs_size}."
            )
            raise ValueError(msg)

        bptt_horizon = max(features, 1)
        obs_flat = obs_flat.reshape(batch_size * bptt_horizon, self._obs_size)

        hidden = self._net(obs_flat)
        hidden = hidden.reshape(batch_size, bptt_horizon, self.hidden_size)

        rnn_state = _RnnState.from_container(state)
        rnn_state_tuple = rnn_state.as_tuple() if rnn_state is not None else None

        hidden, new_state_tuple = self._rnn(hidden, rnn_state_tuple)

        if isinstance(state, dict) and new_state_tuple is not None:
            new_state = _RnnState(hidden=new_state_tuple[0], cell=new_state_tuple[1])
            new_state.assign_to_dict(state)

        hidden = hidden.reshape(batch_size * bptt_horizon, self.hidden_size)
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


class LSTMAgentPolicy(StatefulAgentPolicy[AgentLSTMState]):
    """Per-agent policy that uses the shared LSTM network."""

    def __init__(self, net: LSTMPolicyNet, device: torch.device, action_nvec: tuple):
        self._net = net
        self._device = device
        self._action_nvec = action_nvec

    def agent_state(self) -> Optional[AgentLSTMState]:
        """Get initial state for a new agent.

        For LSTM, we return None and let the network initialize the state on first forward pass.
        """
        return None

    def step_with_state(
        self,
        obs: Union[MettaGridObservation, torch.Tensor],
        state: Optional[AgentLSTMState],
    ) -> Tuple[MettaGridAction, Optional[AgentLSTMState]]:
        """Get action and update state for this agent."""
        # Convert single observation to batch of 1 for network forward pass
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self._device).unsqueeze(0) if obs.dim() < 2 else obs.to(self._device)
        else:
            obs_tensor = torch.tensor(obs, device=self._device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            # For inference, hold state in batch-first dict so forward_eval can reuse it
            state_dict: Dict[str, torch.Tensor]
            if state is not None:
                state_dict = state.to_state_dict()
            else:
                state_dict = {}

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
            new_state = None
            if "lstm_h" in state_dict and "lstm_c" in state_dict:
                new_state = AgentLSTMState.from_state_dict(state_dict)

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
