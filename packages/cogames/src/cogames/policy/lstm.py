import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, StatefulAgentPolicy, TrainablePolicy
from cogames.policy.utils import LSTMState, LSTMStateDict
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation

logger = logging.getLogger("cogames.policies.lstm_policy")


class LSTMPolicyNet(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        # Public: Required by PufferLib for RNN state management
        self.hidden_size = 128

        self._net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(
                torch.nn.Linear(np.prod(env.single_observation_space.shape), self.hidden_size)
            ),
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
        state: Optional[Union[LSTMState, Tuple[torch.Tensor, torch.Tensor], LSTMStateDict]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Handle different input shapes:
        # - bptt_horizon=1: (batch_size, *obs_shape) e.g. (128, 7, 7, 3)
        # - bptt_horizon>1: (segments, bptt_horizon, *obs_shape) e.g. (128, 2, 7, 7, 3)
        # We need to output: (batch_size * bptt_horizon, flattened_obs_size)
        orig_shape = observations.shape

        # Figure out expected flattened obs size from self._net input size
        expected_obs_size = self._net[0].in_features  # First linear layer input size

        # Check if we have temporal dimension
        # With bptt_horizon>1, we get (segments, bptt_horizon, *obs_shape)
        # We need to detect this and reshape to (segments * bptt_horizon, obs_size)
        total_elements = observations.numel()
        batch_size = orig_shape[0]

        # If total_elements / batch_size != expected_obs_size, we have a temporal dimension
        if total_elements // batch_size != expected_obs_size:
            # We have (segments, bptt_horizon, *obs_shape)
            # Calculate bptt_horizon
            bptt_horizon = total_elements // (batch_size * expected_obs_size)
            segments = batch_size
            # Reshape to (segments * bptt_horizon, obs_size)
            observations = observations.reshape(segments * bptt_horizon, expected_obs_size).float()
        else:
            # Normal case: (batch_size, *obs_shape)
            segments = batch_size
            bptt_horizon = 1
            observations = observations.reshape(segments, expected_obs_size).float()

        # Only normalize if values are in uint8 range (0-255)
        if observations.max() > 1.0:
            observations = observations / 255.0

        hidden = self._net(observations)
        hidden = rearrange(hidden, "(b t) h -> b t h", t=bptt_horizon, b=segments)

        expected_layers = self._rnn.num_layers * (2 if self._rnn.bidirectional else 1)
        dict_state: Optional[LSTMStateDict] = state if isinstance(state, dict) else None

        tuple_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
        if dict_state is not None:
            lstm_state = LSTMState.from_dict(dict_state, expected_layers)
            tuple_state = lstm_state.to_tuple() if lstm_state is not None else None
        elif isinstance(state, LSTMState):
            tuple_state = state.to_tuple()
        else:
            tuple_state = state

        if tuple_state is not None:
            h, c = tuple_state
            if h.dim() == 2:
                h = h.unsqueeze(0)
                c = c.unsqueeze(0)
            elif h.dim() == 1:
                h = h.unsqueeze(0).unsqueeze(0)
                c = c.unsqueeze(0).unsqueeze(0)
            tuple_state = (h, c)

        hidden, new_state_tuple = self._rnn(hidden, tuple_state)
        new_state = LSTMState.from_tuple(new_state_tuple, expected_layers)

        if dict_state is not None and new_state is not None:
            new_state.write_dict(dict_state)

        hidden = rearrange(hidden, "b t h -> (b t) h")
        logits = self._action_head(hidden)
        logits = logits.split(self._action_nvec, dim=1)

        values = self._value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Union[LSTMState, Tuple[torch.Tensor, torch.Tensor], LSTMStateDict]] = None,
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
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self._device).unsqueeze(0) if obs.dim() < 2 else obs.to(self._device)
        else:
            obs_tensor = torch.tensor(obs, device=self._device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            state_dict: LSTMStateDict = {}
            if state is not None:
                state.write_dict(state_dict)

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
                for name, param in self._net.named_parameters():
                    if torch.isnan(param).any():
                        logger.error(f"NaN in parameter {name}")

            new_state = LSTMState.from_dict(state_dict, self._net._expected_layers)
            if new_state is not None:
                new_state = new_state.detach()

            # Sample action from the logits
            actions: list[int] = []
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
