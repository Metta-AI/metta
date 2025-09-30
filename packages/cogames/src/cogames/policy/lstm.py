import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, StatefulAgentPolicy, TrainablePolicy
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
        state: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
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

        # Handle state being passed as either a dict (from PufferLib) or tuple (from our API)
        rnn_state = None
        state_is_dict = isinstance(state, dict)
        state_has_keys = state_is_dict and "lstm_h" in state and "lstm_c" in state

        if state is not None:
            if state_has_keys:
                # PufferLib passes state as dict with "lstm_h" and "lstm_c" keys
                h, c = state["lstm_h"], state["lstm_c"]
                # Handle None state (initial state)
                if h is not None and c is not None:
                    # PufferLib uses shape (batch_size, num_layers, hidden_size)
                    # but PyTorch LSTM expects (num_layers, batch_size, hidden_size)
                    if h.dim() == 3:
                        h = h.transpose(0, 1)
                        c = c.transpose(0, 1)
                    elif h.dim() == 2:
                        h = h.unsqueeze(0)
                        c = c.unsqueeze(0)
                    elif h.dim() == 1:
                        h = h.unsqueeze(0).unsqueeze(0)
                        c = c.unsqueeze(0).unsqueeze(0)
                    rnn_state = (h, c)
            elif not state_is_dict:
                # Tuple state for inference mode - assume correct shape
                h, c = state
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                    c = c.unsqueeze(0)
                elif h.dim() == 1:
                    h = h.unsqueeze(0).unsqueeze(0)
                    c = c.unsqueeze(0).unsqueeze(0)
                rnn_state = (h, c)

        hidden, new_state = self._rnn(hidden, rnn_state)

        # If a dict was provided, update it in-place with the new state for subsequent calls
        if state_is_dict and new_state is not None:
            h, c = new_state
            if h.dim() == 3:
                h_store = h.transpose(0, 1)
                c_store = c.transpose(0, 1)
            elif h.dim() == 2:
                h_store = h.unsqueeze(1)
                c_store = c.unsqueeze(1)
            elif h.dim() == 1:
                h_store = h.unsqueeze(0).unsqueeze(1)
                c_store = c.unsqueeze(0).unsqueeze(1)
            else:
                h_store = h
                c_store = c
            state["lstm_h"] = h_store.detach()
            state["lstm_c"] = c_store.detach()

        hidden = rearrange(hidden, "b t h -> (b t) h")
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


class LSTMAgentPolicy(StatefulAgentPolicy[Tuple[torch.Tensor, torch.Tensor]]):
    """Per-agent policy that uses the shared LSTM network."""

    def __init__(self, net: LSTMPolicyNet, device: torch.device, action_nvec: tuple):
        self._net = net
        self._device = device
        self._action_nvec = action_nvec

    def agent_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get initial state for a new agent.

        For LSTM, we return None and let the network initialize the state on first forward pass.
        """
        return None

    def step_with_state(
        self,
        obs: Union[MettaGridObservation, torch.Tensor],
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[MettaGridAction, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get action and update state for this agent."""
        # Convert single observation to batch of 1 for network forward pass
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self._device).unsqueeze(0) if obs.dim() < 2 else obs.to(self._device)
        else:
            obs_tensor = torch.tensor(obs, device=self._device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            # For inference, we pass state through a dict so forward_eval can populate it
            state_dict = {"lstm_h": None, "lstm_c": None}
            if state is not None:
                state_dict["lstm_h"], state_dict["lstm_c"] = state

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
                h, c = state_dict["lstm_h"], state_dict["lstm_c"]
                new_state = (h.detach(), c.detach())

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
