import logging
from typing import Dict, List, Optional, Tuple, Union

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
    """Feed-forward encoder with LSTM head for sequential decision making."""

    def __init__(self, env: MettaGridEnv):
        super().__init__()
        self.hidden_size = 128

        obs_size = int(np.prod(env.single_observation_space.shape))
        self._encoder = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, self.hidden_size)),
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
        state: Optional[Union[LSTMState, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        orig_shape = observations.shape
        obs_size = self._encoder[0].in_features
        total_elements = observations.numel()
        batch_size = orig_shape[0]

        if total_elements // batch_size != obs_size:
            bptt_horizon = total_elements // (batch_size * obs_size)
            segments = batch_size
            obs_flat = observations.reshape(segments * bptt_horizon, obs_size).float()
        else:
            segments = batch_size
            bptt_horizon = 1
            obs_flat = observations.reshape(segments, obs_size).float()

        if obs_flat.max() > 1.0:
            obs_flat = obs_flat / 255.0

        hidden = self._encoder(obs_flat)
        hidden = rearrange(hidden, "(b t) h -> b t h", t=bptt_horizon, b=segments)

        rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        state_is_dict = isinstance(state, dict)
        state_has_keys = state_is_dict and "lstm_h" in state and "lstm_c" in state

        if state is not None:
            if state_has_keys:
                h, c = state["lstm_h"], state["lstm_c"]
                if h is not None and c is not None:
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
            else:
                h, c = state
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                    c = c.unsqueeze(0)
                elif h.dim() == 1:
                    h = h.unsqueeze(0).unsqueeze(0)
                    c = c.unsqueeze(0).unsqueeze(0)
                rnn_state = (h, c)

        hidden, new_state = self._rnn(hidden, rnn_state)

        if state_has_keys:
            h, c = new_state
            if h.dim() == 3:
                h = h.transpose(0, 1)
                c = c.transpose(0, 1)
            elif h.dim() == 2:
                h = h.unsqueeze(1)
                c = c.unsqueeze(1)
            elif h.dim() == 1:
                h = h.unsqueeze(0).unsqueeze(1)
                c = c.unsqueeze(0).unsqueeze(1)
            state["lstm_h"], state["lstm_c"] = h, c

        hidden = rearrange(hidden, "b t h -> (b t) h")
        logits = self._action_head(hidden).split(self._action_nvec, dim=1)
        values = self._value_head(hidden)
        return list(logits), values

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.forward_eval(observations, state)


class LSTMAgentPolicy(StatefulAgentPolicy[LSTMState]):
    """Per-agent policy that uses the shared LSTM network."""

    def __init__(self, net: LSTMPolicyNet, device: torch.device, action_nvec: tuple[int, ...]) -> None:
        self._net = net
        self._device = device
        self._action_nvec = action_nvec

    def agent_state(self) -> Optional[LSTMState]:
        return None

    def step_with_state(
        self,
        obs: Union[MettaGridObservation, torch.Tensor],
        state: Optional[LSTMState],
    ) -> Tuple[MettaGridAction, Optional[LSTMState]]:
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self._device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, device=self._device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            state_dict: LSTMStateDict = {"lstm_h": None, "lstm_c": None}
            if state is not None:
                hidden, cell = state.to_tuple()
                state_dict["lstm_h"], state_dict["lstm_c"] = hidden, cell

            if torch.isnan(obs_tensor).any():
                logger.error("NaN in observation! obs shape: %s", obs_tensor.shape)
            if torch.isinf(obs_tensor).any():
                logger.error("Inf in observation! obs shape: %s", obs_tensor.shape)

            logits, _ = self._net.forward_eval(obs_tensor, state_dict)

            new_state: Optional[LSTMState] = None
            if state_dict["lstm_h"] is not None and state_dict["lstm_c"] is not None:
                h = state_dict["lstm_h"].detach()
                c = state_dict["lstm_c"].detach()
                layers = self._net._rnn.num_layers * (2 if self._net._rnn.bidirectional else 1)
                new_state = LSTMState.from_tuple((h, c), layers)

            actions: list[int] = []
            for logit in logits:
                dist = torch.distributions.Categorical(logits=logit)
                actions.append(dist.sample().item())

            return np.array(actions, dtype=np.int32), new_state


class LSTMPolicy(TrainablePolicy):
    """LSTM-based policy that creates StatefulPolicy wrappers for each agent."""

    def __init__(self, env: MettaGridEnv, device: torch.device) -> None:
        super().__init__()
        self._net = LSTMPolicyNet(env).to(device)
        self._device = device
        self._action_nvec = tuple(env.single_action_space.nvec)
        self._agent_policy = LSTMAgentPolicy(self._net, device, self._action_nvec)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return StatefulAgentPolicy(self._agent_policy, agent_id)

    def load_policy_data(self, checkpoint_path: str) -> None:
        super().load_policy_data(checkpoint_path)
        self._net = self._net.to(self._device)
        self._agent_policy._net = self._net

    def is_recurrent(self) -> bool:
        return True
