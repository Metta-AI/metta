from __future__ import annotations

from typing import Dict

from torch import Tensor

from tensordict import TensorDict
from torch import Tensor


class MettaAgentInterface():
    def encode_observations(self, obs_dict: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        raise NotImplementedError()

    def decode_state(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        raise NotImplementedError()

    def forward(self, obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        raise NotImplementedError()

    def aux_loss(self, obs_dict, rnn_states):
        raise NotImplementedError()
