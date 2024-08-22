

from __future__ import annotations

from sample_factory.algo.utils.action_distributions import sample_actions_log_probs
from sample_factory.model.action_parameterization import ActionParameterizationDefault
from sample_factory.model.core import ModelCoreRNN
from tensordict import TensorDict


from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import model_device
from sample_factory.utils.attr_dict import AttrDict
import torch

from agent.metta_agent import MettaAgent

from typing import Dict

from torch import Tensor


class SampleFactoryAgentWrapper(ActorCritic):
    def __init__(self, agent: MettaAgent):
        super().__init__(agent.observation_space, agent.action_space, AttrDict({
            "normalize_returns": True,
            "normalize_input": False,
            "obs_subtract_mean": 0.0,
            "obs_scale": 1.0,

        }))
        self.agent = agent
        self._core = ModelCoreRNN(agent.cfg.core, agent.cfg.fc.output_dim)

        self._action_parameterization = self.get_action_parameterization()
        self._last_action_distribution = None

    def forward_head(self, obs_dict: Dict[str, Tensor]) -> Tensor:
        td = TensorDict({"obs": obs_dict})
        self.agent.encode_observations(td)
        return td["encoded_obs"]

    def forward_core(self, head_output, rnn_states):
        core_output, new_rnn_states = self._core(head_output, rnn_states)
        return core_output, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        td = TensorDict({"core_output": core_output})
        self.agent.decode_state(td)
        if values_only:
            return td

        action_distribution_params, self._last_action_distribution = self._action_parameterization(td["state"])
        # `action_logits` is not the best name here, better would be "action distribution parameters"
        td["action_logits"] = action_distribution_params
        if not sample_actions:
            return td

        # for non-trivial action spaces it is faster to do these together
        actions, td["log_prob_actions"] = sample_actions_log_probs(self._last_action_distribution)
        assert actions.dim() == 2  # TODO: remove this once we test everything
        td["actions"] = actions.squeeze(dim=1)

        return td

    def forward(self, obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        head_output = self.forward_head(obs_dict)
        core_output, new_rnn_states = self.forward_core(head_output, rnn_states)
        td = self.forward_tail(core_output, values_only, sample_actions=True)
        td["new_rnn_states"] = new_rnn_states
        return td

    def action_distribution(self):
        return self._last_action_distribution

    def aux_loss(self, obs_dict, rnn_states):
        return self.agent.aux_loss(obs_dict, rnn_states)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32

    def get_action_parameterization(self):
        return ActionParameterizationDefault(
            {}, self.agent.decoder_out_size(), self.agent.action_space)

