import gymnasium as gym
import hydra
import numpy as np
import pufferlib
import sys
import pufferlib.cleanrl
sys.modules['pufferlib.frameworks.cleanrl'] = pufferlib.cleanrl
import pufferlib.models
import pufferlib.pytorch
import torch
from omegaconf import OmegaConf
from pufferlib.emulation import PettingZooPufferEnv
from pufferlib.environment import PufferEnv
from tensordict import TensorDict
from torch import nn
from typing import List
from agent.lib.util import make_nn_stack

from agent.metta_agent import MettaAgent


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class PufferAgentWrapper(nn.Module):
    def __init__(self, agent: MettaAgent, env: PettingZooPufferEnv):
        super().__init__()
        # self.hidden_size = agent.decoder_out_size()
        # if isinstance(env.single_action_space, pufferlib.spaces.Discrete):
        #     self.atn_type = make_nn_stack(
        #         input_size=agent.decoder_out_size(),
        #         hidden_sizes=actor_hidden_sizes,
        #         output_size=env.single_action_space.n
        #     )
        #     self.atn_param = None
        # elif len(env.single_action_space.nvec) == 2:
        #     self.atn_type = make_nn_stack(
        #         input_size=agent.decoder_out_size(),
        #         output_size=env.single_action_space.nvec[0],
        #         hidden_sizes=actor_hidden_sizes
        #     )
        #     self.atn_param = make_nn_stack(
        #         input_size=agent.decoder_out_size(),
        #         output_size=env.single_action_space.nvec[1],
        #         hidden_sizes=actor_hidden_sizes
        #     )
        # else:
        #     raise ValueError(f"Unsupported action space: {env.single_action_space}")

        self._agent = agent
        print(self)

    def forward(self, obs, e3b=None):
        x, _ = self.encode_observations(obs)
        return self.decode_actions(x, None, e3b=e3b)

    def encode_observations(self, flat_obs):
        obs = {
            "grid_obs": flat_obs.float(),
            "global_vars": torch.zeros(flat_obs.shape[0], dtype=torch.float32).to(flat_obs.device)
        }
        td = TensorDict({"obs": obs})
        self._agent.obs_encoder(td)
        return td["encoded_obs"], td

    def decode_actions(self, flat_hidden, lookup, concat=None, e3b=None):
        flat_obs = lookup
        obs = {
            "grid_obs": flat_obs.float(),   
            "global_vars": torch.zeros(flat_obs.shape[0], dtype=torch.float32).to(flat_obs.device)
        }
        td = TensorDict({"obs": obs})
        td["core_output"] = flat_hidden

        value = self._agent.critic(td)
        action = self._agent.atn_param(td)

        b = None
        if e3b is not None:
            phi = flat_hidden.detach()        
            u = phi.unsqueeze(1) @ e3b
            b = u @ phi.unsqueeze(2)
            e3b = 0.99*e3b - (u.mT @ u) / (1 + b)
            b = b.squeeze()

        return action, value, e3b, b

def make_policy(env: PufferEnv, cfg: OmegaConf):
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=[ 0 ],
            dtype=np.int32)
    })
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_space,
        env.single_action_space,
        env.grid_features,
        env.global_features,
        _recursive_=False)
    puffer_agent = PufferAgentWrapper(agent, env)

    if cfg.agent.components.core_helper.rnn_num_layers > 0:
        puffer_agent = Recurrent(
            env, puffer_agent, input_size=cfg.agent.components._encoded_obs_.output_size,
            hidden_size=cfg.agent.components.core_helper.rnn_size,
            num_layers=cfg.agent.components.core_helper.rnn_num_layers
        )
        puffer_agent = pufferlib.cleanrl.RecurrentPolicy(puffer_agent)
    else:
        puffer_agent = pufferlib.cleanrl.Policy(puffer_agent)

    puffer_agent._action_names = env.action_names()
    puffer_agent._grid_features = env._grid_env.grid_features()
    return puffer_agent.to(cfg.device)
