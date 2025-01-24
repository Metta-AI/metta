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
    def __init__(self, agent: MettaAgent, cfg: OmegaConf, env: PettingZooPufferEnv):
        super().__init__()
        self.hidden_size = agent.decoder_out_size()
        if isinstance(env.single_action_space, pufferlib.spaces.Discrete):
            clip_scales = getattr(cfg.agent.actor, 'clip_scales', None)
            if clip_scales is not None and not isinstance(clip_scales, list):
                clip_scales = list(clip_scales)
            
            l2_norm_scales = getattr(cfg.agent.actor, 'l2_norm_scales', None)
            if l2_norm_scales is not None and not isinstance(l2_norm_scales, list):
                l2_norm_scales = list(l2_norm_scales)

            self.atn_type = make_nn_stack(
                input_size=agent.decoder_out_size(),
                hidden_sizes=list(cfg.agent.actor.hidden_sizes),
                output_size=env.single_action_space.n,
                global_clipping_value=cfg.trainer.clipping_value,
                clip_scales=clip_scales,
                l2_norm_scales=l2_norm_scales
            )
            self.atn_param = None
        elif len(env.single_action_space.nvec) == 2:
            self.atn_type = make_nn_stack(
                input_size=agent.decoder_out_size(),
                output_size=env.single_action_space.nvec[0],
                hidden_sizes=cfg.actor.hidden_sizes,
                global_clipping_value=cfg.train.clipping_value,
                clip_scales=cfg.actor.clip_scales,
                l2_norm_scales=cfg.actor.l2_norm_scales 
            )
            self.atn_param = make_nn_stack(
                input_size=agent.decoder_out_size(),
                output_size=env.single_action_space.nvec[1],
                hidden_sizes=cfg.actor.hidden_sizes,
                global_clipping_value=cfg.train.clipping_value,
                clip_scales=cfg.actor.clip_scales,
                l2_norm_scales=cfg.actor.l2_norm_scales
            )
        else:
            raise ValueError(f"Unsupported action space: {env.single_action_space}")

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
        self._agent.encode_observations(td)
        return td["encoded_obs"], None

    def decode_actions(self, flat_hidden, lookup, concat=None, e3b=None):
        value = self._agent._critic_linear(flat_hidden)
        if self.atn_param is None:
            action = self.atn_type(flat_hidden)
        else:
            action = [self.atn_type(flat_hidden), self.atn_param(flat_hidden)]

        intrinsic_reward = None
        if e3b is not None:
            phi = flat_hidden.detach()        
            intrinsic_reward = (phi.unsqueeze(1) @ e3b @ phi.unsqueeze(2))
            e3b = 0.95*e3b - (phi.unsqueeze(2) @ phi.unsqueeze(1))/(1 + intrinsic_reward)
            intrinsic_reward = intrinsic_reward.squeeze()
            intrinsic_reward = 0.1*torch.clamp(intrinsic_reward, -1, 1)

        return action, value, e3b, intrinsic_reward

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
    puffer_agent = PufferAgentWrapper(agent, cfg, env)

    if cfg.agent.core.rnn_num_layers > 0:
        puffer_agent = Recurrent(
            env, puffer_agent, input_size=cfg.agent.observation_encoder.fc.output_dim,
            hidden_size=cfg.agent.core.rnn_size,
            num_layers=cfg.agent.core.rnn_num_layers
        )
        puffer_agent = pufferlib.cleanrl.RecurrentPolicy(puffer_agent)
    else:
        puffer_agent = pufferlib.cleanrl.Policy(puffer_agent)

    puffer_agent._action_names = env.action_names()
    puffer_agent._grid_features = env._grid_env.grid_features()
    return puffer_agent.to(cfg.device)
