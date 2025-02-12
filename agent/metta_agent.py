from __future__ import annotations

from typing import List, Tuple

import hydra
from omegaconf import OmegaConf
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import nn
import torch
from tensordict import TensorDict
#we could include sameple_logits in this file
from pufferlib.cleanrl import sample_logits

import gymnasium as gym
import numpy as np
from pufferlib.environment import PufferEnv

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
        obs_shape=env.single_observation_space.shape,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        _recursive_=False)
    
    # delete the below?
    # agent._action_names = env.action_names()
    # agent._grid_features = env._grid_env.grid_features()
    
    # agent.to(cfg.device)
    return agent


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: ObsSpace,
        action_space: ActionSpace,
        grid_features: List[str],
        global_features: List[str],
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.clip_range = cfg.clip_range
        self.action_space = action_space
        self.grid_features = grid_features
        self.obs_key = cfg.observations.obs_key
        self.obs_input_shape = obs_space[self.obs_key].shape[1:]
        self.num_objects = obs_space[self.obs_key].shape[0]
        self.hidden_size = cfg.components._core_.output_size # trainer/Experience uses this for e3b
        
        # delete the below?
        self.e3b = torch.eye(self.hidden_size)

        # are these needed?
        # self.observation_space = obs_space
        # self.global_features = global_features
        

        self.components = {}
        
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], metta_agent=self)
            self.components[component_cfg] = component

        self.components['_action_param_'].setup_layer()
        self.components['_value_'].setup_layer()
        self.components = nn.ModuleDict(self.components)
        print("Agent setup complete.")

    @property
    def lstm(self):
        return self.components["_core_"].layer

    def get_value(self, x, state=None):
        td = TensorDict({"x": x, "state": state})
        self.components["_value_"](td)
        return None, td["_value_"], None

    def get_action_and_value(self, x, state=None, action=None, e3b=None):

        td = TensorDict({"x": x, "state": state})

        self.components["_value_"](td)
        self.components["_action_param_"](td)

        logits = td["_action_param_"]
        value = td["_value_"]
        state = td["state"] 

        e3b, intrinsic_reward = self._e3b_update(td["_core_"].detach(), e3b)

        # what is action?
        # check if self.is_continuous means continuous action space, set to False below
        action, logprob, entropy = sample_logits(logits, action, False)
        return action, logprob, entropy, value, state, e3b, intrinsic_reward
    
    def forward(self, x, state=None, action=None, e3b=None):
        return self.get_action_and_value(x, state, action, e3b)
    
    def _e3b_update(self, phi, e3b):
        intrinsic_reward = None
        if e3b is not None:
            u = phi.unsqueeze(1) @ e3b
            intrinsic_reward = u @ phi.unsqueeze(2)
            e3b = 0.99*e3b - (u.mT @ u) / (1 + intrinsic_reward)
            intrinsic_reward = intrinsic_reward.squeeze()
        return e3b, intrinsic_reward


    def clip_weights(self):
        for component in self.components.values():
            component.clip_weights()

    def l2_reg_loss(self) -> torch.Tensor:
        l2_reg_loss = 0
        for component in self.components.values():
            l2_reg_loss += component.l2_reg_loss() or 0
        return torch.tensor(l2_reg_loss)
    
    def l2_init_loss(self) -> torch.Tensor:
        l2_init_loss = 0
        for component in self.components.values():
            l2_init_loss += component.l2_init_loss() or 0
        return torch.tensor(l2_init_loss)

    def update_l2_init_weight_copy(self):
        for component in self.components.values():
            component.update_l2_init_weight_copy()

    def effective_rank(self, delta: float = 0.01) -> List[dict]:
        effective_ranks = []
        for component in self.components.values():
            effective_ranks.append(component.effective_rank(delta))
        return effective_ranks
