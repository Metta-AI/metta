from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
import gymnasium as gym
import hydra
from omegaconf import OmegaConf

from sample_factory.utils.typing import ActionSpace, ObsSpace
from pufferlib.cleanrl import sample_logits
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
    
    agent.to(cfg.device)
    return agent


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: ObsSpace,
        action_space: ActionSpace,
        grid_features: List[str],
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        # self.obs_shape = obs_shape
        # self.clip_range = cfg.clip_range
        # self.action_space = action_space
        # self.grid_features = grid_features
        # self.obs_key = cfg.observations.obs_key
        # self.obs_input_shape = obs_space[self.obs_key].shape[1:]
        # self.num_objects = obs_space[self.obs_key].shape[0]

        self.agent_attributes = {
            'obs_shape': obs_shape,
            'clip_range': cfg.clip_range,
            'action_space': action_space,
            'grid_features': grid_features,
            'obs_key': cfg.observations.obs_key,
            'obs_input_shape': obs_space[cfg.observations.obs_key].shape[1:],
            'num_objects': obs_space[cfg.observations.obs_key].shape[0]
        }
        
        self.hidden_size = cfg.components._core_.output_size # trainer/Experience uses this for e3b
        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

        self.components = nn.ModuleDict()
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], agent_attributes = self.agent_attributes)
            self.components[component_cfg] = component

        component = self.components['_value_']
        self._setup_components(component)
        component = self.components['_action_param_']
        self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, 'ready', False):
                raise RuntimeError(f"Component {name} in MettaAgent was never setup. It might not be accessible by other components.")
            
        print("Agent setup complete.")

    def _setup_components(self, component):
        if component.input_source is not None:
            self._setup_components(self.components[component.input_source])
        if component.input_source is not None:
            component.setup(self.components[component.input_source]) 
        else:
            component.setup()

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

    def l2_reg_loss(self) -> torch.Tensor:
        '''L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust it by setting l2_norm_scale in your layer config to a multiple of the global loss value or 0 to turn it off.'''
        l2_reg_loss = 0
        for component in self.components.values():
            l2_reg_loss += component.l2_reg_loss() or 0
        return torch.tensor(l2_reg_loss)
    
    def l2_init_loss(self) -> torch.Tensor:
        '''L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust it by setting l2_init_scale in your layer config to a multiple of the global loss value or 0 to turn it off.'''
        l2_init_loss = 0
        for component in self.components.values():
            l2_init_loss += component.l2_init_loss() or 0
        return torch.tensor(l2_init_loss)

    def update_l2_init_weight_copy(self):
        '''Update interval set by l2_init_weight_update_interval. 0 means no updating.'''
        for component in self.components.values():
            component.update_l2_init_weight_copy()

    def clip_weights(self):
        '''Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value effectively turns it off. Adjust it by setting clip_scale in your layer config to a multiple of the global loss value or 0 to turn it off.'''
        if self.agent_attributes['clip_range'] > 0:
            for component in self.components.values():
                component.clip_weights()

    def compute_effective_rank(self, delta: float = 0.01) -> List[dict]:
        '''Effective rank computation is off by default. Set effective_rank to True in the config to turn it on for a given layer.'''
        effective_ranks = []
        for component in self.components.values():
            rank = component.effective_rank(delta)
            if rank is not None:
                effective_ranks.append(rank)
        print(f"Effective ranks: {effective_ranks}")
        return effective_ranks
