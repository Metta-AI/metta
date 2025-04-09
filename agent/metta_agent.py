import logging
from typing import List, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from pufferlib.environment import PufferEnv
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from agent.util.distribution_utils import sample_logits

logger = logging.getLogger("metta_agent")

def make_policy(env: PufferEnv, cfg: OmegaConf):
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=[ 0 ],
            dtype=np.int32)
    })
    return hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.single_observation_space.shape,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        device=cfg.device,
        _recursive_=False)

class DistributedMettaAgent(DistributedDataParallel):
    def __init__(self, agent, device):
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        action_space: gym.spaces.Space,
        grid_features: List[str],
        device: str,
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        self.is_continuous = False
        self.hidden_size = cfg.components._core_.output_size
        self.core_num_layers = cfg.components._core_.nn_params.num_layers
        self.clip_range = cfg.clip_range

        agent_attributes = {
            'obs_shape': obs_shape,
            'clip_range': self.clip_range,
            'action_space': action_space,
            'grid_features': grid_features,
            'obs_key': cfg.observations.obs_key,
            'obs_input_shape': obs_space[cfg.observations.obs_key].shape[1:],
            'num_objects': obs_space[cfg.observations.obs_key].shape[2], # this is hardcoded for channel # at end of tuple
            'hidden_size': self.hidden_size,
            'core_num_layers': self.core_num_layers,
            'is_continuous': self.is_continuous
        }

        agent_attributes['action_type_size'] = action_space.nvec[0]
        agent_attributes['action_param_size'] = action_space.nvec[1]

        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

        self.components = nn.ModuleDict()
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], **agent_attributes)
            self.components[component_cfg] = component

        component = self.components['_value_']
        self._setup_components(component)
        component = self.components['_action_type_']
        self._setup_components(component)
        component = self.components['_action_param_']
        self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, 'ready', False):
                raise RuntimeError(f"Component {name} in MettaAgent was never setup. It might not be accessible by other components.")

        self.components = self.components.to(device)

        self._total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")

    def _setup_components(self, component):
        if component._input_source is not None:
            if isinstance(component._input_source, str):
                self._setup_components(self.components[component._input_source])
            elif isinstance(component._input_source, list):
                for input_source in component._input_source:
                    self._setup_components(self.components[input_source])

        if component._input_source is not None:
            if isinstance(component._input_source, str):
                component.setup(self.components[component._input_source])
            elif isinstance(component._input_source, list):
                input_source_components = {}
                for input_source in component._input_source:
                    input_source_components[input_source] = self.components[input_source]
                component.setup(input_source_components)
        else:
            component.setup()

    @property
    def lstm(self):
        return self.components["_core_"]._net

    @property
    def total_params(self):
        return self._total_params

    def get_value(self, x, state=None):
        td = TensorDict({"x": x, "state": state})
        self.components["_value_"](td)
        return None, td["_value_"], None

    def forward(self, obs, state: TensorDict):
        # This is a terrible spot for this piece of code, but
        # I don't know where the encoder is to put it there.
        # You want to concat this encoding with the pre-lstm state
        # if 'diayn_z' in td:
        #     td['diayn_embed'] = self.diayn_encoder(td['diayn_z'])

        # xcxc fix this?

        self.components["_value_"](state)
        self.components["_action_type_"](state)
        self.components["_action_param_"](state)

        logits = [state["_action_type_"], state["_action_param_"]]
        value = state["_value_"]

        return logits, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def l2_reg_loss(self) -> torch.Tensor:
        '''L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.'''
        l2_reg_loss = 0
        for component in self.components.values():
            l2_reg_loss += component.l2_reg_loss() or 0
        return torch.tensor(l2_reg_loss)

    def l2_init_loss(self) -> torch.Tensor:
        '''L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.'''
        l2_init_loss = 0
        for component in self.components.values():
            l2_init_loss += component.l2_init_loss() or 0
        return torch.tensor(l2_init_loss)

    def update_l2_init_weight_copy(self):
        '''Update interval set by l2_init_weight_update_interval. 0 means no updating.'''
        for component in self.components.values():
            component.update_l2_init_weight_copy()

    def clip_weights(self):
        '''Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global loss value or 0 to turn it off.'''
        if self.clip_range > 0:
            for component in self.components.values():
                component.clip_weights()

    def compute_effective_rank(self, delta: float = 0.01) -> List[dict]:
        '''Effective rank computation is off by default. Set effective_rank to True in the config to turn it on for a given component.'''
        effective_ranks = []
        for component in self.components.values():
            rank = component.compute_effective_rank(delta)
            if rank is not None:
                effective_ranks.append(rank)
        return effective_ranks
