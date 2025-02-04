from __future__ import annotations

from typing import List

import hydra
import omegaconf
from omegaconf import OmegaConf
from sample_factory.model.action_parameterization import ActionParameterizationDefault
from sample_factory.model.core import ModelCoreRNN
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import Tensor
from sample_factory.algo.utils.action_distributions import sample_actions_log_probs

from tensordict import TensorDict
from torch import Tensor, nn
import torch
from agent.agent_interface import MettaAgentInterface
# from agent.lib.util import MettaComponent, MettaNet, _MettaHelperComponent
from agent.lib.observation_normalizer import ObservationNormalizer
from agent.feature_encoder import FeatureListNormalizer
# from agent.components import Composer, Layer
import omegaconf

class MettaAgent(nn.Module, MettaAgentInterface):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        grid_features: List[str],
        global_features: List[str],
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.observation_space = obs_space
        self.action_space = action_space
        self.grid_features = grid_features
        # self.global_features = global_features
        self._obs_key = cfg.observations.obs_key

        self.num_objects = obs_space[self._obs_key].shape[0]

        # self.obs_cfg = cfg.obs
        # cfg.obs.name = 'obs'
        # cfg.obs.input_source = 'obs'
        # cfg.obs.output_size = self._num_objects

        # self.obs = MettaLayer(self, MettaAgent=self, cfg=cfg.obs)
        # self.normalizer = FeatureListNormalizer(grid_features, obs_space[self._obs_key].shape[1:])
        # self.object_normalizer = ObservationNormalizer(self, grid_features)

        self.components = {}

        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], MettaAgent=self)
            self.components[component_cfg] = component

        for component in self.components.values():
            # check if custom components and Obs and Recurrent need these.
            component.set_input_source_size()
            component.initialize_layer()

        self.obs_encoder = MettaNet(self.components, '_encoded_obs_')
        self.atn_param = MettaNet(self.components, '_atn_param_')
        self.critic = MettaNet(self.components, '_value_')

    #def weight helper functions

class MettaLayer(nn.Module):
    def __init__(self, MettaAgent, **cfg):
        cfg = OmegaConf.create(cfg)
        super().__init__()
        self.MettaAgent = MettaAgent
        self.cfg = cfg
        self.name = cfg.name
        self.input_source = cfg.input_source
        self.output_size = cfg.get('output_size', None)
        self.layer_type = cfg.get('layer_type', 'Linear')
        self.nonlinearity = cfg.get('nonlinearity', 'ReLU')

    def set_input_source_size(self):
        if self.input_source == '_obs_':
            self.input_size = self.MettaAgent.num_objects
        elif self.input_source == '_core_':
            self.input_size = self.output_size
        else:
            if isinstance(self.input_source, omegaconf.listconfig.ListConfig):
                self.input_source = list(self.input_source)
                self.input_size = sum(self.MettaAgent.components[src].output_size for src in self.input_source)
            else:
                self.input_size = self.MettaAgent.components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

    def initialize_layer(self):
        if self.layer_type == 'Linear':
            self.layer = getattr(nn, self.layer_type)(self.input_size, self.output_size)
        elif self.layer_type == 'Conv2d':
            # input size is the number of objects
            # output size is the number of channels
            self.layer = getattr(nn, self.layer_type)(self.input_size, self.output_size, self.cfg.kernel_size, self.cfg.stride)
        elif self.layer_type == 'Dropout':
            self.layer = getattr(nn, self.layer_type)(self.cfg.dropout_prob)
        elif self.layer_type == 'Identity':
            self.nonlinearity = None
            self.layer = nn.Identity()
        elif self.layer_type == 'Flatten':
            self.layer = nn.Flatten()
        else:
            raise ValueError(f"Layer type {self.layer_type} not supported")
        # add resnet, etc.

    def forward(self, td: TensorDict):
        # Check if the output is already computed to avoid redundant compute
        if self.name in td:
            return td

        if isinstance(self.input_source, omegaconf.listconfig.ListConfig):
            self.input_source = list(self.input_source)
            #concatenate the inputs
            x = torch.cat([self.MettaAgent.components[src](td) for src in self.input_source], dim=-1)
        else:
            x = self.MettaAgent.components[self.input_source](td)
        x = self.layer(x)

        if self.nonlinearity:
            x = getattr(nn, self.nonlinearity)(x)
        td[self.name] = x
        return td

class MettaNet(nn.Module):
    def __init__(self, components, output_name):
        super().__init__()
        self.components = components  # list of components
        self.output_name = output_name
        
    def compute_component(self, name, td: dict):
        if name in td:
            return td[name]
        if name == '_obs_':
            return td["obs"]
        if name in ('_core_'):
            return td["core_output"]
        comp = self.components[name]
        # For multi-input case, ensure we have a list
        input_sources = comp.input_source if isinstance(comp.input_source, list) else [comp.input_source]
        inputs = []
        for src in input_sources:
            if src == '_obs_':
                inputs.append(td["obs"])
            elif src in ('_core_'):
                inputs.append(td["core_output"])
            else:
                inputs.append(self.compute_component(src, td))
        # Merge inputs accordingly (here we use concatenation)
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        # Here we assume that comp.layer and comp.nonlinearity have been set up by initialize_layer()
        out = comp.layer(x)
        if comp.nonlinearity is not None:
            out = getattr(nn, comp.nonlinearity)(out)
        td[comp.name] = out
        return out

    def forward(self, td: dict):
        # Recursively compute the output for self.output_name
        output = self.compute_component(self.output_name, td)
        return output