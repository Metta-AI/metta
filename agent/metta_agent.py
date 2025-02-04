from __future__ import annotations

from typing import List

import hydra
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
        self.object_normalizer = ObservationNormalizer(self, grid_features)

        self.components = []
        component_cfgs = {cfg.components}

        for component_cfg in component_cfgs.keys():
            component = hydra.utils.instantiate(component_cfgs[component_cfg], MettaAgent=self)
            self.components.append(component)

        for component in self.components:
            # check if custom components and Obs and Recurrent need these.
            component.get_input_source_size()
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
        self.name = cfg.name
        self.input_source = cfg.input_source
        self.output_size = cfg.output_size if 'output_size' in cfg else None
        self.layer_type = 'Linear' if not cfg.layer_type else cfg.layer_type
        self.nonlinearity = 'ReLU' if not cfg.nonlinearity else cfg.nonlinearity

    def set_input_source_size(self):
        if self.input_source == '_ obs_':
            self.input_size = self.MettaAgent.num_objects
        elif self.input_source == '_core_':
            self.input_size = self.output_size
        else:
            self.input_size = self.MettaAgent.components[self.input_source].get_out_size()

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

    def forward(self, td: TensorDict):
        # Check if the output is already computed to avoid redundant compute
        if self.name in td:
            return td

        x = self.MettaAgent.components[self.input_source](td)
        x = self.layer(x)

        if self.nonlinearity:
            x = getattr(nn, self.nonlinearity)(x)
        td[self.name] = x
        return td

class MettaNet(nn.Module):
    def __init__(self, components, output_name):
        super().__init__()
        self.components = components
        self.output_name = output_name
        
        for comp in self.components:
            if comp.name == self.output_name:
                current = comp
                break
        
        chain = []
        while True:
            chain.append(current)
            if current.input_source is None or current.input_source == '_obs_' or current.input_source == '_core_':
                break
            
            higher_comp = None
            for c in self.components:
                if c.name == current.input_source:
                    higher_comp = c
                    break
            
            current = higher_comp
        chain.reverse()
        self._forward_path = nn.ModuleList(chain)

    def forward(self, td: TensorDict):
        for comp in self._forward_path:
            if comp.name == '_obs_':
                td = comp(td["_obs_"])
            elif comp.name == '_core_':
                td = comp(td["_core_"])
            else:
                td = comp(td)
        return td