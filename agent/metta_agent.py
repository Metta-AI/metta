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
        self.obs_key = cfg.observations.obs_key

        self.num_objects = obs_space[self.obs_key].shape[0]

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

        self.components['_encoded_obs_'].set_input_size_and_initialize_layer()
        self.components['_action_param_'].set_input_size_and_initialize_layer()
        self.components['_value_'].set_input_size_and_initialize_layer()

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

    def set_input_size_and_initialize_layer(self):
        if self.input_source == '_obs_':
            self.input_size = self.MettaAgent.num_objects
        elif self.input_source == '_core_':
            self.input_size = self.output_size
        else:
            if isinstance(self.input_source, omegaconf.listconfig.ListConfig):
                self.input_source = list(self.input_source)
                for src in self.input_source:
                    self.MettaAgent.components[src].set_input_size_and_initialize_layer()

                self.input_size = sum(self.MettaAgent.components[src].output_size for src in self.input_source)
            else:
                self.MettaAgent.components[self.input_source].set_input_size_and_initialize_layer()
                self.input_size = self.MettaAgent.components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

        # --- initialize your layer ---
        # can this be simpler?
        if self.layer_type == 'Linear':
            self.layer = getattr(nn, self.layer_type)(self.input_size, self.output_size)
            if self.nonlinearity:
                self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())
        elif self.layer_type == 'Conv2d':
            # input size is the number of objects
            # output size is the number of channels
            self.layer = getattr(nn, self.layer_type)(self.input_size, self.output_size, self.cfg.kernel_size, self.cfg.stride)
            if self.nonlinearity:
                self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())
        elif self.layer_type == 'Dropout':
            self.layer = getattr(nn, self.layer_type)(self.cfg.dropout_prob)
            #delete nonlinearity here
            self.nonlinearity = None
        elif self.layer_type == 'BatchNorm2d':
            self.layer = getattr(nn, self.layer_type)(self.input_size)
            self.nonlinearity = None
        elif self.layer_type == 'Identity':
            self.nonlinearity = None
            self.layer = nn.Identity()
        elif self.layer_type == 'Flatten':
            self.layer = nn.Flatten()
            self.nonlinearity = None
        else:
            raise ValueError(f"Layer type {self.layer_type} not supported")
        # add resnet, etc.

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if self.input_source == '_obs_':
            td[self.name] = td["obs"][self.MettaAgent.obs_key]
        elif self.input_source == '_core_':
            td[self.name] = td["core_output"]
        else:
# need to think about cat vs add vs subtract
            if isinstance(self.input_source, list):
                for src in self.input_source:
                   self.MettaAgent.components[src].forward(td) 
            else:
                self.MettaAgent.components[self.input_source].forward(td)

            # delete this after testing
            print(f"layer name: {self.name}")
            
            if not isinstance(self.input_source, list):
                print(f"input td[self.input_source].shape: {td[self.input_source].shape}") 
            else:
                for src in self.input_source:
                    print(f"input td[{src}].shape: {td[src].shape}")
 

            if isinstance(self.input_source, list):
                inputs = [td[src] for src in self.input_source]
                x = torch.cat(inputs, dim=-1)
                td[self.name] = self.layer(x)
            else:
                td[self.name] = self.layer(td[self.input_source])

        return td

class MettaLayerBase(nn.Module):
    def __init__(self, MettaAgent, **cfg):
        super().__init__()
        self.MettaAgent = MettaAgent
        self.cfg = cfg
        #required attributes
        self.name = None
        self.input_source = None
        self.output_size = None

    def set_input_size_and_initialize_layer(self):
        '''
        Recursively set the input size for the component above your layer.
        This is necessary unless you are a top layer, in which case, you can skip this.
        self.MettaAgent.components[self.input_source].set_input_source_size()
        
        Set your input size to be the output size of the layer above you or otherwise ensure that this is the case.
        self.input_size = self.MettaAgent.components[self.input_source].output_size

        With your own input and output sizes set, initialize your layer, if necessary.
        self.layer = ...

        '''
        raise NotImplementedError(f"The method set_input_source_size() is not implemented yet for object {self.__class__.__name__}.")

    def forward(self, td: TensorDict):
        '''
        First, ensure we're not recomputing in case your layer is already computed.
        if self.name in td:
            return td[self.name]

        First, recursively compute the input to the layer above this layer.
        Skip this if you are a top layer.
        x = self.MettaAgent.components[self.input_source].forward(td)

        Compute this layer's output.
        x = self.layer(x)

        Write your layer's name on your output so the next layer can find it.
        td[self.name] = x

        Pass the full td back.
        return td
        '''
        raise NotImplementedError(f"The method forward() is not implemented yet for object {self.__class__.__name__}.")

# class MettaNet(nn.Module):
#     def __init__(self, components, output_name):
#         super().__init__()
#         self.components = components  
#         self.output_name = output_name
        
#     def compute_component(self, name, td: dict):
#         if name in td:
#             return td[name]
#         if name == '_obs_':
#             return td["obs"]
#         if name in ('_core_'):
#             return td["core_output"]
#         comp = self.components[name]

#         input_sources = comp.input_source if isinstance(comp.input_source, list) else [comp.input_source]
#         inputs = []
#         for src in input_sources:
#             if src == '_obs_':
#                 inputs.append(td["obs"])
#             elif src in ('_core_'):
#                 inputs.append(td["core_output"])
#             else:
#                 inputs.append(self.compute_component(src, td))

#         x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

#         out = comp.layer(x)
#         if comp.nonlinearity is not None:
#             out = getattr(nn, comp.nonlinearity)(out)
#         td[comp.name] = out
#         return out

#     def forward(self, td: dict):
#         output = self.compute_component(self.output_name, td)
#         return output