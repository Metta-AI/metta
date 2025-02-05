from __future__ import annotations

from typing import List

import hydra
from omegaconf import OmegaConf
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import nn
from agent.agent_interface import MettaAgentInterface
from agent.lib.observation_normalizer import ObservationNormalizer
from agent.feature_encoder import FeatureListNormalizer


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

