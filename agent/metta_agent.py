from __future__ import annotations

from typing import List

import hydra
from omegaconf import OmegaConf
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import nn
from agent.agent_interface import MettaAgentInterface
from agent.lib.observation_normalizer import ObservationNormalizer
from agent.feature_encoder import FeatureListNormalizer
import torch


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
        # self.observation_space = obs_space
        self.action_space = action_space
        self.grid_features = grid_features
        # self.global_features = global_features
        # self.obs_key = cfg.observations.obs_key
        self.num_objects = obs_space[self.cfg.observations.obs_key].shape[0]
        self.clip_range = cfg.clip_range


        self.components = {}
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], metta_agent=self)
            self.components[component_cfg] = component

        self.components['_obs_'].output_size = self.num_objects

        self.components['_encoded_obs_'].setup_layer()
        self.components['_action_param_'].setup_layer()
        self.components['_value_'].setup_layer()

    def clip_weights(self):
        for component in self.components.values():
            component.clip_weights()
# ditch the check attribute
    def get_l2_reg_loss(self) -> torch.Tensor:
        l2_reg_loss = torch.tensor(0.0, device=self.weights.device)
        for component in self.components.values():
            if hasattr(component, 'get_l2_reg_loss'):
                l2_reg_loss += component.get_l2_reg_loss()
        return l2_reg_loss
    
    def get_l2_init_loss(self) -> torch.Tensor:
        l2_init_loss = torch.tensor(0.0, device=self.weights.device)
        for component in self.components.values():
            if hasattr(component, 'get_l2_init_loss'):
                l2_init_loss += component.get_l2_init_loss()
        return l2_init_loss

    def update_l2_init_weight_copy(self):
        for component in self.components.values():
            if hasattr(component, 'update_l2_init_weight_copy'):
                component.update_l2_init_weight_copy()

    def get_effective_rank(self) -> torch.Tensor:
        effective_ranks = []
        for component in self.components.values():
            if hasattr(component, 'get_effective_rank'):
                effective_ranks.append(component.get_effective_rank())
        return effective_ranks
