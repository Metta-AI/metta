from typing import List

import torch
from omegaconf import OmegaConf
from pufferlib.pytorch import layer_init
from torch import nn

from agent.feature_encoder import FeatureListNormalizer
from agent.lib.util import make_nn_stack

# ##ObservationNormalization
# These are approximate maximum values for each feature. Ideally they would be defined closer to their source,
# but here we are. If you add / remove a feature, you should add / remove the corresponding normalization.
OBS_NORMALIZATIONS = {
    'agent': 1,
    'agent:hp': 1,
    'agent:frozen': 1,
    'agent:energy': 255,
    'agent:orientation': 1,
    'agent:shield': 1,
    'agent:inv:r1': 5,
    'agent:inv:r2': 5,
    'agent:inv:r3': 5,
    'wall': 1,
    'wall:hp': 10,
    'generator': 1,
    'generator:hp': 30,
    'generator:r1': 30,
    'generator:ready': 1,
    'converter': 1,
    'converter:hp': 30,
    'converter:input_resource': 5,
    'converter:output_resource': 5,
    'converter:output_energy': 100,
    'converter:ready': 1,
    'altar': 1,
    'altar:hp': 30,
    'altar:ready': 1,
    'last_action': 10,
    'last_action_argument': 10,
    'agent:kinship': 10,
}

class SimpleConvAgent(nn.Module):

    def __init__(self,
                 obs_space,
                 grid_features: List[str],
                 global_features: List[str],
                 **cfg):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self._obs_key = cfg.obs_key
        self._num_objects = obs_space[self._obs_key].shape[0]
        self._cnn_channels = cfg.cnn_channels
        self._output_dim = fc_cfg.output_dim
        self._hidden_sizes = fc_cfg.hidden_sizes

        if self._hidden_sizes is None:
            self._hidden_sizes = [] 
        else:
            self._hidden_sizes = [self._hidden_sizes]
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self._num_objects, self._cnn_channels, 5, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(self._cnn_channels, self._cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            make_nn_stack(self._cnn_channels, self._output_dim, self._hidden_sizes),
            nn.ReLU(),
        )

        self._normalizer = None
        if cfg.auto_normalize:
            self._normalizer = FeatureListNormalizer(
                grid_features, obs_space[self._obs_key].shape[1:])

        self._obs_norm = None
        if cfg.normalize_features:
            # #ObservationNormalization
            obs_norms = [OBS_NORMALIZATIONS[k] for k in grid_features]
            self._obs_norm = torch.tensor(obs_norms, dtype=torch.float32)
            self._obs_norm = self._obs_norm.view(1, self._num_objects, 1, 1)

    def forward(self, obs_dict):
        obs = obs_dict[self._obs_key]

        if self._obs_norm is not None:
            self._obs_norm = self._obs_norm.to(obs.device)
            obs = obs / self._obs_norm

        if self._normalizer:
            self._normalizer(obs)

        return self.network(obs)

    def output_dim(self):
        return self._output_dim
