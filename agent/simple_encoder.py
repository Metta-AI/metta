import torch
from omegaconf import OmegaConf
from agent.lib.observation_normalizer import ObservationNormalizer
from pufferlib.pytorch import layer_init
from torch import nn

from agent.feature_encoder import FeatureListNormalizer


class SimpleConvAgent(nn.Module):

    def __init__(self,
                 obs_space,
                 grid_features: list[str],
                 global_features: list[str],
                 **cfg):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self._obs_key = cfg.obs_key
        self._num_objects = obs_space[self._obs_key].shape[0]
        self._cnn_channels = cfg.cnn_channels
        self._output_dim = cfg.fc.output_dim

        layers = [
            layer_init(nn.Conv2d(self._num_objects, self._cnn_channels, 5, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(self._cnn_channels, self._cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self._cnn_channels, self._output_dim)),
            nn.ReLU()
        ]
        for _ in range(cfg.fc.layers - 1):
            layers.append(layer_init(nn.Linear(self._output_dim, self._output_dim)))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

        self._normalizer = None
        if cfg.auto_normalize:
            self._normalizer = FeatureListNormalizer(
                grid_features, obs_space[self._obs_key].shape[1:])

        self.object_normalizer = None
        if cfg.normalize_features:
            self.object_normalizer = ObservationNormalizer(grid_features)


    def forward(self, obs_dict):
        obs = obs_dict[self._obs_key]

        if self.object_normalizer is not None:
            obs = self.object_normalizer(obs)

        if self._normalizer:
            self._normalizer(obs)

        return self.network(obs)

    def output_dim(self):
        return self._output_dim
    
    def get_out_size(self):
        return self._output_dim
    
    def get_in_size(self):
        return self._num_objects
