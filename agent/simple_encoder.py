import torch
from omegaconf import OmegaConf
from pufferlib.pytorch import layer_init
from torch import nn


class SimpleConvAgent(nn.Module):

    def __init__(self,
                obs_space,
                fc_cfg: OmegaConf, **cfg):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self._obs_key = cfg.obs_key
        self._num_objects = obs_space[self._obs_key].shape[0]
        self._cnn_channels = cfg.cnn_channels
        self._output_dim = fc_cfg.output_dim

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self._num_objects, self._cnn_channels, 5, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(self._cnn_channels, self._cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self._cnn_channels, self._output_dim)),
            nn.ReLU(),
        )
        obs_norms = [1] * self._num_objects
        if cfg["normalize_features"]:
            obs_norms = [
                1, # 'agent',
                1, # 'agent:hp',
                1, # 'agent:frozen',
                255, # 'agent:energy',
                1, # 'agent:orientation',
                1, # 'agent:shield',
                5, # 'agent:inv:r1',
                5, # 'agent:inv:r2',
                5, # 'agent:inv:r3',
                1, # 'wall',
                10, # 'wall:hp',
                1, # 'generator',
                30, # 'generator:hp',
                30, # 'generator:r1',
                1, # 'generator:ready',
                1, # 'converter',
                30, # 'converter:hp',
                5, # 'converter:input_resource',
                5, # 'converter:output_resource',
                100, # 'converter:output_energy',
                1, # 'converter:ready',
                1, # 'altar',
                30, # 'altar:hp',
                1, # 'altar:ready'
            ]


        self._obs_norm = torch.tensor(obs_norms, dtype=torch.float32)
        self._obs_norm = self._obs_norm.view(1, self._num_objects, 1, 1)

    def forward(self, obs_dict):
        self._obs_norm = self._obs_norm.to(obs_dict[self._obs_key].device)
        obs = obs_dict[self._obs_key] / self._obs_norm
        return self.network(obs)

    def output_dim(self):
        return self._output_dim