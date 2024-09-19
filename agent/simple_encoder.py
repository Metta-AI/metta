from torch import nn
from .lib.util import layer_init
from omegaconf import OmegaConf

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

    def forward(self, obs_dict):
        obs_transformed = obs_dict[self._obs_key]
        return self.network(obs_transformed)

    def output_dim(self):
        return self._output_dim
