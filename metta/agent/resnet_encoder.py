import torch
from omegaconf import OmegaConf
from pufferlib.pytorch import layer_init
from torch import nn

from metta.agent.lib.observation_normalizer import ObservationNormalizer
from metta.agent.lib.util import name_to_activation

class _ResidualBlock(nn.Module):
    def __init__(self, depth: int, activation: str):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding="same"))
        self.conv2 = layer_init(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding="same"))
        self.activation = name_to_activation(activation)

    def forward(self, x):
        xp = self.activation(self.conv1(x))
        xp = self.activation(self.conv2(xp))
        return x + xp


class _IMPALAishBlock(nn.Module):
    def __init__(self, input_channels: int, depth: int, activation: str):
        super().__init__()
        self.conv = layer_init(nn.Conv2d(input_channels, depth, kernel_size=3, stride=1, padding="same"))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.residual1 = _ResidualBlock(depth, activation)
        self.residual2 = _ResidualBlock(depth, activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return x


def _convolution_shape(shape, kernel_size, stride):
    return tuple((x - kernel_size) // stride + 1 for x in shape)


def _product(iterable):
    prod = 1
    for i in iterable:
        prod *= i
    return prod


class IMPALAishCNN(nn.Module):
    def __init__(self, obs_space, grid_features: list[str], global_features: list[str], **cfg):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        grid_shape = obs_space[cfg.obs_key].shape

        self._output_dim = cfg.fc.output_dim
        self._obs_key = cfg.obs_key
        self.activation = name_to_activation(cfg.activation)

        self.object_normalizer = None
        if cfg.normalize_features:
            self.object_normalizer = ObservationNormalizer(grid_features)

        if isinstance(cfg.cnn_channels, int):
            channels = (grid_shape[0], cfg.cnn_channels)
        else:
            channels = (grid_shape[0], *cfg.cnn_channels)

        grid_width_height = grid_shape[1:]

        cnn_blocks = []
        for in_channels, out_channels in zip(channels, channels[1:], strict=False):
            cnn_blocks.append(_IMPALAishBlock(in_channels, out_channels, cfg.activation))
            grid_width_height = _convolution_shape(grid_width_height, 3, 2)

        self.cnn_layers = nn.Sequential(*cnn_blocks)

        cnn_flattened_size = _product(grid_width_height) * channels[-1]
        self.fc_layer = layer_init(nn.Linear(cnn_flattened_size, self._output_dim))

    def forward(self, obs_dict):
        obs = obs_dict[self._obs_key]

        if self.object_normalizer is not None:
            obs = self.object_normalizer(obs)

        x = self.cnn_layers(obs)
        x = torch.flatten(x, start_dim=1)

        x = self.activation(x)
        x = self.fc_layer(x)
        x = self.activation(x)

        return x

    def output_dim(self):
        return self._output_dim
