from amago.nets.cnn import CNN
import gin

import torch
import torch.nn as nn

from agent.lib.observation_normalizer import ObservationNormalizer


@gin.configurable
class MettaAmagoCNN(CNN):
    def __init__(
        self,
        img_shape: tuple[int],
        channels_first: bool,
        activation: str,
        channels=[32, 64],
        *,
        grid_features: list[str] = [],
    ):
        super().__init__(
            img_shape, channels_first=channels_first, activation=activation
        )
        C = img_shape[0] if self.channels_first else img_shape[-1]
        self.conv1 = nn.Conv2d(C, channels[0], kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1)

        self.object_normalizer = ObservationNormalizer(grid_features)

    def conv_forward(self, imgs):
        obs = imgs

        if self.object_normalizer is not None:
            obs = self.object_normalizer(obs)
        
        x = self.activation(self.conv1(obs))
        x = self.activation(self.conv2(x))
        return x
