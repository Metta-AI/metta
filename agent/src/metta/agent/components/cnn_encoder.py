from typing import Tuple

import pufferlib.pytorch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class CNNEncoderConfig(ComponentConfig):
    """This class hardcodes for two CNNs and a two layer MLP. The box shaper and obs normalizer are not configurable."""

    in_key: str
    out_key: str
    name: str = "cnn_encoder"
    cnn1_cfg: dict = Field(default_factory=lambda: {"out_channels": 64, "kernel_size": 5, "stride": 3})
    cnn2_cfg: dict = Field(default_factory=lambda: {"out_channels": 64, "kernel_size": 3, "stride": 1})
    fc1_cfg: dict = Field(default_factory=lambda: {"out_features": 128})
    encoded_obs_cfg: dict = Field(default_factory=lambda: {"out_features": 128})

    def make_component(self, env=None):
        return CNNEncoder(config=self, env=env)


class CNNEncoder(nn.Module):
    def __init__(self, config: CNNEncoderConfig, env=None):
        super().__init__()
        self.config = config

        num_layers = max(env.feature_normalizations.keys()) + 1

        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(num_layers, **self.config.cnn1_cfg),
            std=1.0,  # Match YAML orthogonal gain=1
        )

        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.config.cnn1_cfg["out_channels"], **self.config.cnn2_cfg),
            std=1.0,  # Match YAML orthogonal gain=1,
        )

        self.flatten = nn.Flatten()

        # Match YAML: Linear layers use orthogonal with gain=1
        conv2_out_hw = self._compute_conv_stack_shape(env.obs_height, env.obs_width)
        flattened_dim = self.config.cnn2_cfg["out_channels"] * conv2_out_hw[0] * conv2_out_hw[1]

        self.fc1 = pufferlib.pytorch.layer_init(
            nn.Linear(flattened_dim, self.config.fc1_cfg["out_features"]),
            std=1.0,
        )
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.fc1_cfg["out_features"], self.config.encoded_obs_cfg["out_features"]), std=1.0
        )

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass through the CNN encoder stack."""
        x = td[self.config.in_key]
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.encoded_obs(x))
        td[self.config.out_key] = x

        return td

    def _compute_conv_stack_shape(self, height: int, width: int) -> Tuple[int, int]:
        conv1_h = self._conv_output_dim(
            height,
            kernel=self.config.cnn1_cfg["kernel_size"],
            stride=self.config.cnn1_cfg.get("stride", 1),
            padding=self.config.cnn1_cfg.get("padding", 0),
        )
        conv1_w = self._conv_output_dim(
            width,
            kernel=self.config.cnn1_cfg["kernel_size"],
            stride=self.config.cnn1_cfg.get("stride", 1),
            padding=self.config.cnn1_cfg.get("padding", 0),
        )

        conv2_h = self._conv_output_dim(
            conv1_h,
            kernel=self.config.cnn2_cfg["kernel_size"],
            stride=self.config.cnn2_cfg.get("stride", 1),
            padding=self.config.cnn2_cfg.get("padding", 0),
        )
        conv2_w = self._conv_output_dim(
            conv1_w,
            kernel=self.config.cnn2_cfg["kernel_size"],
            stride=self.config.cnn2_cfg.get("stride", 1),
            padding=self.config.cnn2_cfg.get("padding", 0),
        )

        return conv2_h, conv2_w

    @staticmethod
    def _conv_output_dim(size: int, kernel: int, stride: int, padding: int) -> int:
        numerator = size + 2 * padding - kernel
        if (numerator % stride) != 0:
            raise ValueError(
                "CNNEncoder expects stride to exactly divide the padded input minus kernel size. "
                f"Got size={size}, kernel={kernel}, stride={stride}, padding={padding}."
            )
        return numerator // stride + 1
