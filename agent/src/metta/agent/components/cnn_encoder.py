import math
from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict

import pufferlib.pytorch
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

        # Match YAML: Linear layers use orthogonal with gain=1. Avoid LazyLinear so modules
        # materialize before DistributedDataParallel wrapping.
        flattened_size = self._compute_flattened_size(
            (env.obs_height, env.obs_width),
            self.config.cnn1_cfg,
            self.config.cnn2_cfg,
        )
        self.fc1 = pufferlib.pytorch.layer_init(
            nn.Linear(flattened_size, self.config.fc1_cfg["out_features"]),
            std=1.0,
        )
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.fc1_cfg["out_features"], self.config.encoded_obs_cfg["out_features"]), std=1.0
        )

    @staticmethod
    def _conv2d_output_shape(
        input_hw: Tuple[int, int],
        kernel_size,
        stride,
        padding=0,
        dilation=1,
    ) -> Tuple[int, int]:
        """Compute spatial output size for Conv2D with given parameters."""

        def _to_pair(val) -> Tuple[int, int]:
            if isinstance(val, tuple):
                return val
            return (val, val)

        h, w = input_hw
        k_h, k_w = _to_pair(kernel_size)
        s_h, s_w = _to_pair(stride)
        p_h, p_w = _to_pair(padding)
        d_h, d_w = _to_pair(dilation)

        def _single(out_size, k, s, p, d) -> int:
            return math.floor((out_size + 2 * p - d * (k - 1) - 1) / s + 1)

        return (
            _single(h, k_h, s_h, p_h, d_h),
            _single(w, k_w, s_w, p_w, d_w),
        )

    def _compute_flattened_size(
        self,
        input_hw: Tuple[int, int],
        conv1_cfg: dict,
        conv2_cfg: dict,
    ) -> int:
        conv1_hw = self._conv2d_output_shape(
            input_hw,
            conv1_cfg.get("kernel_size", 3),
            conv1_cfg.get("stride", 1),
            conv1_cfg.get("padding", 0),
            conv1_cfg.get("dilation", 1),
        )
        conv2_hw = self._conv2d_output_shape(
            conv1_hw,
            conv2_cfg.get("kernel_size", 3),
            conv2_cfg.get("stride", 1),
            conv2_cfg.get("padding", 0),
            conv2_cfg.get("dilation", 1),
        )

        conv2_out_channels = conv2_cfg.get("out_channels")
        if conv2_out_channels is None:
            raise ValueError("cnn2_cfg must define 'out_channels' to compute flattened size")

        flattened = conv2_out_channels * conv2_hw[0] * conv2_hw[1]
        if flattened <= 0:
            raise ValueError(
                "Computed flattened size for CNN encoder is non-positive. "
                "Check convolution hyperparameters and observation dimensions."
            )
        return flattened

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass through the CNN encoder stack."""
        x = td[self.config.in_key]
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.encoded_obs(x))
        td[self.config.out_key] = x

        return td
