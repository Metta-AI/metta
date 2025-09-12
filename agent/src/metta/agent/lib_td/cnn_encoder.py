from typing import Optional

import pufferlib.pytorch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict

from metta.common.config.config import Config


class CNNEncoderConfig(Config):
    """This class hardcodes for two CNNs and a two layer MLP. The box shaper and obs normalizer are not configurable."""

    cnn1_cfg: dict = Field(default_factory=lambda: {"out_channels": 64, "kernel_size": 5, "stride": 3})
    cnn2_cfg: dict = Field(default_factory=lambda: {"out_channels": 64, "kernel_size": 3, "stride": 1})
    fc1_cfg: dict = Field(default_factory=lambda: {"out_features": 128})
    encoded_obs_cfg: dict = Field(default_factory=lambda: {"out_features": 128})

    def instantiate(self):
        return CNNEncoder(config=self)


class CNNEncoder(nn.Module):
    def __init__(self, config: Optional[CNNEncoderConfig] = None):
        super().__init__()
        self.config = config or CNNEncoderConfig()

        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.num_layers, **self.config.cnn1_cfg),
            std=1.0,  # Match YAML orthogonal gain=1
        )

        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.config.cnn1_cfg["out_channels"], **self.config.cnn2_cfg),
            std=1.0,  # Match YAML orthogonal gain=1,
        )

        self.flatten = nn.Flatten()

        # Match YAML: Linear layers use orthogonal with gain=1
        self.fc1 = pufferlib.pytorch.layer_init(nn.LazyLinear(self.config.fc1_cfg["out_features"]), std=1.0)
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.fc1_cfg["out_features"], self.config.encoded_obs_cfg["out_features"]), std=1.0
        )

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass shows how we can use tensor dicts or just regular tensors interchangeably within a forward.
        For instance, if you want one of the stack's outputs in a loss, you can simply add td["layer_name"] = x at the
        correct line.
        For this reason, avoid using nn.sequential so you can have access to intermediate outputs."""
        x = td["box_obs"]
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.encoded_obs(x)
        x = F.relu(x)
        td["encoded_obs"] = x

        return td
