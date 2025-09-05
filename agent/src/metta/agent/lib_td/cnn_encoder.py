from typing import Optional

import pufferlib.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict

from metta.agent.lib_td.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib_td.observation_normalizer import ObservationNormalizer
from metta.common.config.config import Config


class CNNEncoderConfig(Config):
    """This class hardcodes for two CNNs and a two layer MLP. The box shaper and obs normalizer are not configurable."""

    cnn1_cfg: dict = Field(default_factory=lambda: {"out_channels": 64, "kernel_size": 5, "stride": 3})
    cnn2_cfg: dict = Field(default_factory=lambda: {"out_channels": 64, "kernel_size": 3, "stride": 1})
    fc1_cfg: dict = Field(default_factory=lambda: {"out_features": 128})
    encoded_obs_cfg: dict = Field(default_factory=lambda: {"out_features": 128})


class CNNEncoder(nn.Module):
    def __init__(self, obs_meta: dict, config: Optional[CNNEncoderConfig] = None):
        super().__init__()
        self.config = config or CNNEncoderConfig()

        self.obs_shaper = ObsTokenToBoxShaper(
            obs_meta["obs_space"], obs_meta["obs_width"], obs_meta["obs_height"], obs_meta["feature_normalizations"]
        )
        self.num_layers = max(obs_meta["feature_normalizations"].keys()) + 1
        self.obs_normalizer = ObservationNormalizer(obs_meta["feature_normalizations"])

        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.num_layers, **self.config.cnn1_cfg),
            std=1.0,  # Match YAML orthogonal gain=1
        )

        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.config.cnn1_cfg["out_channels"], **self.config.cnn2_cfg),
            std=1.0,  # Match YAML orthogonal gain=1,
        )

        test_input = torch.zeros(1, self.num_layers, obs_meta["obs_width"], obs_meta["obs_height"])
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Match YAML: Linear layers use orthogonal with gain=1
        self.fc1 = pufferlib.pytorch.layer_init(
            nn.Linear(self.flattened_size, self.config.fc1_cfg["out_features"]), std=1.0
        )
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.fc1_cfg["out_features"], self.config.encoded_obs_cfg["out_features"]), std=1.0
        )

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass shows how we can use tensor dicts or just regular tensors interchangeably within a forward.
        For instance, if you want one of the stack's outputs in a loss, you can simply add td["layer_name"] = x at the
        correct line.
        For this reason, avoid using nn.sequential so you can have access to intermediate outputs."""
        td = self.obs_shaper(td)
        td = self.obs_normalizer(td)
        x = td["obs_normalizer"]
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
