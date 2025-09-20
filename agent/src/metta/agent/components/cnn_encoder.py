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
        self.fc1 = nn.LazyLinear(self.config.fc1_cfg["out_features"])
        self._fc1_initialized = False
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.fc1_cfg["out_features"], self.config.encoded_obs_cfg["out_features"]), std=1.0
        )

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass through the CNN encoder stack."""
        x = td[self.config.in_key]
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.flatten(x)
        if not self._fc1_initialized:
            flattened = x
            _ = self.fc1(flattened)
            pufferlib.pytorch.layer_init(self.fc1, std=1.0)
            self._fc1_initialized = True
            x = self.fc1(flattened)
        else:
            x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.encoded_obs(x))
        td[self.config.out_key] = x

        return td
