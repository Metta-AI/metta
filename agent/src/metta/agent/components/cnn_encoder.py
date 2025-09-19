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
        height1 = self._conv_out_dim(env.obs_height, self.config.cnn1_cfg)
        width1 = self._conv_out_dim(env.obs_width, self.config.cnn1_cfg)
        height2 = self._conv_out_dim(height1, self.config.cnn2_cfg)
        width2 = self._conv_out_dim(width1, self.config.cnn2_cfg)
        flattened_dim = self.config.cnn2_cfg["out_channels"] * height2 * width2

        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(flattened_dim, self.config.fc1_cfg["out_features"]), std=1.0)
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.fc1_cfg["out_features"], self.config.encoded_obs_cfg["out_features"]), std=1.0
        )

    @staticmethod
    def _conv_out_dim(size: int, cfg: dict) -> int:
        padding = cfg.get("padding", 0) or 0
        stride = cfg.get("stride", 1)
        kernel = cfg["kernel_size"]
        numerator = size + 2 * padding - kernel
        if numerator < 0 or numerator % stride != 0:
            raise ValueError(f"Invalid conv params: size={size}, kernel={kernel}, stride={stride}, padding={padding}")
        out = numerator // stride + 1
        if out <= 0:
            raise ValueError(
                f"Convolution with kernel={kernel}, stride={stride}, padding={padding} produced non-positive dim"
            )
        return out

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass shows how we can use tensor dicts or just regular tensors interchangeably within a forward.
        For instance, if you want one of the stack's outputs in a loss, you can simply add td["layer_name"] = x at the
        correct line.
        For this reason, avoid using nn.sequential so you can have access to intermediate outputs."""
        x = td[self.config.in_key]
        x = self.cnn1(x)
        x = F.relu(x)
        td["cnn1"] = x
        x = self.cnn2(x)
        x = F.relu(x)
        td["cnn2"] = x
        x = self.flatten(x)
        td["obs_flattener"] = x
        x = self.fc1(x)
        x = F.relu(x)
        td["fc1"] = x
        x = self.encoded_obs(x)
        x = F.relu(x)
        td[self.config.out_key] = x

        return td
