from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class NoiseLayerConfig(ComponentConfig):
    """Configuration for NoiseLayer."""

    in_key: str
    out_key: str
    std: float
    name: str = "noise_layer"
    noise_during_eval: bool = False
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    def make_component(self, env=None) -> nn.Module:
        return NoiseLayer(config=self)


class NoiseLayer(nn.Module):
    """Applies additive Gaussian noise to a TensorDict entry."""

    def __init__(self, config: NoiseLayerConfig):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.std = self._validate_std(config.std)
        self.noise_during_eval = bool(config.noise_during_eval)
        self.clamp_min = config.clamp_min
        self.clamp_max = config.clamp_max

    def set_noise(self, *, std: float, noise_during_eval: Optional[bool] = None) -> None:
        """Update noise parameters at runtime."""
        self.std = self._validate_std(std)
        if noise_during_eval is not None:
            self.noise_during_eval = bool(noise_during_eval)

    def forward(self, td: TensorDict) -> TensorDict:
        tensor = td[self.in_key]
        noisy_tensor = self._maybe_add_noise(tensor)
        if self.clamp_min is not None or self.clamp_max is not None:
            noisy_tensor = torch.clamp(noisy_tensor, min=self.clamp_min, max=self.clamp_max)
        td[self.out_key] = noisy_tensor
        return td

    def _maybe_add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std == 0.0:
            return tensor
        if not tensor.is_floating_point():
            raise TypeError("NoiseLayer expects floating point tensors when std is non-zero.")
        if not self.training and not self.noise_during_eval:
            return tensor

        noise = torch.randn_like(tensor) * self.std
        return tensor + noise

    def extra_repr(self) -> str:
        return (
            f"in_key={self.in_key}, out_key={self.out_key}, std={self.std}, "
            f"noise_during_eval={self.noise_during_eval}, "
            f"clamp_min={self.clamp_min}, clamp_max={self.clamp_max}"
        )

    @staticmethod
    def _validate_std(std: float) -> float:
        if std < 0.0:
            raise ValueError("Noise standard deviation must be non-negative.")
        return float(std)
