"""Kernel configuration utilities for AGaLiTe feature maps."""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from mettagrid.config import Config


def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """ELU plus one nonlinearity used in AGaLiTe kernel variants."""
    return F.elu(x, alpha=1.0) + 1.0


class AGaLiTeKernelConfig(Config):
    """Configures feature map activations for AGaLiTe attention."""

    name: Literal["relu", "eluplus1"] = "relu"
    nu: int = 4

    def feature_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Activation applied to key/query projections (φ and ψ maps)."""
        if self.name == "relu":
            return F.relu
        if self.name == "eluplus1":
            return _elu_plus_one
        raise ValueError(f"Unsupported AGaLiTe kernel '{self.name}'.")

    def gamma_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Activation used for γ projections. Always sigmoid per paper."""
        return torch.sigmoid

    def project_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Activation for auxiliary projections p1/p2/p3."""
        return self.feature_activation()

    def gamma_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projection applied before gamma gating; identity for default kernel."""
        if self.name == "relu":
            return tensor
        return self.project_activation()(tensor)
