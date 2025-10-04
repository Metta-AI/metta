"""Kernel configuration utilities for AGaLiTe feature maps."""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F

from mettagrid.base_config import Config


def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """ELU plus one nonlinearity used in AGaLiTe kernel variants."""
    return F.elu(x, alpha=1.0) + 1.0


class AGaLiTeKernelConfig(Config):
    """Configures feature map activations for AGaLiTe attention."""

    name: Literal["relu", "eluplus1", "dpfp", "pp_relu", "pp_eluplus1"] = "relu"
    nu: int = 4

    def feature_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Activation applied to key/query projections (φ and ψ maps)."""
        if self.name == "relu":
            return F.relu
        if self.name == "eluplus1":
            return _elu_plus_one
        if self.name == "pp_relu":
            return F.relu
        if self.name == "pp_eluplus1":
            return _elu_plus_one
        raise ValueError(f"Unsupported AGaLiTe kernel '{self.name}'.")

    def gamma_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Activation used for γ projections. Always sigmoid per paper."""
        return torch.sigmoid

    def project_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Activation for auxiliary projections p1/p2/p3."""
        return self.feature_activation()

    def feature_dim(self, base_dim: int, eta: int) -> int:
        """Return the dimensionality of the transformed feature map."""

        if self.name == "dpfp":
            return 2 * base_dim * max(self.nu, 1)
        return base_dim * eta

    def feature_map(self, base: torch.Tensor, proj: torch.Tensor, eta: int) -> torch.Tensor:
        """Apply the kernel-specific feature mapping."""

        if self.name == "dpfp":
            return self._dpfp(base)

        feature_act = self.feature_activation()
        proj_act = self.project_activation()
        mapped = torch.einsum(
            "...d,...e->...de",
            feature_act(base),
            proj_act(proj),
        )
        return mapped.reshape(*base.shape[:-1], base.shape[-1] * eta)

    def gamma_map(self, gamma: torch.Tensor, proj: torch.Tensor, eta: int) -> torch.Tensor:
        """Kernel-specific map used for gamma gating."""

        if self.name == "dpfp":
            return torch.sigmoid(self._dpfp(gamma))

        proj_act = self.project_activation()
        mapped = torch.einsum(
            "...d,...e->...de",
            torch.sigmoid(gamma),
            proj_act(proj),
        )
        return mapped.reshape(*gamma.shape[:-1], gamma.shape[-1] * eta)

    def _dpfp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Deterministic positive feature projection (DPFP)."""

        nu = max(self.nu, 1)
        pos = torch.relu(tensor)
        neg = torch.relu(-tensor)
        stacked = torch.cat([pos, neg], dim=-1)

        outs = []
        for shift in range(1, nu + 1):
            rolled = torch.roll(stacked, shifts=shift, dims=-1)
            outs.append(stacked * rolled)
        return torch.cat(outs, dim=-1)
