"""Core AGaLiTe transformer built from reusable blocks."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from metta.agent.components.agalite_enhanced import AGaLiTeTransformerLayer
from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig

logger = logging.getLogger(__name__)


class AGaLiTeCore(nn.Module):
    """Stack of AGaLiTe transformer layers."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        *,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        gru_bias: float = 2.0,
        kernel: Optional[AGaLiTeKernelConfig] = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.d_ffc = d_ffc
        self.n_heads = n_heads
        self.reset_on_terminate = reset_on_terminate
        self.eta = eta
        self.r = r

        self.kernel = kernel or AGaLiTeKernelConfig()
        logger.debug("initialising AGaLiTe core (layers=%s, heads=%s, eta=%s, r=%s)", n_layers, n_heads, eta, r)

        self.encoders = nn.ModuleList()
        for layer_idx in range(n_layers):
            layer = AGaLiTeTransformerLayer(
                d_model=d_model,
                d_head=d_head,
                d_ffc=d_ffc,
                n_heads=n_heads,
                eta=eta,
                r=r,
                kernel=self.kernel,
                use_input_proj=layer_idx == 0,
                gru_bias=gru_bias,
                reset_hidden_on_terminate=reset_on_terminate,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            self.encoders.append(layer)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Dict[str, Tuple]
    ) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
        output = inputs
        updated_memory: Dict[str, Tuple] = {}

        for idx, encoder in enumerate(self.encoders):
            key = f"layer_{idx + 1}"
            output, layer_memory = encoder(output, terminations, memory[key])
            updated_memory[key] = layer_memory

        return output, updated_memory

    def initialize_memory(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Tuple]:
        if device is None:
            device = torch.device("cpu")

        memory: Dict[str, Tuple] = {}
        for idx, encoder in enumerate(self.encoders):
            key = f"layer_{idx + 1}"
            memory[key] = encoder.initialize_memory(batch_size, device)
        return memory

    @property
    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "eta": self.eta,
            "r": self.r,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
        }

    def get_config(self) -> Dict[str, object]:
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_head": self.d_head,
            "d_ffc": self.d_ffc,
            "n_heads": self.n_heads,
            "eta": self.eta,
            "r": self.r,
            "reset_on_terminate": self.reset_on_terminate,
        }


__all__ = ["AGaLiTeCore"]
