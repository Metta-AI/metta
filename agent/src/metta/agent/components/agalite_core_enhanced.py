"""
Enhanced AGaLiTe core backed by the paper-aligned attention layers.

This module provides transformer functionality with mode switching between
GaLiTe (exact) and AGaLiTe (approximated) implementations.
"""

import logging
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from metta.agent.components.agalite_enhanced import EnhancedTransformerEncoder
from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig

logger = logging.getLogger(__name__)


class EnhancedAGaLiTeCore(nn.Module):
    """
    Enhanced AGaLiTe transformer core with full paper implementation.

    Supports two modes:
    - "galite": Exact linear attention without approximation
    - "agalite": Full AGaLiTe with oscillatory approximation
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        mode: Literal["galite", "agalite"] = "agalite",
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        gru_bias: float = 2.0,
        kernel: Optional[AGaLiTeKernelConfig] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.d_ffc = d_ffc
        self.n_heads = n_heads
        self.mode = mode
        self.reset_on_terminate = reset_on_terminate

        self.eta = eta
        self.r = r
        logger.info(f"Using {mode} mode with parameters: eta={self.eta}, r={self.r}")

        self.kernel = kernel or AGaLiTeKernelConfig()

        # Create encoder layers
        self.encoders = nn.ModuleList()
        for layer_idx in range(n_layers):
            # Use enhanced implementation with mode selection
            use_dense = layer_idx == 0  # Use dense layer for first layer
            encoder = EnhancedTransformerEncoder(
                d_model=d_model,
                d_head=d_head,
                d_ffc=d_ffc,
                n_heads=n_heads,
                eta=self.eta,
                r=self.r,
                kernel=self.kernel,
                mode=mode,
                use_dense=use_dense,
                gru_bias=gru_bias,
                reset_hidden_on_terminate=reset_on_terminate,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            self.encoders.append(encoder)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Dict[str, Tuple]
    ) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
        """Forward pass through all encoder layers."""
        u_i = inputs
        new_memory = {}

        for layer_idx, encoder in enumerate(self.encoders):
            layer_key = f"layer_{layer_idx + 1}"

            u_i, memory_updated = encoder(u_i, terminations, memory[layer_key])

            new_memory[layer_key] = memory_updated

        return u_i, new_memory

    def initialize_memory(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Tuple]:
        """Initialize memory for all layers based on mode."""
        if device is None:
            device = torch.device("cpu")

        memory_dict = {}

        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx + 1}"

            encoder = self.encoders[layer_idx]
            memory_dict[layer_key] = encoder.initialize_memory(batch_size, device)

        return memory_dict

    @property
    def parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown for analysis."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "mode": self.mode,
            "eta": self.eta,
            "r": self.r,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
        }

    def get_config(self) -> Dict:
        """Get model configuration for checkpointing and reproducibility."""
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_head": self.d_head,
            "d_ffc": self.d_ffc,
            "n_heads": self.n_heads,
            "eta": self.eta,
            "r": self.r,
            "mode": self.mode,
            "reset_on_terminate": self.reset_on_terminate,
        }
