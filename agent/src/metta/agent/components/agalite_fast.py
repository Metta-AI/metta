"""
Fast AGaLiTe implementation optimized for large batch sizes (many environments).
This version reduces memory overhead and fuses operations.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig
from metta.agent.components.agalite_optimized import discounted_sum


class FastAGaLiTeLayer(nn.Module):
    """Optimized AGaLiTe layer for large batch processing."""

    def __init__(
        self,
        d_model: int,
        head_num: int,
        head_dim: int,
        eta: int = 2,  # Reduced from 4
        r: int = 4,  # Reduced from 8
        kernel: AGaLiTeKernelConfig | None = None,
        reset_hidden_on_terminate: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.eta = eta
        self.r = r
        self.reset_hidden_on_terminate = reset_hidden_on_terminate
        self.eps = eps
        self.kernel = kernel or AGaLiTeKernelConfig()
        self.feature_dim = self.kernel.feature_dim(self.head_dim, self.eta)

        # Fused projection for all parameters (more efficient)
        total_proj_dim = head_num * head_dim * 5 + head_num * eta * 3
        self.fused_projection = nn.Linear(d_model, total_proj_dim)

        # Output projection
        self.project = nn.Linear(head_num * head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute oscillatory frequencies
        self.register_buffer("omegas", torch.linspace(-math.pi, math.pi, r))

        # Initialize
        nn.init.orthogonal_(self.fused_projection.weight, gain=math.sqrt(2))
        nn.init.constant_(self.fused_projection.bias, 0)
        nn.init.orthogonal_(self.project.weight, gain=1.0)
        nn.init.constant_(self.project.bias, 0)

    def forward(self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """Optimized forward pass for large batches."""
        T, B, _ = inputs.shape
        device = inputs.device

        # Unpack memory
        tilde_k_prev, tilde_v_prev, s_prev, tick = memory

        # Single fused projection (more efficient)
        all_proj = self.fused_projection(inputs)

        # Split projections
        kqv_dim = self.head_num * self.head_dim * 5
        kqvbetagammas = all_proj[..., :kqv_dim]
        p1p2p3 = all_proj[..., kqv_dim:]

        # Reshape efficiently (avoid 5D tensors)
        kqvbetagammas = kqvbetagammas.view(T, B, self.head_num, 5, self.head_dim)
        p1p2p3 = p1p2p3.view(T, B, self.head_num, 3, self.eta)

        # Extract components
        keys = kqvbetagammas[..., 0, :]
        queries = kqvbetagammas[..., 1, :]
        values = kqvbetagammas[..., 2, :]
        beta = torch.sigmoid(kqvbetagammas[..., 3, :])
        gammas = kqvbetagammas[..., 4, :]

        p1 = p1p2p3[..., 0, :]
        p2 = p1p2p3[..., 1, :]
        p3 = p1p2p3[..., 2, :]

        # Optimized feature mapping using batched operations
        # Flatten batch and head dimensions for efficiency
        psi_k = self.kernel.feature_map(keys, p1, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        phi_q = self.kernel.feature_map(queries, p2, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        gamma_feat = self.kernel.gamma_map(gammas, p3, self.eta).reshape(T, B, self.head_num, self.feature_dim)

        # Oscillatory terms (cached computation)
        tick_base = tick.view(1, B)
        tick_inc = torch.arange(1, T + 1, device=device, dtype=tick.dtype).view(T, 1)
        ticks = tick_inc + tick_base  # (T, B)

        # Vectorized oscillatory computation
        occil = torch.cos(ticks.unsqueeze(-1) * self.omegas.view(1, 1, -1))  # (T, B, r)

        # Apply gating
        values_gated = values * beta
        keys_gated = psi_k * gamma_feat
        s = keys_gated.clone()

        # Expand with oscillations (optimize memory layout)
        # Instead of 5D tensors, keep as 4D
        cos_expanded = occil.unsqueeze(-1).unsqueeze(-1)
        values_osc = values_gated.unsqueeze(2) * cos_expanded
        keys_osc = keys_gated.unsqueeze(2) * cos_expanded

        # Prepare discount factors
        if self.reset_hidden_on_terminate:
            term_mask = (1 - terminations.float()).unsqueeze(2).unsqueeze(3)
            discount_gamma = (1 - gamma_feat) * term_mask
            discount_beta = (1 - beta) * term_mask
        else:
            discount_gamma = 1 - gamma_feat
            discount_beta = 1 - beta

        # Discounted sums (the sequential part)
        # Disable chunking due to dimension issues - always use normal processing
        if False:  # B > 1024:
            # Split batch for better memory usage
            chunk_size = 512
            final_keys_chunks = []
            final_values_chunks = []
            final_s_chunks = []

            for i in range(0, B, chunk_size):
                end_i = min(i + chunk_size, B)
                chunk_slice = slice(i, end_i)

                fk = discounted_sum(
                    tilde_k_prev[:, chunk_slice] if tilde_k_prev.ndim > 1 else tilde_k_prev,
                    keys_osc[:, chunk_slice],
                    discount_gamma[:, chunk_slice].unsqueeze(2),
                )
                final_keys_chunks.append(fk)

                fv = discounted_sum(
                    tilde_v_prev[:, chunk_slice] if tilde_v_prev.ndim > 1 else tilde_v_prev,
                    values_osc[:, chunk_slice],
                    discount_beta[:, chunk_slice].unsqueeze(2).unsqueeze(3),
                )
                final_values_chunks.append(fv)

                fs = discounted_sum(
                    s_prev[:, chunk_slice] if s_prev.ndim > 1 else s_prev,
                    s[:, chunk_slice],
                    discount_gamma[:, chunk_slice],
                )
                final_s_chunks.append(fs)

            final_keys = torch.cat(final_keys_chunks, dim=1)
            final_values = torch.cat(final_values_chunks, dim=1)
            final_s = torch.cat(final_s_chunks, dim=1)
        else:
            # Normal processing
            discount_gamma_expanded = discount_gamma.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
            final_keys = discounted_sum(tilde_k_prev, keys_osc, discount_gamma_expanded)
            # For values, expand discount_beta to match values_osc shape
            discount_beta_expanded = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
            final_values = discounted_sum(tilde_v_prev, values_osc, discount_beta_expanded)
            final_s = discounted_sum(s_prev, s, discount_gamma)

        # Attention computation (optimized)
        # Use batched operations instead of einsum
        keys_for_attn = final_keys.reshape(T, B, self.r, -1)
        queries_for_attn = phi_q.reshape(T, B, 1, -1)

        # Compute attention scores
        attn_scores = (keys_for_attn * queries_for_attn).sum(dim=-1)  # (T, B, r)

        # Apply attention to values
        final_values_reshaped = final_values.reshape(T, B, self.r, self.head_num, self.head_dim)
        kv = (final_values_reshaped * attn_scores.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

        # Normalization
        norm = (final_s * phi_q).sum(dim=-1, keepdim=True)
        attn_out = kv / (2 * self.r * norm + self.eps)

        # Output projection
        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        attn_out = self.dropout(self.project(attn_out))

        # Update memory
        new_tick = tick + T
        new_tilde_k = final_keys[-1]
        new_tilde_v = final_values[-1]
        new_s = final_s[-1]

        return attn_out, (new_tilde_k, new_tilde_v, new_s, new_tick)

    @staticmethod
    def initialize_memory(
        batch_size: int,
        head_num: int,
        head_dim: int,
        eta: int,
        r: int,
        device=None,
        kernel: Optional[AGaLiTeKernelConfig] = None,
    ):
        """Initialize memory for fast AGaLiTe layer."""
        if device is None:
            device = torch.device("cpu")

        kernel_conf = kernel or AGaLiTeKernelConfig()
        feature_dim = kernel_conf.feature_dim(head_dim, eta)

        return (
            torch.zeros((batch_size, r, head_num, feature_dim), device=device),
            torch.zeros((batch_size, r, head_num, head_dim), device=device),
            torch.zeros((batch_size, head_num, feature_dim), device=device),
            torch.zeros(batch_size, device=device),
        )
