"""
Fast AGaLiTe implementation optimized for large batch sizes (many environments).
This version reduces memory overhead and fuses operations.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metta.agent.modules.agalite_optimized import discounted_sum


class FastAGaLiTeLayer(nn.Module):
    """High-performance AGaLiTe layer with optimized batching and JIT compilation."""

    def __init__(
        self,
        d_model: int,
        head_num: int,
        head_dim: int,
        eta: int = 2,  # Reduced from 4
        r: int = 4,  # Reduced from 8
        reset_hidden_on_terminate: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-5,  # Increased for better stability
    ):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.eta = eta
        self.r = r
        self.reset_hidden_on_terminate = reset_hidden_on_terminate
        self.eps = eps

        # Fused projection for all parameters (more efficient)
        total_proj_dim = head_num * head_dim * 5 + head_num * eta * 3
        self.fused_projection = nn.Linear(d_model, total_proj_dim)

        # Output projection
        self.project = nn.Linear(head_num * head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute oscillatory frequencies
        self.register_buffer("omegas", torch.linspace(-math.pi, math.pi, r))

        # Initialize with conservative values for stability
        # Recurrent layers benefit from smaller initialization
        init_std = 0.5  # Conservative gain for recurrent architecture
        nn.init.orthogonal_(self.fused_projection.weight, gain=init_std)
        nn.init.constant_(self.fused_projection.bias, 0.0)
        nn.init.orthogonal_(self.project.weight, gain=init_std)
        nn.init.constant_(self.project.bias, 0.0)

    @torch._dynamo.disable  # Avoid graph breaks in recurrent computation
    def forward(self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """Optimized forward pass with compiler directives."""
        T, B, _ = inputs.shape
        device = inputs.device

        # Unpack memory
        tilde_k_prev, tilde_v_prev, s_prev, tick = memory

        # Validate memory shapes
        expected_k_shape = (B, self.r, self.head_num, self.eta * self.head_dim)
        expected_v_shape = (B, self.r, self.head_num, self.head_dim)
        expected_s_shape = (B, self.head_num, self.eta * self.head_dim)

        if tilde_k_prev.shape != expected_k_shape:
            raise ValueError(
                f"Memory tilde_k_prev shape {tilde_k_prev.shape} doesn't match expected {expected_k_shape}"
            )
        if tilde_v_prev.shape != expected_v_shape:
            raise ValueError(
                f"Memory tilde_v_prev shape {tilde_v_prev.shape} doesn't match expected {expected_v_shape}"
            )
        if s_prev.shape != expected_s_shape:
            raise ValueError(f"Memory s_prev shape {s_prev.shape} doesn't match expected {expected_s_shape}")

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
        TB = T * B
        keys_flat = F.relu(keys).reshape(TB * self.head_num, self.head_dim)
        p1_flat = F.relu(p1).reshape(TB * self.head_num, self.eta)

        # Use batched matrix multiply instead of einsum
        keys_expanded = torch.bmm(keys_flat.unsqueeze(2), p1_flat.unsqueeze(1)).reshape(
            T, B, self.head_num, self.head_dim * self.eta
        )

        # Similar for queries and gammas
        queries_flat = F.relu(queries).reshape(TB * self.head_num, self.head_dim)
        p2_flat = F.relu(p2).reshape(TB * self.head_num, self.eta)
        queries_expanded = torch.bmm(queries_flat.unsqueeze(2), p2_flat.unsqueeze(1)).reshape(
            T, B, self.head_num, self.head_dim * self.eta
        )

        gammas_flat = torch.sigmoid(gammas).reshape(TB * self.head_num, self.head_dim)
        p3_flat = torch.sigmoid(p3).reshape(TB * self.head_num, self.eta)
        gammas_expanded = torch.bmm(gammas_flat.unsqueeze(2), p3_flat.unsqueeze(1)).reshape(
            T, B, self.head_num, self.head_dim * self.eta
        )

        # Oscillatory terms (cached computation)
        tick_inc = torch.arange(1, T + 1, device=device, dtype=tick.dtype)
        ticks = tick + tick_inc.view(T, 1, 1)

        # Vectorized oscillatory computation
        occil = torch.cos(ticks @ self.omegas.unsqueeze(0))  # (T, B, r)

        # Apply gating
        values_gated = values * beta
        keys_gated = keys_expanded * gammas_expanded
        s = keys_gated.clone()

        # Expand with oscillations (optimize memory layout)
        # Instead of 5D tensors, keep as 4D
        values_osc = values_gated.unsqueeze(2) * occil.unsqueeze(-1).unsqueeze(-1)
        keys_osc = keys_gated.unsqueeze(2) * occil.unsqueeze(-1).unsqueeze(-1)

        # Prepare discount factors
        if self.reset_hidden_on_terminate:
            term_mask = (1 - terminations.float()).unsqueeze(2).unsqueeze(3)
            discount_gamma = (1 - gammas_expanded) * term_mask
            # For beta, we need term_mask to broadcast with (T, B, head_num, head_dim)
            # beta has shape (T, B, head_num, head_dim)
            term_mask_beta = (1 - terminations.float()).unsqueeze(2).unsqueeze(3)
            discount_beta = (1 - beta) * term_mask_beta
        else:
            discount_gamma = 1 - gammas_expanded
            discount_beta = 1 - beta

        # Discounted sums - use chunking for better memory efficiency
        # Lower threshold for better parallelization
        if B > 256:
            # Optimal chunk size for GPU parallelization
            chunk_size = 128
            final_keys_chunks = []
            final_values_chunks = []
            final_s_chunks = []

            for i in range(0, B, chunk_size):
                end_i = min(i + chunk_size, B)
                chunk_slice = slice(i, end_i)

                fk = discounted_sum(
                    tilde_k_prev[chunk_slice] if tilde_k_prev.ndim > 1 else tilde_k_prev,
                    keys_osc[:, chunk_slice],
                    discount_gamma[:, chunk_slice].unsqueeze(2),
                )
                final_keys_chunks.append(fk)

                # For values: tilde_v_prev is (B, r, head_num, head_dim)
                # values_osc is (T, B, r, head_num, head_dim)
                # discount_beta is (T, B, head_num, head_dim)
                # We need to expand discount_beta to match values_osc shape
                discount_beta_expanded = discount_beta[:, chunk_slice].unsqueeze(2).expand(-1, -1, self.r, -1, -1)
                fv = discounted_sum(
                    tilde_v_prev[chunk_slice] if tilde_v_prev.ndim > 1 else tilde_v_prev,
                    values_osc[:, chunk_slice],
                    discount_beta_expanded,
                )
                final_values_chunks.append(fv)

                fs = discounted_sum(
                    s_prev[chunk_slice] if s_prev.ndim > 1 else s_prev,
                    s[:, chunk_slice],
                    discount_gamma[:, chunk_slice],
                )
                final_s_chunks.append(fs)

            final_keys = torch.cat(final_keys_chunks, dim=1)
            final_values = torch.cat(final_values_chunks, dim=1)
            final_s = torch.cat(final_s_chunks, dim=1)
        else:
            # Normal processing
            final_keys = discounted_sum(tilde_k_prev, keys_osc, discount_gamma.unsqueeze(2))
            # For values, expand discount_beta to match values_osc shape
            discount_beta_expanded = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
            final_values = discounted_sum(tilde_v_prev, values_osc, discount_beta_expanded)
            final_s = discounted_sum(s_prev, s, discount_gamma)

        # Attention computation (optimized)
        # Use batched operations instead of einsum
        keys_for_attn = final_keys.reshape(T, B, self.r, -1)
        queries_for_attn = queries_expanded.reshape(T, B, 1, -1)

        # Compute attention scores
        attn_scores = (keys_for_attn * queries_for_attn).sum(dim=-1)  # (T, B, r)

        # Apply attention to values
        final_values_reshaped = final_values.reshape(T, B, self.r, self.head_num, self.head_dim)
        kv = (final_values_reshaped * attn_scores.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

        # Normalization with improved numerical stability
        norm = (final_s * queries_expanded).sum(dim=-1, keepdim=True)
        # Ensure norm is non-negative and bounded
        norm = torch.clamp(torch.abs(norm), min=self.eps, max=1e6)
        # Add regularization term to prevent division issues
        denominator = 2 * self.r * norm + 1e-3  # Balanced epsilon for stability
        attn_out = kv / denominator

        # Clamp output to prevent extreme values from propagating
        attn_out = torch.clamp(attn_out, min=-100, max=100)

        # Output projection
        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        attn_out = self.dropout(self.project(attn_out))

        # Final stability check
        if torch.isnan(attn_out).any():
            # If NaN detected, return zeros to prevent propagation
            attn_out = torch.zeros_like(attn_out)

        # Update memory - completely detach from computation graph
        # We need to create new tensors that have no connection to the computation graph
        new_tick = (tick + T).detach()

        # Extract the last timestep and ensure complete detachment
        # Using .data creates a tensor that shares storage but has no grad connection
        new_tilde_k = final_keys[-1].data.clone()
        new_tilde_v = final_values[-1].data.clone()
        new_s = final_s[-1].data.clone()

        return attn_out, (new_tilde_k, new_tilde_v, new_s, new_tick)

    @staticmethod
    def initialize_memory(batch_size: int, head_num: int, head_dim: int, eta: int, r: int, device=None):
        """Initialize memory for fast AGaLiTe layer."""
        if device is None:
            device = torch.device("cpu")

        return (
            torch.zeros((batch_size, r, head_num, eta * head_dim), device=device),
            torch.zeros((batch_size, r, head_num, head_dim), device=device),
            torch.zeros((batch_size, head_num, eta * head_dim), device=device),
            torch.zeros((batch_size, 1), device=device),
        )
