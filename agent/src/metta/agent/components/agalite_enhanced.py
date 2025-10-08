"""AGaLiTe attention blocks reusing standard transformer utilities."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig
from metta.agent.components.agalite_optimized import discounted_sum
from metta.agent.policies.gtrxl import FusedGRUGating


class AGaLiTeAttentionLayer(nn.Module):
    """Approximate AGaLiTe attention with oscillatory feature maps."""

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        head_num: int,
        eta: int,
        r: int,
        kernel: AGaLiTeKernelConfig,
        dropout: float = 0.0,
        eps: float = 1e-6,
        reset_hidden_on_terminate: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.eta = eta
        self.r = r
        self.eps = eps
        self.reset_hidden_on_terminate = reset_hidden_on_terminate
        self.kernel = kernel
        self.feature_dim = self.kernel.feature_dim(self.head_dim, self.eta)

        # Projections
        self.q_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.beta_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.gamma_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.p1_proj = nn.Linear(input_dim, head_num * eta, bias=False)
        self.p2_proj = nn.Linear(input_dim, head_num * eta, bias=False)
        self.p3_proj = nn.Linear(input_dim, head_num * eta, bias=False)

        self.out_proj = nn.Linear(head_num * head_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        omegas = torch.linspace(-math.pi, math.pi, r)
        self.register_buffer("omegas", omegas)
        self._cos_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._sin_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        modules = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.gamma_proj,
            self.p1_proj,
            self.p2_proj,
            self.p3_proj,
        ]
        for module in modules:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

        nn.init.orthogonal_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        T, B, _ = inputs.shape
        device = inputs.device

        tilde_k_prev, tilde_v_prev, s_prev, tick = memory

        queries = self.q_proj(inputs).view(T, B, self.head_num, self.head_dim)
        keys = self.k_proj(inputs).view(T, B, self.head_num, self.head_dim)
        values = self.v_proj(inputs).view(T, B, self.head_num, self.head_dim)

        beta = torch.sigmoid(self.beta_proj(inputs).view(T, B, self.head_num, self.head_dim))
        gamma = torch.sigmoid(self.gamma_proj(inputs).view(T, B, self.head_num, self.head_dim))

        p1 = self.p1_proj(inputs).view(T, B, self.head_num, self.eta)
        p2 = self.p2_proj(inputs).view(T, B, self.head_num, self.eta)
        p3 = self.p3_proj(inputs).view(T, B, self.head_num, self.eta)

        phi_q = self.kernel.feature_map(queries, p2, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        psi_k = self.kernel.feature_map(keys, p1, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        gamma_feat = self.kernel.gamma_map(gamma, p3, self.eta).reshape(T, B, self.head_num, self.feature_dim)

        cos_step, sin_step = self._cached_trig(T, device, inputs.dtype)
        tick = tick.to(dtype=cos_step.dtype, device=device)
        omegas = self.omegas.to(device=cos_step.device, dtype=cos_step.dtype)
        cos_tick = torch.cos(tick.view(1, B, 1) * omegas.view(1, 1, -1))
        sin_tick = torch.sin(tick.view(1, B, 1) * omegas.view(1, 1, -1))
        cos_terms = cos_tick * cos_step.view(T, 1, self.r) - sin_tick * sin_step.view(T, 1, self.r)

        gated_values = values * beta
        gated_keys = psi_k * gamma_feat

        cos_expanded = cos_terms.unsqueeze(-1).unsqueeze(-1)
        values_osc = gated_values.unsqueeze(2) * cos_expanded
        keys_osc = gated_keys.unsqueeze(2) * cos_expanded

        if self.reset_hidden_on_terminate:
            term_mask = (1 - terminations.float()).unsqueeze(-1).unsqueeze(-1)
            discount_gamma = (1 - gamma_feat) * term_mask
            discount_beta = (1 - beta) * term_mask
        else:
            discount_gamma = 1 - gamma_feat
            discount_beta = 1 - beta

        discount_gamma_r = discount_gamma.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
        discount_beta_r = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)

        final_keys = discounted_sum(tilde_k_prev, keys_osc, discount_gamma_r)
        final_values = discounted_sum(tilde_v_prev, values_osc, discount_beta_r)
        final_s = discounted_sum(s_prev, gated_keys, discount_gamma)

        keys_dot_queries = torch.einsum("tbrhD,tbhD->tbrh", final_keys, phi_q)
        kv = torch.einsum("tbrhd,tbrh->tbhd", final_values, keys_dot_queries)

        norm = torch.einsum("tbhD,tbhD->tbh", final_s, phi_q)
        attn_out = kv / (2 * self.r * norm.unsqueeze(-1) + self.eps)

        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        output = self.dropout(self.out_proj(attn_out))

        new_tick = tick + T
        new_tilde_k = final_keys[-1].detach()
        new_tilde_v = final_values[-1].detach()
        new_s = final_s[-1].detach()

        return output, (new_tilde_k, new_tilde_v, new_s, new_tick)

    def _cached_trig(
        self, timesteps: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (timesteps, device, dtype)
        if key not in self._cos_cache:
            steps = torch.arange(1, timesteps + 1, device=device, dtype=dtype)
            angles = steps.unsqueeze(-1) * self.omegas.to(device=device, dtype=dtype)
            cos_vals = torch.cos(angles)
            sin_vals = torch.sin(angles)
            self._cos_cache[key] = cos_vals
            self._sin_cache[key] = sin_vals
        return self._cos_cache[key], self._sin_cache[key]

    @staticmethod
    def initialize_memory(
        batch_size: int,
        head_num: int,
        head_dim: int,
        eta: int,
        r: int,
        device: Optional[torch.device] = None,
        kernel: Optional[AGaLiTeKernelConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = torch.device("cpu")

        kernel_conf = kernel or AGaLiTeKernelConfig()
        feature_dim = kernel_conf.feature_dim(head_dim, eta)

        tilde_k = torch.zeros(batch_size, r, head_num, feature_dim, device=device)
        tilde_v = torch.zeros(batch_size, r, head_num, head_dim, device=device)
        s = torch.zeros(batch_size, head_num, feature_dim, device=device)
        tick = torch.zeros(batch_size, device=device)
        return tilde_k, tilde_v, s, tick


class AGaLiTeTransformerLayer(nn.Module):
    """Single AGaLiTe transformer layer using shared GRU-style gating."""

    def __init__(
        self,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        kernel: AGaLiTeKernelConfig,
        *,
        use_input_proj: bool = False,
        gru_bias: float = 2.0,
        reset_hidden_on_terminate: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.use_input_proj = use_input_proj

        if use_input_proj:
            self.input_proj = nn.Linear(d_model, d_model)
            nn.init.orthogonal_(self.input_proj.weight, gain=math.sqrt(2))
            nn.init.constant_(self.input_proj.bias, 0)

        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.attention = AGaLiTeAttentionLayer(
            input_dim=d_model,
            head_dim=d_head,
            head_num=n_heads,
            eta=eta,
            r=r,
            kernel=kernel,
            dropout=dropout,
            reset_hidden_on_terminate=reset_hidden_on_terminate,
        )
        self._head_num = n_heads
        self._head_dim = d_head
        self._eta = eta
        self._r = r
        self._kernel = kernel

        self.gate1 = FusedGRUGating(d_model, bias=gru_bias)
        self.gate2 = FusedGRUGating(d_model, bias=gru_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.ffc = nn.Sequential(
            nn.Linear(d_model, d_ffc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffc, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)

        for module in self.ffc:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple
    ) -> Tuple[torch.Tensor, Tuple]:
        x = F.relu(self.input_proj(inputs)) if self.use_input_proj else inputs

        ln1 = self.ln1(x)
        attn_out, new_memory = self.attention(ln1, terminations, memory)
        attn_out = self.attn_dropout(F.relu(attn_out))
        gated = self.gate1(x, attn_out)

        ln2 = self.ln2(gated)
        ffc_out = self.ff_dropout(F.relu(self.ffc(ln2)))
        out = self.gate2(gated, ffc_out)

        return out, new_memory

    def initialize_memory(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple:
        return self.attention.initialize_memory(
            batch_size=batch_size,
            head_num=self._head_num,
            head_dim=self._head_dim,
            eta=self._eta,
            r=self._r,
            device=device,
            kernel=self._kernel,
        )


__all__ = ["AGaLiTeAttentionLayer", "AGaLiTeTransformerLayer"]
