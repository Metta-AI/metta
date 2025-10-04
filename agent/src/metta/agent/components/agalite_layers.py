"""
AGaLiTe (Approximate Gated Linear Transformer) layers in PyTorch.
Ported from JAX implementation to PyTorch.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import optimized version - critical for performance
from metta.agent.components.agalite_optimized import discounted_sum


class GRUGatingUnit(nn.Module):
    """GRU Gating Unit as used in AGaLiTe."""

    def __init__(self, input_dim: int, bg: float = 2.0):
        super().__init__()
        self.input_dim = input_dim

        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bgp = nn.Parameter(torch.full((input_dim,), bg))

        # Initialize weights
        for module in [self.Wr, self.Ur, self.Wz, self.Uz, self.Wg, self.Ug]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Previous hidden state (T, B, dim) or (B, dim)
            y: Current input (T, B, dim) or (B, dim)
        Returns:
            Gated output with same shape as input
        """
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)
        h = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h


class AttentionAGaLiTeLayer(nn.Module):
    """
    AGaLiTe attention layer with linear attention, oscillatory components,
    and GRU-style gating.
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        head_num: int,
        eta: int,
        r: int,
        dropout: float = 0.0,
        eps: float = 1e-5,
        reset_hidden_on_terminate: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.eta = eta
        self.r = r
        self.eps = eps
        self.reset_hidden_on_terminate = reset_hidden_on_terminate

        # Linear projections
        self.linear_kqvbetagammas = nn.Linear(input_dim, head_num * head_dim * 5)
        self.linear_p1p2p3 = nn.Linear(input_dim, head_num * eta * 3)
        self.project = nn.Linear(head_num * head_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for module in [self.linear_kqvbetagammas, self.linear_p1p2p3, self.project]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass for AGaLiTe attention."""
        T, B, _ = inputs.shape
        device = inputs.device

        tilde_k_prev, tilde_v_prev, s_prev, tick = memory

        # Project to keys, queries, values, beta, gamma
        kqvbetagammas = self.linear_kqvbetagammas(inputs)  # (T, B, head_num * head_dim * 5)
        kqvbetagammas = kqvbetagammas.view(T, B, self.head_num, 5, self.head_dim)
        keys, queries, values, beta, gammas = kqvbetagammas.unbind(3)

        # Project to p1, p2, p3
        p1p2p3 = self.linear_p1p2p3(inputs)  # (T, B, head_num * eta * 3)
        p1p2p3 = p1p2p3.view(T, B, self.head_num, 3, self.eta)
        p1, p2, p3 = p1p2p3.unbind(3)

        # Compute feature mapped keys, queries, gammas
        # Shape: (T, B, head_num, eta * head_dim)
        keys = torch.einsum("tbhd,tbhn->tbhdn", F.relu(keys), F.relu(p1)).flatten(-2)
        queries = torch.einsum("tbhd,tbhn->tbhdn", F.relu(queries), F.relu(p2)).flatten(-2)
        gammas = torch.einsum("tbhd,tbhn->tbhdn", torch.sigmoid(gammas), torch.sigmoid(p3)).flatten(-2)

        beta = torch.sigmoid(beta)  # (T, B, head_num, head_dim)

        # Update tick and compute oscillatory terms
        tick_inc = torch.arange(1, T + 1, device=device).view(T, 1, 1)  # (T, 1, 1)
        ticks = tick.unsqueeze(0) + tick_inc  # (T, B, 1)

        omegas = torch.linspace(-math.pi, math.pi, self.r, device=device)
        occil = torch.cos(torch.einsum("tbi,j->tbj", ticks, omegas))  # (T, B, r)

        # Apply gating to values and keys
        values = values * beta  # (T, B, head_num, head_dim)
        values = torch.einsum("tbhd,tbr->tbrhd", values, occil)  # (T, B, r, head_num, head_dim)

        keys_gated = keys * gammas  # (T, B, head_num, eta * head_dim)
        s = keys_gated.clone()

        keys = torch.einsum("tbhd,tbr->tbrhd", keys_gated, occil)  # (T, B, r, head_num, eta * head_dim)

        # Compute discount factors
        if self.reset_hidden_on_terminate:
            term_expand = terminations.unsqueeze(2).unsqueeze(3).float()  # (T, B, 1, 1) - convert to float
            discount_gamma = (1 - gammas) * (1 - term_expand)
            discount_beta = (1 - beta) * (1 - term_expand)
        else:
            discount_gamma = 1 - gammas
            discount_beta = 1 - beta

        # Reshape for discounted sum
        discount_gamma_r = discount_gamma.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
        discount_beta_r = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)

        # Compute discounted sums
        final_keys = discounted_sum(tilde_k_prev, keys, discount_gamma_r)
        final_values = discounted_sum(tilde_v_prev, values, discount_beta_r)
        final_s = discounted_sum(s_prev, s, discount_gamma)

        # Compute attention output
        keys_dot_queries = torch.einsum("tbrhd,tbhd->tbrh", final_keys, queries)
        kv = torch.einsum("tbrhd,tbrh->tbhd", final_values, keys_dot_queries)

        norm = torch.einsum("tbhd,tbhd->tbh", final_s, queries)
        attn_out = kv / (2 * self.r * norm.unsqueeze(-1) + self.eps)

        # Reshape and project output
        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        attn_out = self.dropout(self.project(attn_out))

        # Update memory - take the last timestep
        # CRITICAL: Detach memory to prevent gradient accumulation across episodes
        new_tick = tick + T
        new_tilde_k = final_keys[-1].detach()  # (B, r, head_num, eta * head_dim)
        new_tilde_v = final_values[-1].detach()  # (B, r, head_num, head_dim)
        new_s = final_s[-1].detach()  # (B, head_num, eta * head_dim)

        return attn_out, (new_tilde_k, new_tilde_v, new_s, new_tick)

    @staticmethod
    def initialize_memory(
        batch_size: int, head_num: int, head_dim: int, eta: int, r: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize memory for AGaLiTe attention layer."""
        if device is None:
            device = torch.device("cpu")

        tilde_k_prev = torch.zeros((batch_size, r, head_num, eta * head_dim), device=device)
        tilde_v_prev = torch.zeros((batch_size, r, head_num, head_dim), device=device)
        s_prev = torch.zeros((batch_size, head_num, eta * head_dim), device=device)
        tick = torch.zeros((batch_size, 1), device=device)

        return (tilde_k_prev, tilde_v_prev, s_prev, tick)


class RecurrentLinearTransformerEncoder(nn.Module):
    """Recurrent Linear Transformer Encoder layer as used in AGaLiTe."""

    def __init__(
        self,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        use_dense: bool = False,
        gru_bias: float = 2.0,
        reset_hidden_on_terminate: bool = True,
        embedding_act: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_dense = use_dense
        self.embedding_act = embedding_act

        if use_dense:
            self.emb_layer = nn.Linear(d_model, d_model)
            nn.init.orthogonal_(self.emb_layer.weight, gain=math.sqrt(2))
            nn.init.constant_(self.emb_layer.bias, 0)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attention = AttentionAGaLiTeLayer(
            input_dim=d_model,
            head_dim=d_head,
            head_num=n_heads,
            eta=eta,
            r=r,
            dropout=dropout,
            reset_hidden_on_terminate=reset_hidden_on_terminate,
        )

        self.gru1 = GRUGatingUnit(d_model, gru_bias)
        self.gru2 = GRUGatingUnit(d_model, gru_bias)

        self.ffc = nn.Sequential(nn.Linear(d_model, d_ffc), nn.ReLU(), nn.Linear(d_ffc, d_model))

        for module in self.ffc:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass."""
        # Input embedding
        if self.use_dense:
            inputs_enc = self.emb_layer(inputs)
            if self.embedding_act:
                inputs_enc = F.relu(inputs_enc)
        else:
            inputs_enc = inputs

        # Layer norm + attention
        ln1_out = self.ln1(inputs_enc)
        attn_out, new_memory = self.attention(ln1_out, terminations, memory)
        attn_out = F.relu(attn_out)

        # First GRU gating
        gating1_out = self.gru1(inputs_enc, attn_out)

        # Layer norm + feed-forward
        ln2_out = self.ln2(gating1_out)
        ffc_out = self.ffc(ln2_out)
        ffc_out = F.relu(ffc_out)
        ffc_out = self.dropout(ffc_out)

        # Second GRU gating
        out = self.gru2(gating1_out, ffc_out)

        return out, new_memory
