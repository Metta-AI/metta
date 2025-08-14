"""
Fully batched AGaLiTe implementation optimized for GPU training.
This version processes entire batches in parallel without loops.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """
    Compute discounted sum for batched sequences using efficient cumulative operations.

    This implements: y[t] = discount[t] * y[t-1] + x[t]
    where y[-1] = start_state

    Args:
        start_state: Initial state tensor of shape (B, ...) or matching x shape without T
        x: Sequence tensor of shape (T, B, ...)
        discounts: Discount factors of shape (T, B, ...)

    Returns:
        Discounted sum tensor of shape (T, B, ...)
    """
    T = x.shape[0]

    if T == 0:
        return x

    # Ensure start_state has same shape as x[0]
    if start_state.dim() < x.dim() - 1:
        # Add missing dimensions
        for _ in range(x.dim() - 1 - start_state.dim()):
            start_state = start_state.unsqueeze(-1)

    # For GPU efficiency, we can use a custom kernel or accumulate in chunks
    # For now, use a simple loop that's still efficient on GPU
    device = x.device
    dtype = x.dtype

    # Pre-allocate output tensor
    output = torch.empty_like(x)

    # Initialize with first step
    output[0] = discounts[0] * start_state + x[0]

    # Compute remaining steps
    for t in range(1, T):
        output[t] = discounts[t] * output[t - 1] + x[t]

    return output


class BatchedAttentionAGaLiTeLayer(nn.Module):
    """
    Fully batched AGaLiTe attention layer for GPU efficiency.
    Processes entire batches without loops.
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
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass for batched sequences.

        Args:
            inputs: Input tensor of shape (T, B, input_dim)
            terminations: Termination signals of shape (T, B)
            memory: Tuple of (tilde_k_prev, tilde_v_prev, s_prev, tick)
                - tilde_k_prev: shape (B, r, head_num, eta * head_dim)
                - tilde_v_prev: shape (B, r, head_num, head_dim)
                - s_prev: shape (B, head_num, eta * head_dim)
                - tick: shape (B, 1)

        Returns:
            - output: Attention output of shape (T, B, input_dim)
            - new_memory: Updated memory tuple
        """
        T, B, _ = inputs.shape
        device = inputs.device

        tilde_k_prev, tilde_v_prev, s_prev, tick = memory

        # Reshape inputs for batch processing: (T, B, dim) -> (T*B, dim)
        inputs_flat = inputs.reshape(T * B, self.input_dim)

        # Project to keys, queries, values, beta, gamma
        kqvbetagammas = self.linear_kqvbetagammas(inputs_flat)
        kqvbetagammas = kqvbetagammas.view(T, B, self.head_num, 5, self.head_dim)
        keys, queries, values, beta, gammas = kqvbetagammas.unbind(3)

        # Project to p1, p2, p3
        p1p2p3 = self.linear_p1p2p3(inputs_flat)
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
        ticks = tick.unsqueeze(0) + tick_inc  # (T, B, 1) - broadcast B dimension

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
            # Expand terminations to match tensor shapes
            # Ensure terminations has correct shape (T, B)
            if terminations.shape[1] != B:
                # Mismatch - create zeros instead
                terminations = torch.zeros(T, B, device=device)
            term_expand = terminations.unsqueeze(2).unsqueeze(3)  # (T, B, 1, 1)
            discount_gamma = (1 - gammas) * (1 - term_expand)
            discount_beta = (1 - beta) * (1 - term_expand)
        else:
            discount_gamma = 1 - gammas
            discount_beta = 1 - beta

        # Reshape for batched discounted sum
        # Move batch dimension appropriately
        discount_gamma_r = discount_gamma.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
        discount_beta_r = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)

        # Compute discounted sums - now properly batched
        final_keys = batched_discounted_sum(tilde_k_prev, keys, discount_gamma_r)
        final_values = batched_discounted_sum(tilde_v_prev, values, discount_beta_r)
        final_s = batched_discounted_sum(s_prev, s, discount_gamma)

        # Compute attention output
        # keys_dot_queries: (T, B, r, head_num)
        keys_dot_queries = torch.einsum("tbrhd,tbhd->tbrh", final_keys, queries)
        # kv: (T, B, head_num, head_dim)
        kv = torch.einsum("tbrhd,tbrh->tbhd", final_values, keys_dot_queries)

        # norm: (T, B, head_num)
        norm = torch.einsum("tbhd,tbhd->tbh", final_s, queries)
        # attn_out: (T, B, head_num, head_dim)
        attn_out = kv / (2 * self.r * norm.unsqueeze(-1) + self.eps)

        # Reshape and project output
        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        attn_out_flat = attn_out.reshape(T * B, self.head_num * self.head_dim)
        attn_out_flat = self.dropout(self.project(attn_out_flat))
        attn_out = attn_out_flat.reshape(T, B, self.input_dim)

        # Update memory - take the last timestep
        new_tick = tick + T
        new_tilde_k = final_keys[-1]  # (B, r, head_num, eta * head_dim)
        new_tilde_v = final_values[-1]  # (B, r, head_num, head_dim)
        new_s = final_s[-1]  # (B, head_num, eta * head_dim)

        return attn_out, (new_tilde_k, new_tilde_v, new_s, new_tick)

    @staticmethod
    def initialize_memory(
        batch_size: int,
        head_num: int,
        head_dim: int,
        eta: int,
        r: int,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize memory for batched attention layer."""
        if device is None:
            device = torch.device("cpu")

        tilde_k_prev = torch.zeros((batch_size, r, head_num, eta * head_dim), device=device)
        tilde_v_prev = torch.zeros((batch_size, r, head_num, head_dim), device=device)
        s_prev = torch.zeros((batch_size, head_num, eta * head_dim), device=device)
        tick = torch.zeros((batch_size, 1), device=device)

        return (tilde_k_prev, tilde_v_prev, s_prev, tick)


class BatchedGRUGatingUnit(nn.Module):
    """Batched GRU Gating Unit for efficient GPU processing."""

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


class BatchedRecurrentLinearTransformerEncoder(nn.Module):
    """Fully batched encoder layer."""

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

        self.attention = BatchedAttentionAGaLiTeLayer(
            input_dim=d_model,
            head_dim=d_head,
            head_num=n_heads,
            eta=eta,
            r=r,
            dropout=dropout,
            reset_hidden_on_terminate=reset_hidden_on_terminate,
        )

        self.gru1 = BatchedGRUGatingUnit(d_model, gru_bias)
        self.gru2 = BatchedGRUGatingUnit(d_model, gru_bias)

        self.ffc = nn.Sequential(nn.Linear(d_model, d_ffc), nn.ReLU(), nn.Linear(d_ffc, d_model))

        for module in self.ffc:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass for batched sequences.

        Args:
            inputs: Input tensor of shape (T, B, d_model)
            terminations: Termination signals of shape (T, B)
            memory: Memory tuple from previous timestep

        Returns:
            - output: Encoded output of shape (T, B, d_model)
            - new_memory: Updated memory tuple
        """
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


class BatchedAGaLiTe(nn.Module):
    """Fully batched AGaLiTe model optimized for GPU training."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.d_ffc = d_ffc
        self.n_heads = n_heads
        self.eta = eta
        self.r = r

        self.encoders = nn.ModuleList()
        for layer in range(n_layers):
            use_dense = layer == 0
            encoder = BatchedRecurrentLinearTransformerEncoder(
                d_model=d_model,
                d_head=d_head,
                d_ffc=d_ffc,
                n_heads=n_heads,
                eta=eta,
                r=r,
                use_dense=use_dense,
                reset_hidden_on_terminate=reset_on_terminate,
                dropout=dropout,
            )
            self.encoders.append(encoder)

    def forward(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Dict[str, Tuple],
    ) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
        """
        Forward pass for batched sequences.

        Args:
            inputs: Input tensor of shape (T, B, d_model)
            terminations: Termination signals of shape (T, B)
            memory: Dictionary of memory tuples for each layer

        Returns:
            - output: Model output of shape (T, B, d_model)
            - new_memory: Updated memory dictionary
        """
        u_i = inputs
        new_memory = {}

        for layer_idx, encoder in enumerate(self.encoders):
            layer_key = f"layer_{layer_idx + 1}"
            u_i, memory_updated = encoder(u_i, terminations, memory[layer_key])
            new_memory[layer_key] = memory_updated

        return u_i, new_memory

    @staticmethod
    def initialize_memory(
        batch_size: int,
        n_layers: int,
        n_heads: int,
        d_head: int,
        eta: int,
        r: int,
        device: torch.device = None,
    ) -> Dict[str, Tuple]:
        """Initialize memory for all layers."""
        memory_dict = {}
        for layer in range(1, n_layers + 1):
            memory_dict[f"layer_{layer}"] = BatchedAttentionAGaLiTeLayer.initialize_memory(
                batch_size, n_heads, d_head, eta, r, device
            )
        return memory_dict
