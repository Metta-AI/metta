"""
Enhanced AGaLiTe implementation with proper paper features.

This module implements both GaLiTe (exact) and AGaLiTe (approximated) modes
following the paper specification more closely while preserving batch optimizations.
"""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig
from metta.agent.components.agalite_optimized import discounted_sum


class EnhancedGRUGatingUnit(nn.Module):
    """Enhanced GRU Gating Unit with improved initialization and control."""

    def __init__(self, input_dim: int, bg: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)

        # Main gating components
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)

        # Learnable bias parameter for forget gate
        self.bgp = nn.Parameter(torch.full((input_dim,), bg))

        # Enhanced initialization following paper recommendations
        for module in [self.Wr, self.Ur, self.Wz, self.Uz, self.Wg, self.Ug]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Enhanced GRU forward with dropout.
        """
        # Reset gate
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))

        # Update gate with learnable bias
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)

        # Candidate hidden state
        h_tilde = torch.tanh(self.Wg(y) + self.Ug(r * x))
        h_tilde = self.dropout(h_tilde)

        # Final output
        return (1 - z) * x + z * h_tilde


class GaLiTeAttentionLayer(nn.Module):
    """
    Exact GaLiTe (Gated Linear Transformers) attention layer.

    This implements the non-approximated version using exact linear attention
    with element-wise gating but without oscillatory approximation.
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        head_num: int,
        eta: int,
        kernel: AGaLiTeKernelConfig,
        dropout: float = 0.0,
        eps: float = 1e-6,
        reset_hidden_on_terminate: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.eta = eta
        self.eps = eps
        self.reset_hidden_on_terminate = reset_hidden_on_terminate
        self.kernel = kernel
        self.feature_dim = self.kernel.feature_dim(self.head_dim, self.eta)

        # Feature map projections with proper dimensions
        self.q_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)

        # Gating parameters
        self.beta_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.gamma_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)

        # Feature map parameters for outer products
        self.p1_proj = nn.Linear(input_dim, head_num * eta, bias=False)
        self.p2_proj = nn.Linear(input_dim, head_num * eta, bias=False)
        self.p3_proj = nn.Linear(input_dim, head_num * eta, bias=False)

        # Output projection
        self.out_proj = nn.Linear(head_num * head_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize all projections
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.gamma_proj,
            self.p1_proj,
            self.p2_proj,
            self.p3_proj,
        ]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

        nn.init.orthogonal_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        GaLiTe attention forward pass.
        """
        T, B, _ = inputs.shape

        kv_state_prev, norm_state_prev = memory

        # Project to attention components
        queries = self.q_proj(inputs).view(T, B, self.head_num, self.head_dim)
        keys = self.k_proj(inputs).view(T, B, self.head_num, self.head_dim)
        values = self.v_proj(inputs).view(T, B, self.head_num, self.head_dim)

        # Gating parameters
        beta = torch.sigmoid(self.beta_proj(inputs).view(T, B, self.head_num, self.head_dim))
        gamma = torch.sigmoid(self.gamma_proj(inputs).view(T, B, self.head_num, self.head_dim))

        # Feature map parameters
        p1 = self.p1_proj(inputs).view(T, B, self.head_num, self.eta)
        p2 = self.p2_proj(inputs).view(T, B, self.head_num, self.eta)
        p3 = self.p3_proj(inputs).view(T, B, self.head_num, self.eta)

        # Apply feature maps with outer products (paper eq. 4-6)
        phi_q = self.kernel.feature_map(queries, p2, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        psi_k = self.kernel.feature_map(keys, p1, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        gamma_feat = self.kernel.gamma_map(gamma, p3, self.eta).reshape(T, B, self.head_num, self.feature_dim)

        # Apply gating
        gated_values = values * beta  # (T, B, head_num, head_dim)
        gated_keys = psi_k * gamma_feat  # (T, B, head_num, feature_dim)

        # Prepare discount factors for discounted sum
        if self.reset_hidden_on_terminate:
            term_mask = (1 - terminations.float()).unsqueeze(-1).unsqueeze(-1)  # (T, B, 1, 1)
            discount_gamma = (1 - gamma_feat) * term_mask  # (T, B, head_num, head_dim * eta)
        else:
            discount_gamma = 1 - gamma_feat

        # Update states using discounted sum
        # KV state: (B, head_num, head_dim * eta, head_dim)
        kv_updates = torch.einsum("tbhd,tbhD->tbhDd", gated_values, gated_keys)
        new_kv_state = discounted_sum(kv_state_prev, kv_updates, discount_gamma.unsqueeze(-1))

        # Normalization state: (B, head_num, head_dim * eta)
        new_norm_state = discounted_sum(norm_state_prev, gated_keys, discount_gamma)

        # Compute attention output
        # Output = KV @ φ(q) / (φ(q)^T @ norm + ε)
        attn_num = torch.einsum("tbhDd,tbhD->tbhd", new_kv_state, phi_q)
        attn_denom = torch.einsum("tbhD,tbhD->tbh", phi_q, new_norm_state)

        attn_out = attn_num / (attn_denom.unsqueeze(-1) + self.eps)

        # Project to output
        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        output = self.dropout(self.out_proj(attn_out))

        # Return output and memory (detach for memory efficiency)
        final_kv_state = new_kv_state[-1]
        final_norm_state = new_norm_state[-1]

        return output, (final_kv_state, final_norm_state)

    @staticmethod
    def initialize_memory(
        batch_size: int,
        head_num: int,
        head_dim: int,
        eta: int,
        device: Optional[torch.device] = None,
        kernel: Optional[AGaLiTeKernelConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize memory for GaLiTe attention layer."""
        if device is None:
            device = torch.device("cpu")

        kernel_conf = kernel or AGaLiTeKernelConfig()
        feature_dim = kernel_conf.feature_dim(head_dim, eta)

        kv_state = torch.zeros(batch_size, head_num, feature_dim, head_dim, device=device)
        norm_state = torch.zeros(batch_size, head_num, feature_dim, device=device)

        return (kv_state, norm_state)


class AGaLiTeAttentionLayer(nn.Module):
    """
    AGaLiTe (Approximated Gated Linear Transformers) attention layer.

    This implements the approximated version using trigonometric functions
    to approximate the Kronecker delta function for efficiency.
    """

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
    ):
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

        # All projections (same as GaLiTe but with r-dimensional expansion)
        self.q_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)

        # Gating parameters
        self.beta_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)
        self.gamma_proj = nn.Linear(input_dim, head_num * head_dim, bias=False)

        # Feature map parameters
        self.p1_proj = nn.Linear(input_dim, head_num * eta, bias=False)
        self.p2_proj = nn.Linear(input_dim, head_num * eta, bias=False)
        self.p3_proj = nn.Linear(input_dim, head_num * eta, bias=False)

        # Output projection
        self.out_proj = nn.Linear(head_num * head_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute oscillatory frequencies (ω ∈ [-π, π])
        omegas = torch.linspace(-math.pi, math.pi, r)
        self.register_buffer("omegas", omegas)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.gamma_proj,
            self.p1_proj,
            self.p2_proj,
            self.p3_proj,
        ]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

        nn.init.orthogonal_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        AGaLiTe attention forward pass with oscillatory approximation.
        """
        T, B, _ = inputs.shape
        device = inputs.device

        tilde_k_prev, tilde_v_prev, s_prev, tick = memory

        # Project to attention components
        queries = self.q_proj(inputs).view(T, B, self.head_num, self.head_dim)
        keys = self.k_proj(inputs).view(T, B, self.head_num, self.head_dim)
        values = self.v_proj(inputs).view(T, B, self.head_num, self.head_dim)

        # Gating parameters
        beta = torch.sigmoid(self.beta_proj(inputs).view(T, B, self.head_num, self.head_dim))
        gamma = torch.sigmoid(self.gamma_proj(inputs).view(T, B, self.head_num, self.head_dim))

        # Feature map parameters
        p1 = self.p1_proj(inputs).view(T, B, self.head_num, self.eta)
        p2 = self.p2_proj(inputs).view(T, B, self.head_num, self.eta)
        p3 = self.p3_proj(inputs).view(T, B, self.head_num, self.eta)

        phi_q = self.kernel.feature_map(queries, p2, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        psi_k = self.kernel.feature_map(keys, p1, self.eta).reshape(T, B, self.head_num, self.feature_dim)
        gamma_feat = self.kernel.gamma_map(gamma, p3, self.eta).reshape(T, B, self.head_num, self.feature_dim)

        # Update tick and compute oscillatory terms
        tick_base = tick.view(1, B)
        tick_inc = torch.arange(1, T + 1, device=device, dtype=tick.dtype).view(T, 1)
        ticks = tick_inc + tick_base  # (T, B)

        # Compute oscillatory components: cos(ω · t)
        cos_terms = torch.cos(ticks.unsqueeze(-1) * self.omegas.view(1, 1, -1))  # (T, B, r)

        # Apply gating and expand with oscillatory terms
        gated_values = values * beta  # (T, B, head_num, head_dim)
        gated_keys = psi_k * gamma_feat  # (T, B, head_num, head_dim * eta)

        # Expand with oscillatory terms
        cos_expanded = cos_terms.unsqueeze(-1).unsqueeze(-1)
        values_osc = gated_values.unsqueeze(2) * cos_expanded  # (T, B, r, head_num, head_dim)
        keys_osc = gated_keys.unsqueeze(2) * cos_expanded  # (T, B, r, head_num, head_dim * eta)

        # Prepare discount factors
        if self.reset_hidden_on_terminate:
            term_mask = (1 - terminations.float()).unsqueeze(-1).unsqueeze(-1)
            discount_gamma = (1 - gamma_feat) * term_mask
            discount_beta = (1 - beta) * term_mask
        else:
            discount_gamma = 1 - gamma_feat
            discount_beta = 1 - beta

        # Expand discount factors for r dimension
        discount_gamma_r = discount_gamma.unsqueeze(2).expand(-1, -1, self.r, -1, -1)
        discount_beta_r = discount_beta.unsqueeze(2).expand(-1, -1, self.r, -1, -1)

        # Update oscillatory states using discounted sum
        final_keys = discounted_sum(tilde_k_prev, keys_osc, discount_gamma_r)
        final_values = discounted_sum(tilde_v_prev, values_osc, discount_beta_r)
        final_s = discounted_sum(s_prev, gated_keys, discount_gamma)

        # Compute attention output using oscillatory approximation
        keys_dot_queries = torch.einsum("tbrhD,tbhD->tbrh", final_keys, phi_q)  # (T, B, r, head_num)
        kv = torch.einsum("tbrhd,tbrh->tbhd", final_values, keys_dot_queries)  # (T, B, head_num, head_dim)

        # Normalization with oscillatory terms
        norm = torch.einsum("tbhD,tbhD->tbh", final_s, phi_q)  # (T, B, head_num)
        attn_out = kv / (2 * self.r * norm.unsqueeze(-1) + self.eps)

        # Project to output
        attn_out = attn_out.reshape(T, B, self.head_num * self.head_dim)
        output = self.dropout(self.out_proj(attn_out))

        # Update memory (detach for efficiency)
        new_tick = tick + T
        new_tilde_k = final_keys[-1].detach()
        new_tilde_v = final_values[-1].detach()
        new_s = final_s[-1].detach()

        return output, (new_tilde_k, new_tilde_v, new_s, new_tick)

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
        """Initialize memory for AGaLiTe attention layer."""
        if device is None:
            device = torch.device("cpu")

        kernel_conf = kernel or AGaLiTeKernelConfig()
        feature_dim = kernel_conf.feature_dim(head_dim, eta)

        tilde_k = torch.zeros(batch_size, r, head_num, feature_dim, device=device)
        tilde_v = torch.zeros(batch_size, r, head_num, head_dim, device=device)
        s = torch.zeros(batch_size, head_num, feature_dim, device=device)
        tick = torch.zeros(batch_size, device=device)

        return (tilde_k, tilde_v, s, tick)


class EnhancedTransformerEncoder(nn.Module):
    """
    Enhanced Transformer encoder with mode selection between GaLiTe and AGaLiTe.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        kernel: AGaLiTeKernelConfig,
        mode: Literal["galite", "agalite"] = "agalite",
        use_dense: bool = False,
        gru_bias: float = 2.0,
        reset_hidden_on_terminate: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.use_dense = use_dense
        self.kernel = kernel

        # Input embedding layer (used for first layer)
        if use_dense:
            self.emb_layer = nn.Linear(d_model, d_model)
            nn.init.orthogonal_(self.emb_layer.weight, gain=math.sqrt(2))
            nn.init.constant_(self.emb_layer.bias, 0)

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Attention layer (mode-dependent)
        if mode == "galite":
            self.attention = GaLiTeAttentionLayer(
                input_dim=d_model,
                head_dim=d_head,
                head_num=n_heads,
                eta=eta,
                kernel=kernel,
                dropout=dropout,
                reset_hidden_on_terminate=reset_hidden_on_terminate,
            )
        else:  # agalite
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

        # GRU gating units
        self.gru1 = EnhancedGRUGatingUnit(d_model, gru_bias, dropout)
        self.gru2 = EnhancedGRUGatingUnit(d_model, gru_bias, dropout)

        # Feedforward network
        self.ffc = nn.Sequential(
            nn.Linear(d_model, d_ffc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffc, d_model),
        )

        # Initialize feedforward weights
        for module in self.ffc:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """Enhanced transformer encoder forward pass."""

        # Input embedding (first layer only)
        if self.use_dense:
            inputs_enc = F.relu(self.emb_layer(inputs))
        else:
            inputs_enc = inputs

        # Pre-norm attention
        ln1_out = self.ln1(inputs_enc)
        attn_out, new_memory = self.attention(ln1_out, terminations, memory)
        attn_out = F.relu(attn_out)

        # First GRU gating (residual connection)
        gating1_out = self.gru1(inputs_enc, attn_out)

        # Pre-norm feedforward
        ln2_out = self.ln2(gating1_out)
        ffc_out = self.ffc(ln2_out)
        ffc_out = F.relu(ffc_out)

        # Second GRU gating (residual connection)
        out = self.gru2(gating1_out, ffc_out)

        return out, new_memory

    def initialize_memory(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple:
        """Initialize memory based on attention layer type."""
        if self.mode == "galite":
            return self.attention.initialize_memory(
                batch_size=batch_size,
                head_num=self.attention.head_num,
                head_dim=self.attention.head_dim,
                eta=self.attention.eta,
                device=device,
                kernel=self.kernel,
            )
        else:  # agalite
            return self.attention.initialize_memory(
                batch_size=batch_size,
                head_num=self.attention.head_num,
                head_dim=self.attention.head_dim,
                eta=self.attention.eta,
                r=self.attention.r,
                device=device,
                kernel=self.kernel,
            )
