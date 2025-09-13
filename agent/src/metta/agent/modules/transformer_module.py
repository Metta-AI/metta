"""
Transformer module with GTrXL-style stabilization optimized for parallel processing.

This is the stable baseline transformer that processes entire BPTT trajectories
at once, using stabilization techniques from GTrXL and optimizations
for processing thousands of environments/agents in parallel.

Key features:
- GRU-style gating for training stability
- Fused projections for memory efficiency
- Batched operations for parallel environment/agent processing
- Pre-normalization for better gradient flow
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = 0.1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe * self.scale_factor
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum positional encoding length {self.pe.size(0)}")
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class FusedGRUGating(nn.Module):
    """Fused GRU gating mechanism for efficient batch processing."""

    def __init__(self, d_model: int, bg: float = 0.5):
        """Initialize GRU gating.

        Args:
            d_model: Model dimension
            bg: Bias for update gate (higher values favor identity mapping for GTrXL)
        """
        super().__init__()
        self.gate_proj = nn.Linear(2 * d_model, 3 * d_model, bias=False)
        self.bg = nn.Parameter(torch.full([d_model], bg))

        nn.init.orthogonal_(self.gate_proj.weight, gain=1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply GRU gating with fused operations.

        Args:
            x: Residual stream (T, B*num_agents, d_model)
            y: Transformed input (T, B*num_agents, d_model)

        Returns:
            Gated output
        """
        gates = self.gate_proj(torch.cat([x, y], dim=-1))
        gates = gates.reshape(*gates.shape[:-1], 3, -1)

        r = torch.sigmoid(gates[..., 0, :])
        z = torch.sigmoid(gates[..., 1, :] - self.bg)
        h = torch.tanh(gates[..., 2, :] * r)

        return (1 - z) * x + z * h


class MultiHeadSelfAttention(nn.Module):
    """Fused multi-head self-attention optimized for batch processing."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_causal_mask: bool = True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_causal_mask = use_causal_mask

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass optimized for large batches.

        Args:
            x: Input of shape (T, B, d_model) where B = num_envs * num_agents
            mask: Optional attention mask

        Returns:
            Output of shape (T, B, d_model)
        """
        T, B, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(T, B, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 3, 1, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(self.n_heads * B, T, self.d_k)
        k = k.reshape(self.n_heads * B, T, self.d_k)
        v = v.reshape(self.n_heads * B, T, self.d_k)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.bmm(attn_weights, v)

        out = out.reshape(self.n_heads, B, T, self.d_k)
        out = out.permute(2, 1, 0, 3).reshape(T, B, self.d_model)

        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with GTrXL-style stabilization and parallel optimizations."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.use_gating = use_gating

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, use_causal_mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

        if use_gating:
            self.gate1 = FusedGRUGating(d_model, bg=2.0)
            self.gate2 = FusedGRUGating(d_model, bg=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process batch of sequences from multiple environments/agents.

        GTrXL architecture with Identity Map Reordering:
        1. LayerNorm(x) -> Attention -> GRU Gate(x, attn)
        2. LayerNorm(h1) -> FFN -> GRU Gate(h1, ffn)

        Key insight: Pre-normalization creates identity path from input to output
        """
        attn_out = self.attention(self.norm1(x))

        if self.use_gating:
            h1 = self.gate1(x, attn_out)
        else:
            h1 = x + attn_out

        ff_out = self.feed_forward(self.norm2(h1))

        if self.use_gating:
            h2 = self.gate2(h1, ff_out)
        else:
            h2 = h1 + ff_out

        return h2


class TransformerModule(nn.Module):
    """
    GTrXL (Gated Transformer-XL) implementation optimized for reinforcement learning.

    Key features:
    - Memory mechanism for handling sequences beyond context window
    - GRU-style gating for training stability
    - Pre-normalization for better gradient flow
    - Gradient stopping through memory
    - Fused projections for memory efficiency
    - Batched operations for parallel environment/agent processing
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        memory_len: int = 64,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        """Initialize the GTrXL transformer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length (should be >= BPTT horizon)
            memory_len: Length of memory to maintain from previous segments
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating (GTrXL-style)
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.memory_len = memory_len
        self.use_gating = use_gating

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.use_input_proj = True
        if self.use_input_proj:
            self.input_proj = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
            nn.init.constant_(self.input_proj.bias, 0)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, use_causal_mask, use_gating) for _ in range(n_layers)]
        )

        self.output_norm = nn.LayerNorm(d_model)

        if use_gating:
            self._init_gates_for_identity()

        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"GTrXL initialized: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, memory_len={memory_len}, use_gating={use_gating}"
        )

    def _init_gates_for_identity(self):
        """Initialize gates to favor identity mapping at start of training.

        GTrXL gating initialization:
        - Orthogonal weight init (gain=1.0) for stability
        - Higher bias values (bg=2.0) to favor identity mapping initially
        - This creates a clear identity path from input to output
        """
        pass

    def forward(self, x: torch.Tensor, memory: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with GTrXL memory mechanism.

        Args:
            x: Input of shape (T, B, d_model) where B = num_envs * num_agents
            memory: Dictionary containing memory from previous segments

        Returns:
            output: Tensor of same shape as input
            new_memory: Updated memory dictionary
        """
        if memory is None:
            memory = self.initialize_memory(1)

        if x.dim() == 3:
            T, B, D = x.shape
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            T, B, D = x.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        past_memory = memory.get("hidden_states")
        if past_memory is not None and past_memory[0].size(1) != B:
            memory = self.initialize_memory(B)
            past_memory = memory.get("hidden_states")
            if past_memory is not None:
                past_memory = [mem.to(x.device) for mem in past_memory]
                memory["hidden_states"] = past_memory

        if self.use_input_proj:
            x = self.input_proj(x)
            x = F.relu(x)

        x = self.positional_encoding(x)

        new_memory_states = []

        current_hidden = x
        for i, layer in enumerate(self.layers):
            layer_memory = past_memory[i] if past_memory is not None else None

            if layer_memory is not None:
                layer_memory = layer_memory.detach()
                layer_memory = layer_memory.to(current_hidden.device)
                extended_input = torch.cat([layer_memory, current_hidden], dim=0)
            else:
                extended_input = current_hidden

            layer_output = layer(extended_input)

            current_hidden = layer_output[-T:]

            if self.memory_len > 0:
                memory_to_store = layer_output[-self.memory_len :].detach()
                new_memory_states.append(memory_to_store)

        output = self.output_norm(current_hidden)

        new_memory = {"hidden_states": new_memory_states if new_memory_states else None}

        return output, new_memory

    def forward_chunked(self, x: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
        """Process very long sequences in chunks to manage memory.

        Useful when T * B * d_model is very large.

        Args:
            x: Input of shape (T, B, d_model)
            chunk_size: Process sequences in chunks of this size

        Returns:
            Output of shape (T, B, d_model)
        """
        T, B, _ = x.shape

        if T <= chunk_size:
            out, _ = self.forward(x)
            return out

        outputs = []
        overlap = chunk_size // 4

        for i in range(0, T, chunk_size - overlap):
            chunk_end = min(i + chunk_size, T)
            chunk = x[i:chunk_end]

            chunk_out, _ = self.forward(chunk)

            if i > 0 and overlap > 0:
                blend_region = min(overlap, len(outputs[-1]))
                alpha = torch.linspace(0, 1, blend_region, device=x.device).reshape(-1, 1, 1)
                outputs[-1][-blend_region:] = (1 - alpha) * outputs[-1][-blend_region:] + alpha * chunk_out[
                    :blend_region
                ]
                chunk_out = chunk_out[blend_region:]

            outputs.append(chunk_out)

        return torch.cat(outputs, dim=0)

    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize GTrXL memory for the transformer.

        Args:
            batch_size: Batch size (num_envs * num_agents)

        Returns:
            Memory dictionary with initialized hidden states for each layer
        """
        if self.memory_len <= 0:
            return {"hidden_states": None}

        memory_states = []
        device = next(self.parameters()).device
        for _ in range(self.n_layers):
            layer_memory = torch.zeros(self.memory_len, batch_size, self.d_model, device=device)
            memory_states.append(layer_memory)

        return {"hidden_states": memory_states}
