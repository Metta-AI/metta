"""
Full-context transformer with GTrXL-style stabilization optimized for parallel processing.

This module implements a transformer decoder that processes entire BPTT trajectories
at once, using state-of-the-art stabilization techniques from GTrXL and optimizations
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
        # Scale factor to prevent positional encoding from overwhelming features
        self.scale_factor = 0.1

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Scale down positional encoding
        pe = pe * self.scale_factor
        # Shape: (max_len, 1, d_model) for broadcasting
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
            bg: Bias for update gate (reduced for better gradient flow)
        """
        super().__init__()
        # Fused projection for all gates (more efficient)
        self.gate_proj = nn.Linear(2 * d_model, 3 * d_model, bias=False)
        self.bg = nn.Parameter(torch.full([d_model], bg))

        # Initialize with smaller gain for transformer stability
        nn.init.orthogonal_(self.gate_proj.weight, gain=1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply GRU gating with fused operations.

        Args:
            x: Residual stream (T, B*num_agents, d_model)
            y: Transformed input (T, B*num_agents, d_model)

        Returns:
            Gated output
        """
        # Fused projection for all gates
        gates = self.gate_proj(torch.cat([x, y], dim=-1))
        gates = gates.reshape(*gates.shape[:-1], 3, -1)

        # Split and apply activations
        r = torch.sigmoid(gates[..., 0, :])  # Reset gate
        z = torch.sigmoid(gates[..., 1, :] - self.bg)  # Update gate with bias
        h = torch.tanh(gates[..., 2, :] * r)  # Candidate with reset

        return (1 - z) * x + z * h  # Interpolate


# Backward compatibility alias for old checkpoints
GRUGatingMechanism = FusedGRUGating


class MultiHeadSelfAttention(nn.Module):
    """Fused multi-head self-attention optimized for batch processing."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_causal_mask: bool = True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_causal_mask = use_causal_mask

        # Fused QKV projection (3x more efficient than separate projections)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Initialize weights with smaller gain for transformers
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

        # Fused QKV projection and reshape for multi-head
        qkv = self.qkv_proj(x)  # (T, B, 3*d_model)
        qkv = qkv.reshape(T, B, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 3, 1, 0, 4)  # (3, n_heads, B, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (n_heads, B, T, d_k)

        # Efficient batched attention computation
        # Merge heads and batch for parallel processing
        q = q.reshape(self.n_heads * B, T, self.d_k)
        k = k.reshape(self.n_heads * B, T, self.d_k)
        v = v.reshape(self.n_heads * B, T, self.d_k)

        # Scaled dot-product attention with batched operations
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (n_heads*B, T, T)

        # Apply causal mask if needed
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float("-inf"))

        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.bmm(attn_weights, v)  # (n_heads*B, T, d_k)

        # Reshape back
        out = out.reshape(self.n_heads, B, T, self.d_k)
        out = out.permute(2, 1, 0, 3).reshape(T, B, self.d_model)

        # Output projection
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

        # Post-normalization (matching AGaLiTe architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Core components
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, use_causal_mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # Use ReLU like AGaLiTe, not GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Initialize feed-forward with smaller gain for transformers
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

        # Gating mechanisms (instead of residual connections)
        if use_gating:
            self.gate1 = FusedGRUGating(d_model, bg=0.5)  # Reduced bias
            self.gate2 = FusedGRUGating(d_model, bg=0.5)  # Reduced bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process batch of sequences from multiple environments/agents.

        Following AGaLiTe architecture:
        1. LayerNorm -> Attention -> ReLU -> GRU Gate
        2. LayerNorm -> FFN -> ReLU -> GRU Gate
        """
        # Layer norm + attention (post-norm style like AGaLiTe)
        ln1_out = self.norm1(x)
        attn_out = self.attention(ln1_out)
        attn_out = F.relu(attn_out)  # ReLU activation after attention like AGaLiTe

        # First GRU gating
        if self.use_gating:
            gating1_out = self.gate1(x, attn_out)
        else:
            gating1_out = x + attn_out

        # Layer norm + feed-forward
        ln2_out = self.norm2(gating1_out)
        ff_out = self.feed_forward(ln2_out)
        ff_out = F.relu(ff_out)  # ReLU activation after FFN like AGaLiTe

        # Second GRU gating
        if self.use_gating:
            out = self.gate2(gating1_out, ff_out)
        else:
            out = gating1_out + ff_out

        return out


class FullContextTransformer(nn.Module):
    """
    Full-context transformer optimized for parallel processing across environments/agents.

    Key features:
    - Processes entire BPTT trajectories as context
    - Fused projections for memory efficiency
    - Batched operations for parallel environment/agent processing
    - GTrXL-style gating for stability
    - Pre-normalization for better gradient flow
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        """Initialize the transformer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length (should be >= BPTT horizon)
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating (GTrXL-style)
        """
        super().__init__()

        self.d_model = d_model
        self.use_gating = use_gating

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Optional input embedding layer (like AGaLiTe's use_dense)
        self.use_input_proj = True
        if self.use_input_proj:
            self.input_proj = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
            nn.init.constant_(self.input_proj.bias, 0)

        # Transformer layers with gating
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, use_causal_mask, use_gating) for _ in range(n_layers)]
        )

        # Output layer norm (applied at the end)
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize gates for identity mapping if using gating
        if use_gating:
            self._init_gates_for_identity()

        # Log successful initialization
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"FullContextTransformer initialized: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, use_gating={use_gating}"
        )

    def _init_gates_for_identity(self):
        """Initialize gates to favor identity mapping at start of training.

        The GRU gates are already initialized with:
        - Orthogonal weight init (gain=1.0) for stability
        - Bias bg=0.5 for update gate to allow gradient flow
        This provides better gradient flow for transformers.
        """
        # No additional modification needed - gates are properly initialized
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass processing sequences from multiple environments/agents.

        Args:
            x: Input of shape (T, B, d_model) where B = num_envs * num_agents
                or (B, T, d_model)

        Returns:
            Output of same shape as input
        """
        # Log input shape for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"FullContextTransformer.forward input shape: {x.shape}")

        # Handle both (T, B, d_model) and (B, T, d_model) formats
        if x.dim() == 3:
            # The TransformerWrapper should always give us (T, B, d_model)
            # where T is sequence length and B is batch size
            T, B, D = x.shape
            needs_transpose = False

            # Sanity check - if T is unreasonably large, something is wrong
            if T > self.positional_encoding.pe.size(0):
                logger.error(
                    f"Sequence length {T} exceeds max positional encoding length {self.positional_encoding.pe.size(0)}"
                )
                logger.error(f"Input shape: {x.shape}")
                # Try to handle it as (B, T, d_model) instead
                if B <= self.positional_encoding.pe.size(0):
                    logger.warning("Assuming input is (B, T, d_model) and transposing")
                    x = x.transpose(0, 1)
                    T, B, D = x.shape
                    needs_transpose = True
                else:
                    raise ValueError(f"Sequence length {T} exceeds maximum {self.positional_encoding.pe.size(0)}")
        elif x.dim() == 2:
            # Single timestep (B, d_model) - add time dimension
            x = x.unsqueeze(0)  # (1, B, d_model)
            T, B, D = x.shape
            needs_transpose = False
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        logger.debug(f"After reshaping: T={T}, B={B}, D={D}")

        # Optional input projection (like AGaLiTe's embedding layer)
        if self.use_input_proj:
            x = self.input_proj(x)
            x = F.relu(x)  # Activation after input projection

        # Add positional encoding
        x = self.positional_encoding(x)

        # Process through transformer layers
        # All environments/agents are processed in parallel
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.output_norm(x)

        # Restore original format if needed
        if needs_transpose:
            x = x.transpose(0, 1)
        elif x.size(0) == 1:
            # Remove time dimension if it was added for single timestep
            x = x.squeeze(0)

        return x

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
            return self.forward(x)

        # Process in chunks with overlap for context
        outputs = []
        overlap = chunk_size // 4  # 25% overlap

        for i in range(0, T, chunk_size - overlap):
            chunk_end = min(i + chunk_size, T)
            chunk = x[i:chunk_end]

            # Process chunk
            chunk_out = self.forward(chunk)

            # Handle overlap by blending
            if i > 0 and overlap > 0:
                # Blend with previous chunk's overlap region
                blend_region = min(overlap, len(outputs[-1]))
                alpha = torch.linspace(0, 1, blend_region, device=x.device).reshape(-1, 1, 1)
                outputs[-1][-blend_region:] = (1 - alpha) * outputs[-1][-blend_region:] + alpha * chunk_out[
                    :blend_region
                ]
                chunk_out = chunk_out[blend_region:]

            outputs.append(chunk_out)

        return torch.cat(outputs, dim=0)

    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory for the transformer.

        This transformer doesn't use recurrent memory, but we provide
        this method for compatibility with the infrastructure.

        Args:
            batch_size: Batch size (num_envs * num_agents)

        Returns:
            Empty memory dict
        """
        return {}


class FullContextTransformerPolicy(nn.Module):
    """
    Policy network using the full-context transformer.

    This combines the transformer with actor and critic heads
    for reinforcement learning.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        """Initialize the policy network.

        Args:
            observation_dim: Dimension of observations
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension (d_model for transformer)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Observation encoder (matching activation patterns)
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  # Use ReLU for consistency
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Initialize encoder with orthogonal init
        for module in self.obs_encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Full-context transformer
        self.transformer = FullContextTransformer(
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  # Use ReLU for consistency
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  # Use ReLU for consistency
            nn.Linear(hidden_dim, 1),
        )

        # Initialize actor and critic with orthogonal init
        for head in [self.actor, self.critic]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                    nn.init.constant_(module.bias, 0)

    def encode_observations(self, obs: torch.Tensor, state: Optional[Dict] = None) -> torch.Tensor:
        """Encode observations to hidden representation.

        Args:
            obs: Observations tensor
            state: Optional state dict (unused)

        Returns:
            Encoded observations
        """
        return self.obs_encoder(obs)

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden states to action logits and values.

        Args:
            hidden: Hidden state tensor

        Returns:
            logits: Action logits
            values: Value estimates
        """
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values

    def forward(
        self, obs: torch.Tensor, terminations: Optional[torch.Tensor] = None, memory: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Full forward pass through the policy.

        Args:
            obs: Observations of shape (seq_len, batch, obs_dim) or (batch, seq_len, obs_dim)
                 where batch = num_envs * num_agents
            terminations: Optional termination flags
            memory: Optional memory dict

        Returns:
            logits: Action logits
            values: Value estimates
            memory: Updated memory dict (empty for this transformer)
        """
        # Handle different input shapes
        if obs.dim() == 3 and obs.shape[1] != obs.shape[2]:
            # Likely (batch, seq_len, obs_dim), transpose to (seq_len, batch, obs_dim)
            obs = obs.transpose(0, 1)

        # Encode observations
        seq_len, batch_size = obs.shape[:2]
        obs_flat = obs.reshape(-1, obs.shape[-1])
        hidden = self.encode_observations(obs_flat)
        hidden = hidden.reshape(seq_len, batch_size, -1)

        # Pass through transformer
        hidden = self.transformer(hidden)

        # Decode to actions and values
        hidden_flat = hidden.reshape(-1, self.hidden_dim)
        logits, values = self.decode_actions(hidden_flat)

        # Reshape outputs
        logits = logits.reshape(seq_len, batch_size, -1)
        values = values.reshape(seq_len, batch_size)

        return logits, values, memory or {}

    def initialize_memory(self, batch_size: int) -> Dict:
        """Initialize memory (empty for this transformer)."""
        return self.transformer.initialize_memory(batch_size)
