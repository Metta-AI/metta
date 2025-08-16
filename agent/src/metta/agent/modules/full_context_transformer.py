"""
Full-context transformer with GTrXL-style stabilization for BPTT sequences.

This module implements a transformer decoder that processes entire BPTT trajectories
at once, using state-of-the-art stabilization techniques from GTrXL (Gated Transformer-XL).
Key improvements include:
- GRU-style gating instead of residual connections for stability
- Pre-normalization for better gradient flow
- Identity initialization for learning Markovian policies initially
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GRUGatingMechanism(nn.Module):
    """GRU-style gating mechanism from GTrXL for stabilizing transformer training."""
    
    def __init__(self, d_model: int, bg: float = 2.0):
        """Initialize GRU gating.
        
        Args:
            d_model: Model dimension
            bg: Bias for update gate (higher = more identity-like at start)
        """
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.full([d_model], bg))  # Learnable bias
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply GRU gating.
        
        Args:
            x: Input (residual stream)
            y: Transformed input (from attention/FFN)
            
        Returns:
            Gated output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))  # Reset gate
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)  # Update gate (with bias)
        h = self.tanh(self.Wg(y) + self.Ug(r * x))  # Candidate
        return (1 - z) * x + z * h  # Interpolate


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_causal_mask = use_causal_mask
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (seq_len, batch, d_model)
        """
        seq_len, batch_size, _ = x.shape
        
        # Linear projections in batch from d_model => h x d_k
        q = self.q_linear(x).view(seq_len, batch_size, self.n_heads, self.d_k)
        k = self.k_linear(x).view(seq_len, batch_size, self.n_heads, self.d_k)
        v = self.v_linear(x).view(seq_len, batch_size, self.n_heads, self.d_k)
        
        # Transpose for attention: (batch, n_heads, seq_len, d_k)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask if needed
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        # Transpose back: (seq_len, batch, d_model)
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_len, batch_size, self.d_model)
        
        output = self.out_linear(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer block with GTrXL-style stabilization."""
    
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
        
        # Pre-normalization (key for stability)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Core components
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, use_causal_mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Gating mechanisms (instead of residual connections)
        if use_gating:
            self.gate1 = GRUGatingMechanism(d_model)
            self.gate2 = GRUGatingMechanism(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention
        normalized = self.norm1(x)
        attn_out = self.attention(normalized)
        
        # Gate or residual
        if self.use_gating:
            x = self.gate1(x, attn_out)
        else:
            x = x + attn_out
        
        # Pre-norm + feed-forward
        normalized = self.norm2(x)
        ff_out = self.feed_forward(normalized)
        
        # Gate or residual
        if self.use_gating:
            x = self.gate2(x, ff_out)
        else:
            x = x + ff_out
        
        return x


class FullContextTransformer(nn.Module):
    """Full-context transformer with GTrXL-style stabilization."""
    
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
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating (GTrXL-style)
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_gating = use_gating
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Input layer norm (stabilizes initial training)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer layers with gating
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_causal_mask, use_gating)
            for _ in range(n_layers)
        ])
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)
        
        # Optional: Identity map initialization (helps with learning Markovian policies initially)
        if use_gating:
            self._init_gates_for_identity()
    
    def _init_gates_for_identity(self):
        """Initialize gates to favor identity mapping at start of training."""
        for layer in self.layers:
            if hasattr(layer, 'gate1'):
                # Initialize to favor passing through original input
                nn.init.xavier_uniform_(layer.gate1.Wz.weight, gain=0.1)
                nn.init.xavier_uniform_(layer.gate1.Uz.weight, gain=0.1)
                nn.init.xavier_uniform_(layer.gate2.Wz.weight, gain=0.1)
                nn.init.xavier_uniform_(layer.gate2.Uz.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (T, B, d_model) or (B, T, d_model)
        
        Returns:
            Output tensor of same shape as input
        """
        # Handle both (T, B, d_model) and (B, T, d_model) formats
        if x.dim() == 3 and x.shape[0] != x.shape[1]:
            # Assume (T, B, d_model) if T != B
            T, B, _ = x.shape
            needs_transpose = False
        else:
            # Assume (B, T, d_model)
            B, T, _ = x.shape
            x = x.transpose(0, 1)  # Convert to (T, B, d_model)
            needs_transpose = True
        
        # Normalize input (stabilization)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Restore original format if needed
        if needs_transpose:
            x = x.transpose(0, 1)  # Convert back to (B, T, d_model)
        
        return x
    
    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory for the transformer.
        
        This transformer doesn't use recurrent memory.
        
        Args:
            batch_size: Batch size
            
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
        """
        Initialize the policy network.
        
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
        
        # Observation encoder (with layer norm for stability)
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Full-context transformer with GTrXL improvements
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
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def encode_observations(self, obs: torch.Tensor, state: Optional[Dict] = None) -> torch.Tensor:
        """
        Encode observations to hidden representation.
        
        Args:
            obs: Observations tensor
            state: Optional state dict (unused)
            
        Returns:
            Encoded observations
        """
        return self.obs_encoder(obs)
    
    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode hidden states to action logits and values.
        
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
        self,
        obs: torch.Tensor,
        terminations: Optional[torch.Tensor] = None,
        memory: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Full forward pass through the policy.
        
        Args:
            obs: Observations of shape (seq_len, batch, obs_dim) or (batch, seq_len, obs_dim)
            terminations: Optional termination flags
            memory: Optional memory dict
            
        Returns:
            logits: Action logits
            values: Value estimates
            memory: Updated memory dict
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