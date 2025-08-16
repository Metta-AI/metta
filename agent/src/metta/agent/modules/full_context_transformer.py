"""
Full-context transformer that processes entire BPTT trajectories.

This module implements a transformer decoder that can view the entire BPTT horizon
at once, using all observations as context to predict the next action.
Unlike recurrent transformers (AGaLiTe/GTrXL), this uses standard self-attention
without memory compression or approximations.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
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
        return x + self.pe[:x.size(0)]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
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
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        # Transpose back: (seq_len, batch, d_model)
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_len, batch_size, self.d_model)
        
        output = self.out_linear(context)
        return output


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (seq_len, batch, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class FullContextTransformer(nn.Module):
    """
    Full-context transformer that processes entire BPTT sequences.
    
    This transformer views the entire trajectory at once and uses all
    observations as context to produce outputs. It includes both actor
    and critic heads for RL applications.
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
    ):
        """
        Initialize the full-context transformer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking (for autoregressive)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_causal_mask = use_causal_mask
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            Causal mask tensor
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def forward(
        self,
        x: torch.Tensor,
        terminations: Optional[torch.Tensor] = None,
        memory: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            terminations: Optional termination flags (seq_len, batch)
            memory: Optional memory dict (unused, kept for compatibility)
        
        Returns:
            output: Transformed tensor of shape (seq_len, batch, d_model)
            memory: Updated memory (returns empty dict for compatibility)
        """
        seq_len, batch_size, _ = x.shape
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Create causal mask if needed
        mask = None
        if self.use_causal_mask:
            mask = self.create_causal_mask(seq_len, x.device)
            # Expand for batch and heads
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(batch_size, self.n_heads, -1, -1)
        
        # Handle terminations by masking attention
        if terminations is not None:
            # Create termination mask: don't attend to positions after termination
            term_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            for b in range(batch_size):
                for t in range(seq_len):
                    if terminations[t, b]:
                        # Mask out all positions after termination
                        if t < seq_len - 1:
                            term_mask[t+1:, :t+1] = False
            
            term_mask = term_mask.unsqueeze(0).unsqueeze(0)
            term_mask = term_mask.expand(batch_size, self.n_heads, -1, -1)
            
            if mask is not None:
                mask = mask & term_mask
            else:
                mask = term_mask
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Return empty memory dict for compatibility
        return x, {}
    
    def initialize_memory(self, batch_size: int) -> Dict:
        """
        Initialize memory for the transformer.
        
        This transformer doesn't use recurrent memory, but we provide
        this method for compatibility with the infrastructure.
        
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
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Full-context transformer
        self.transformer = FullContextTransformer(
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        hidden, memory = self.transformer(hidden, terminations, memory)
        
        # Decode to actions and values
        hidden_flat = hidden.reshape(-1, self.hidden_dim)
        logits, values = self.decode_actions(hidden_flat)
        
        # Reshape outputs
        logits = logits.reshape(seq_len, batch_size, -1)
        values = values.reshape(seq_len, batch_size)
        
        return logits, values, memory
    
    def initialize_memory(self, batch_size: int) -> Dict:
        """Initialize memory (empty for this transformer)."""
        return self.transformer.initialize_memory(batch_size)