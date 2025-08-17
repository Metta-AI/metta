"""
Improved full-context transformer with better initialization and positional encoding.

Key improvements:
1. Proper transformer initialization (smaller gains)
2. Learnable positional embeddings option
3. Better attention scaling
4. Layer normalization placement optimization
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedPositionalEncoding(nn.Module):
    """Improved positional encoding with scaling and learnable option."""
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 10000, 
        dropout: float = 0.1,
        scale_factor: float = 0.1,  # Scale down positional encoding
        learnable: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = scale_factor
        self.learnable = learnable
        
        if learnable:
            # Learnable positional embeddings (like BERT)
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)
        else:
            # Sinusoidal positional encoding with proper scaling
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # Scale down to prevent overwhelming input features
            pe = pe * scale_factor
            self.register_buffer("pe", pe.unsqueeze(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add scaled positional encoding to input."""
        seq_len = x.size(0)
        if self.learnable:
            x = x + self.pe[:seq_len]
        else:
            x = x + self.pe[:seq_len]
        return self.dropout(x)


class ImprovedGRUGating(nn.Module):
    """Improved GRU gating with better initialization."""
    
    def __init__(self, d_model: int, bg: float = 0.5):  # Reduced bias
        super().__init__()
        # Use smaller initialization for transformer components
        self.gate_proj = nn.Linear(2 * d_model, 3 * d_model, bias=False)
        self.bg = nn.Parameter(torch.full([d_model], bg))
        
        # Smaller initialization for transformers
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=1.0)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply GRU gating with proper scaling."""
        gates = self.gate_proj(torch.cat([x, y], dim=-1))
        gates = gates.reshape(*gates.shape[:-1], 3, -1)
        
        r = torch.sigmoid(gates[..., 0, :])  # Reset gate
        z = torch.sigmoid(gates[..., 1, :] - self.bg)  # Update gate
        h = torch.tanh(gates[..., 2, :] * r)  # Candidate
        
        return (1 - z) * x + z * h


class ImprovedMultiHeadAttention(nn.Module):
    """Improved multi-head attention with better scaling."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        attention_dropout: float = 0.1  # Separate attention dropout
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_causal_mask = use_causal_mask
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Improved scaling factor
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Xavier initialization for transformers (gain=1.0)
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with improved attention computation."""
        T, B, _ = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(T, B, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 3, 1, 0, 4)  # (3, n_heads, B, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Efficient attention computation
        q = q.reshape(self.n_heads * B, T, self.d_k)
        k = k.reshape(self.n_heads * B, T, self.d_k)
        v = v.reshape(self.n_heads * B, T, self.d_k)
        
        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        
        # Apply causal mask if needed
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float("-inf"))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax and dropout on attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.bmm(attn_weights, v)
        
        # Reshape back
        out = out.reshape(self.n_heads, B, T, self.d_k)
        out = out.permute(2, 1, 0, 3).reshape(T, B, self.d_model)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class ImprovedTransformerBlock(nn.Module):
    """Improved transformer block with better normalization and initialization."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
        activation: str = "gelu"  # Use GELU like modern transformers
    ):
        super().__init__()
        self.use_gating = use_gating
        
        # Pre-normalization (like modern transformers)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Core components
        self.attention = ImprovedMultiHeadAttention(
            d_model, n_heads, dropout, use_causal_mask, attention_dropout=dropout
        )
        
        # Choose activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.ReLU()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Initialize feed-forward with Xavier
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)
        
        # Optional gating mechanisms
        if use_gating:
            self.gate1 = ImprovedGRUGating(d_model, bg=0.5)
            self.gate2 = ImprovedGRUGating(d_model, bg=0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-normalization."""
        # Pre-norm attention
        attn_out = self.attention(self.norm1(x))
        
        if self.use_gating:
            x = self.gate1(x, attn_out)
        else:
            x = x + attn_out
        
        # Pre-norm feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        
        if self.use_gating:
            x = self.gate2(x, ff_out)
        else:
            x = x + ff_out
        
        return x


class ImprovedFullContextTransformer(nn.Module):
    """
    Improved full-context transformer with better initialization and features.
    
    Key improvements:
    1. Proper weight initialization for transformers
    2. Learnable positional embeddings option
    3. Better positional encoding scaling
    4. Pre-normalization pattern
    5. GELU activation option
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = False,  # Often not needed for RL
        use_gating: bool = True,
        learnable_pos: bool = True,  # Use learnable positional embeddings
        pos_scale: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_gating = use_gating
        
        # Improved positional encoding
        self.positional_encoding = ImprovedPositionalEncoding(
            d_model, max_seq_len, dropout, 
            scale_factor=pos_scale,
            learnable=learnable_pos
        )
        
        # Optional input projection with smaller initialization
        self.input_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.constant_(self.input_proj.bias, 0)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ImprovedTransformerBlock(
                d_model, n_heads, d_ff, dropout, 
                use_causal_mask, use_gating, activation
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.output_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Initialize transformer parameters with smaller values
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with transformer-appropriate values."""
        # Apply scaled initialization to all parameters
        for p in self.parameters():
            if p.dim() > 1:
                # Use Xavier uniform with gain=1.0 for transformers
                nn.init.xavier_uniform_(p, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through improved transformer."""
        # Handle input shapes
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add time dimension
        
        # Input projection and activation
        x = self.input_proj(x)
        x = F.gelu(x)  # Use GELU for input activation
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Remove time dimension if it was added
        if x.size(0) == 1:
            x = x.squeeze(0)
        
        return x
    
    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory (empty for this transformer)."""
        return {}