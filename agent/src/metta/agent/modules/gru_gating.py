"""
GRU Gating Unit for AGaLiTe architecture.
Based on the AGaLiTe paper's use of GRU units for gradient flow control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GRUGatingUnit(nn.Module):
    """
    GRU-style gating unit for combining two inputs.
    This helps with gradient flow and prevents vanishing gradients.
    
    Based on the AGaLiTe paper equation:
    y = g * x + (1 - g) * y_prev
    where g is a learned gate based on both inputs.
    """
    
    def __init__(self, d_model: int, bias: float = 2.0):
        """
        Args:
            d_model: Model dimension
            bias: Initial bias for the gate (positive = prefer new input)
        """
        super().__init__()
        self.d_model = d_model
        
        # Gate computation layers
        self.w_g = nn.Linear(d_model * 2, d_model, bias=False)
        self.u_g = nn.Linear(d_model, d_model, bias=False)
        
        # Bias parameter (learnable)
        self.bias = nn.Parameter(torch.tensor(bias))
        
        # Initialize weights
        nn.init.orthogonal_(self.w_g.weight)
        nn.init.orthogonal_(self.u_g.weight)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply GRU gating to combine two inputs.
        
        Args:
            x: Previous/residual input (T, B, d_model) or (B, d_model)
            y: New input to gate (T, B, d_model) or (B, d_model)
        
        Returns:
            Gated combination of x and y
        """
        # Compute gate
        # Concatenate inputs
        concat = torch.cat([x, y], dim=-1)
        
        # Compute gating scores
        g = self.w_g(concat) + self.u_g(x) + self.bias
        g = torch.sigmoid(g)
        
        # Apply gating
        output = g * x + (1 - g) * y
        
        return output


class SimpleGRUGatingUnit(nn.Module):
    """
    Simplified GRU gating unit with fewer parameters.
    Used in some AGaLiTe variants for efficiency.
    """
    
    def __init__(self, d_model: int, bias: float = 2.0):
        super().__init__()
        self.d_model = d_model
        
        # Single projection for gate
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        
        # Initialize
        nn.init.orthogonal_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, bias)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Simple gating: g = sigmoid(W[x;y] + b)
        output = g * x + (1 - g) * y
        """
        concat = torch.cat([x, y], dim=-1)
        g = torch.sigmoid(self.gate_proj(concat))
        return g * x + (1 - g) * y