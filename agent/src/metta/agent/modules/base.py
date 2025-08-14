"""Base classes and utilities for PyTorch agents."""

import torch.nn as nn
import pufferlib.models


def init_layer(layer: nn.Module, std: float = 1.0) -> nn.Module:
    """Initialize layer weights to match ComponentPolicy initialization."""
    nn.init.orthogonal_(layer.weight, gain=std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class LSTMBase(pufferlib.models.LSTMWrapper):
    """Base class for LSTM agents with proper initialization.
    
    Fixes LSTM bias initialization to 1.0 (matching ComponentPolicy) instead 
    of pufferlib's default of 0.0. This is critical for proper LSTM learning,
    especially for forget gates.
    """
    
    def __init__(self, env, policy, input_size: int = 128, hidden_size: int = 128):
        super().__init__(env, policy, input_size, hidden_size)
        
        # Fix LSTM bias initialization to match ComponentPolicy
        # ComponentPolicy initializes LSTM biases to 1, but pufferlib's LSTMWrapper initializes to 0
        # This is critical for proper LSTM learning, especially for forget gates
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1.0)
        
        # Initialize placeholders for action tensors that MettaAgent will set
        self.action_index_tensor = None
        self.cum_action_max_params = None