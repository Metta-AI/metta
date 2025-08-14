"""
Base class for LSTM-based agents that fixes LSTM initialization.

This class extends pufferlib.models.LSTMWrapper and ensures LSTM biases
are initialized to 1.0 (matching ComponentPolicy behavior) instead of
pufferlib's default of 0.0. This is critical for proper LSTM learning,
especially for forget gates.
"""

import torch.nn as nn
import pufferlib.models


class LSTMBase(pufferlib.models.LSTMWrapper):
    """Base class for LSTM agents with proper initialization."""

    def __init__(self, env, policy, input_size=128, hidden_size=128):
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
