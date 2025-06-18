"""
Simple decoder for the LSTM memory.
"""

import torch
import torch.nn as nn


class LocationDecoder(nn.Module):
    def __init__(self, lstm_c_dim, lstm_h_dim, num_agents):
        super().__init__()
        probe = nn.Linear(lstm_c_dim + lstm_h_dim, num_agents)

    def forward(self, lstm_memory) -> torch.Tensor:
        pass
