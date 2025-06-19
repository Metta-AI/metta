"""
Simple decoder for the LSTM memory.
"""

import torch
import torch.nn as nn


class LocationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Linear(2 * 2 * 128, 2)  # num_layers * #(lstm_c, lstm_h) * dim

    def forward(self, lstm_h, lstm_c) -> torch.Tensor:
        # lstm_h: (2, batch_size, 128)
        # lstm_c: (2, batch_size, 128)
        # so first we need to move the batch dimension to the first dimension
        lstm_h = lstm_h.transpose(0, 1)
        lstm_c = lstm_c.transpose(0, 1)

        # then we reshape the tensors to (batch_size, 2 * 2 * 128)
        lstm_h = lstm_h.reshape(lstm_h.shape[0], -1)
        lstm_c = lstm_c.reshape(lstm_c.shape[0], -1)

        input = torch.cat([lstm_h, lstm_c], dim=-1)
        print(f"Input shape: {input.shape}")

        # then we flatten the batch dimension
        input = input.view(input.shape[0], -1)

        assert input.shape[1:] == (2 * 2 * 128,), f"Input shape: {input.shape}"

        return self.probe(input)
