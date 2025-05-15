import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import override

from metta.agent.lib.metta_layer import LayerBase


class LSTM(LayerBase):
    """
    LSTM layer that handles tensor reshaping and state management automatically.

    This class wraps a PyTorch LSTM with proper tensor shape handling, making it easier
    to integrate LSTMs into neural network policies without dealing with complex tensor
    manipulations. It handles reshaping inputs/outputs, manages hidden states, and ensures
    consistent tensor dimensions throughout the forward pass.

    The layer processes tensors of shape [B, T, ...] or [B, ...], where:
    - B is the batch size
    - T is an optional time dimension

    It reshapes inputs appropriately for the LSTM, processes them through the network,
    and reshapes outputs back to the expected format, while also managing the LSTM's
    hidden state.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, obs_shape, hidden_size, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        self.hidden_size = hidden_size
        self.num_layers = self._nn_params["num_layers"]
        self._out_tensor_shape = [self.hidden_size]

    @override
    def _make_net(self) -> nn.Module:
        net = nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1)  # torch's default is uniform (gain is an int)

        return net

    @torch.compile(disable=True)  # Dynamo doesn't support compiling LSTMs
    @override
    def _forward(self, td: TensorDict) -> TensorDict:
        x = td["x"]
        hidden = td[self._sources[0]["name"]]

        # Get LSTM states separately
        lstm_h = td.get("lstm_h", None)
        lstm_c = td.get("lstm_c", None)

        # Prepare state tuple for PyTorch LSTM
        state = None
        if lstm_h is not None and lstm_c is not None:
            state = (lstm_h, lstm_c)

        # Validate input shapes
        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            raise ValueError("Invalid input tensor shape", x.shape)

        # Determine batch and time dimensions
        if x_n == space_n + 1:
            B, T = x_shape[0], 1
        elif x_n == space_n + 2:
            B, T = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        # Validate state shape consistency
        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B, (
                f"State batch dimension mismatch. Expected {B}, got {state[0].shape[1]}"
            )
        assert hidden.shape == (B * T, self._in_tensor_shapes[0][0]), (
            f"Hidden feature dimension mismatch. Expected {(B * T, self._in_tensor_shapes[0][0])}, got {hidden.shape}"
        )

        # Reshape for LSTM which expects [seq_len, batch, features]
        hidden = hidden.reshape(B, T, self._in_tensor_shapes[0][0])
        hidden = hidden.transpose(0, 1)  # [T, B, features]

        # Forward pass through LSTM
        hidden, (new_h, new_c) = self._net(hidden, state)

        # Reshape back to original format
        hidden = hidden.transpose(0, 1)  # [B, T, hidden_size]
        hidden = hidden.reshape(B * T, self.hidden_size)

        # Store results in TensorDict
        td[self._name] = hidden
        td["lstm_h"] = new_h.detach()  # Store hidden state separately
        td["lstm_c"] = new_c.detach()  # Store cell state separately

        return td
