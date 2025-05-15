from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from typing_extensions import override

from metta.agent.lib.metta_layer import LayerBase


class LSTM(LayerBase):
    """
    LSTM layer with PackedSequence optimizations that maintains serializability.

    This class wraps a PyTorch LSTM with proper tensor shape handling, making it easier
    to integrate LSTMs into neural network policies without dealing with complex tensor
    manipulations. It handles reshaping inputs/outputs, manages hidden states, and ensures
    consistent tensor dimensions throughout the forward pass.

    The layer supports variable-length sequences through PackedSequence when seq_lengths
    are provided, improving efficiency for padded sequences.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, obs_shape, hidden_size, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        self.hidden_size = hidden_size
        self.num_layers = self._nn_params.get("num_layers", 1)

        # For PackedSequence, batch_first affects packing/unpacking
        self._nn_params["batch_first"] = True

        self._out_tensor_shape = [self.hidden_size]

    def _make_net(self) -> nn.Module:
        net = nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)

        return net

    def _pack_sequence(
        self, batch: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, nn.utils.rnn.PackedSequence]:
        """Function to sort and pack a sequence."""
        # Sort sequences by length
        sort_idx = torch.argsort(lengths, descending=True)
        unsort_idx = torch.argsort(sort_idx)

        sorted_lengths = lengths[sort_idx]
        sorted_batch = batch[sort_idx]

        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            sorted_batch, sorted_lengths.tolist(), batch_first=True, enforce_sorted=True
        )

        # Return sorted batch, indices for later unpacking, and packed sequence
        return sorted_batch, unsort_idx, sorted_lengths, packed

    def _unpack_sequence(self, packed_output: nn.utils.rnn.PackedSequence, unsort_idx: torch.Tensor) -> torch.Tensor:
        """Function to unpack and unsort outputs."""
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden_unsorted = hidden[unsort_idx]
        return hidden_unsorted

    @torch.compile(disable=True)
    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        x = data["x"]
        hidden = data[self._sources[0]["name"]]

        # Get sequence lengths if available
        seq_lengths = data.get("seq_lengths", None)

        # Get LSTM states
        lstm_h = data.get("lstm_h", None)
        lstm_c = data.get("lstm_c", None)

        state = None
        if lstm_h is not None and lstm_c is not None:
            state = (lstm_h, lstm_c)

        # Process shapes
        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)

        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, T = x_shape[0], 1
        elif x_n == space_n + 2:
            B, T = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B, (
                f"State batch dimension mismatch. Expected {B}, got {state[0].shape[1]}"
            )

        # Reshape hidden features for LSTM
        hidden = hidden.reshape(B, T, self._in_tensor_shapes[0][0])

        # Use PackedSequence for variable-length sequences
        if seq_lengths is not None and T > 1:  # Only pack if we have multiple timesteps
            # Ensure seq_lengths is on CPU for sorting operations
            lengths = seq_lengths.cpu()

            # Sort by sequence length
            sort_idx = torch.argsort(lengths, descending=True)
            unsort_idx = torch.argsort(sort_idx)

            sorted_lengths = lengths[sort_idx]
            sorted_batch = hidden[sort_idx]

            # Sort states if present
            sorted_state = None
            if state is not None:
                h, c = state
                sorted_h = h.index_select(1, sort_idx)
                sorted_c = c.index_select(1, sort_idx)
                sorted_state = (sorted_h, sorted_c)

            # Pack sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                sorted_batch, sorted_lengths.tolist(), batch_first=True, enforce_sorted=True
            )

            # Process through LSTM
            packed_output, (new_h, new_c) = self._net(packed, sorted_state)

            # Unpack sequence
            hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            # Unsort outputs
            hidden = hidden[unsort_idx]
            new_h = new_h.index_select(1, unsort_idx)
            new_c = new_c.index_select(1, unsort_idx)
        else:
            # Standard processing for fixed-length or single-step sequences
            hidden, (new_h, new_c) = self._net(hidden, state)

        # Reshape output to match expected format
        hidden = hidden.reshape(B * T, self.hidden_size)

        # Store results
        data[self._name] = hidden
        data["lstm_h"] = new_h.detach()
        data["lstm_c"] = new_c.detach()

        return data
