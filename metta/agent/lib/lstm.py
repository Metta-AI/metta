from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import override

from metta.agent.lib.metta_layer import LayerBase


class LSTM(LayerBase):
    """
    LSTM layer with TorchScript and PackedSequence optimizations.
    """

    def __init__(self, obs_shape, hidden_size, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)
        self.hidden_size = hidden_size
        self.num_layers = self._nn_params["num_layers"]
        self._out_tensor_shape = [self.hidden_size]

        # For PackedSequence, batch_first affects packing/unpacking
        self._nn_params["batch_first"] = True

        # Flag to track if we're using scripted functions
        self._use_scripted_funcs = False

    @override
    def _make_net(self) -> nn.Module:
        net = nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1)

        # Try to script the network itself
        try:
            net = torch.jit.script(net)
        except Exception as e:
            print(f"Warning: Failed to script LSTM network: {e}")

        # Create scripted helper functions for sequence packing operations
        try:
            # Script the sequence operations for better performance
            self._scripted_pack = torch.jit.script(self._pack_sequence)
            self._scripted_unpack = torch.jit.script(self._unpack_sequence)
            self._use_scripted_funcs = True
        except Exception as e:
            print(f"Warning: Failed to script sequence operations: {e}")

        return net

    @staticmethod
    @torch.jit.script
    def _pack_sequence(
        batch: torch.Tensor, lengths: torch.Tensor, batch_first: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TorchScript-compatible function to sort and pack a sequence."""
        # Sort sequences by length
        sort_idx = torch.argsort(lengths, descending=True)
        unsort_idx = torch.argsort(sort_idx)

        sorted_lengths = lengths[sort_idx]
        sorted_batch = batch[sort_idx]

        # Return sorted batch and indices for later unpacking
        return sorted_batch, unsort_idx, sorted_lengths

    @staticmethod
    @torch.jit.script
    def _unpack_sequence(
        hidden: torch.Tensor, h: torch.Tensor, c: torch.Tensor, unsort_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TorchScript-compatible function to unsort outputs."""
        hidden_unsorted = hidden[unsort_idx]
        h_unsorted = h.index_select(1, unsort_idx)
        c_unsorted = c.index_select(1, unsort_idx)
        return hidden_unsorted, h_unsorted, c_unsorted

    @torch.compile(disable=True)
    @override
    def _forward(self, td: TensorDict) -> TensorDict:
        x = td["x"]
        hidden = td[self._sources[0]["name"]]

        # Get sequence lengths if available
        seq_lengths = td.get("seq_lengths", None)

        # Get LSTM states
        lstm_h = td.get("lstm_h", None)
        lstm_c = td.get("lstm_c", None)

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

            if self._use_scripted_funcs:
                # Use scripted functions for better performance
                sorted_batch, unsort_idx, sorted_lengths = self._scripted_pack(hidden, lengths)

                if state is not None:
                    h, c = state
                    sorted_h = h.index_select(1, unsort_idx.to(h.device))
                    sorted_c = c.index_select(1, unsort_idx.to(c.device))
                    sorted_state = (sorted_h, sorted_c)
                else:
                    sorted_state = None

                # Pack sequence
                packed = nn.utils.rnn.pack_padded_sequence(
                    sorted_batch, sorted_lengths.tolist(), batch_first=True, enforce_sorted=True
                )

                # Process through LSTM
                packed_output, (new_h, new_c) = self._net(packed, sorted_state)

                # Unpack sequence
                hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

                # Unsort outputs
                hidden, new_h, new_c = self._scripted_unpack(hidden, new_h, new_c, unsort_idx.to(hidden.device))
            else:
                # Fallback to standard implementation
                sort_idx = torch.argsort(lengths, descending=True)
                unsort_idx = torch.argsort(sort_idx)

                sorted_lengths = lengths[sort_idx]
                sorted_batch = hidden[sort_idx]

                sorted_state = None
                if state is not None:
                    h, c = state
                    sorted_h = h.index_select(1, sort_idx)
                    sorted_c = c.index_select(1, sort_idx)
                    sorted_state = (sorted_h, sorted_c)

                packed = nn.utils.rnn.pack_padded_sequence(
                    sorted_batch, sorted_lengths.tolist(), batch_first=True, enforce_sorted=True
                )

                packed_output, (new_h, new_c) = self._net(packed, sorted_state)

                hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

                hidden = hidden[unsort_idx]
                new_h = new_h.index_select(1, unsort_idx)
                new_c = new_c.index_select(1, unsort_idx)
        else:
            # Standard processing for fixed-length or single-step sequences
            hidden, (new_h, new_c) = self._net(hidden, state)

        # Reshape output to match expected format
        hidden = hidden.reshape(B * T, self.hidden_size)

        # Store results
        td[self._name] = hidden
        td["lstm_h"] = new_h.detach()
        td["lstm_c"] = new_c.detach()

        return td
