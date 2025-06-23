from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase
from metta.agent.lib.metta_module import MettaDict, MettaModule, UniqueInKeyMixin, UniqueOutKeyMixin


class LSTM(LayerBase):
    """
    LSTM layer that handles tensor reshaping and state management automatically.

    This class wraps a PyTorch LSTM with proper tensor shape handling, making it easier
    to integrate LSTMs into neural network policies without dealing with complex tensor
    manipulations. It handles reshaping inputs/outputs, manages hidden states, and ensures
    consistent tensor dimensions throughout the forward pass.

    The layer processes tensors of shape [B, TT, ...] or [B, ...], where:
    - B is the batch size
    - TT is an optional time dimension

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
        # self._out_tensor_shape = [hidden_size] # delete this
        self.num_layers = self._nn_params["num_layers"]

    def _make_net(self):
        self._out_tensor_shape = [self.hidden_size]
        net = nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)  # torch's default is uniform

        return net

    @torch.compile(disable=True)  # Dynamo doesn't support compiling LSTMs
    def _forward(self, td: TensorDict):
        x = td["x"]
        hidden = td[self._sources[0]["name"]]
        state = td["state"]

        if state is not None:
            h_state = state[: self.num_layers].contiguous()
            c_state = state[self.num_layers :].contiguous()
            state = (h_state, c_state)

        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        # Minimal production check for state batch size
        if state is not None:
            if state[0].shape[1] != B or state[1].shape[1] != B:
                raise ValueError(
                    f"LSTM state batch size mismatch: expected {B}, got h={state[0].shape[1]}, c={state[1].shape[1]}"
                )

        # Minimal production check for hidden shape
        expected_hidden_shape = (B * TT, self._in_tensor_shapes[0][0])
        if hidden.shape != expected_hidden_shape:
            raise ValueError(f"Hidden state shape {hidden.shape} does not match expected {expected_hidden_shape}")

        # Debug-only verbose logging
        if __debug__:
            print("[LEGACY LSTM] Input x shape:", x.shape, "sample:", x.flatten()[:5])
            if state is not None:
                print("[LEGACY LSTM] State[0] shape:", state[0].shape, "sample:", state[0].flatten()[:5])
                print("[LEGACY LSTM] State[1] shape:", state[1].shape, "sample:", state[1].flatten()[:5])
            else:
                print("[LEGACY LSTM] State: None")

        # These are the important ones
        B = td["_B_"]
        TT = td["_TT_"]
        hidden = rearrange(hidden, "(b t) h -> t b h", b=B, t=TT)

        hidden, state = self._net(hidden, state)

        if __debug__:
            print("[LEGACY LSTM] Output hidden shape:", hidden.shape, "sample:", hidden.flatten()[:5])
            if state is not None:
                print("[LEGACY LSTM] Output state[0] shape:", state[0].shape, "sample:", state[0].flatten()[:5])
                print("[LEGACY LSTM] Output state[1] shape:", state[1].shape, "sample:", state[1].flatten()[:5])
            else:
                print("[LEGACY LSTM] Output state: None")

        hidden = rearrange(hidden, "t b h -> (b t) h")

        if state is not None:
            state = tuple(s.detach() for s in state)
            state = torch.cat(state, dim=0)

        td[self._name] = hidden
        td["state"] = state

        return td


class MettaEncodedLSTM(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    """LSTM layer that processes encoded input sequences (under the key 'encoded_features') and maintains internal state via md.data[self.output_key]['state'].
    The input is expected to be an encoded/processed feature tensor, not a raw observation.
    Assumes B (batch size) and TT (time steps) are provided in the global metadata dict as md.data['global']['batch_size'] and md.data['global']['tt'].
    Shape validation is handled upstream.
    """

    def __init__(
        self,
        obs_shape: List[int],
        hidden_size: int,
        num_layers: int = 1,
        input_key: str = "encoded_features",
        output_key: str = "lstm_output",
    ):
        self.input_key = input_key
        self.output_key = output_key
        self._obs_shape = list(obs_shape)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__(
            in_keys=[input_key],
            out_keys=[output_key],
            input_features_shape=obs_shape,
            output_features_shape=[hidden_size],
        )
        self._net = nn.LSTM(input_size=obs_shape[0], hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        for name, param in self._net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)

    @torch.compile(disable=True)
    def _compute(self, md: MettaDict) -> dict:
        # B and TT are expected to be present in the global metadata dict
        B = md.data["global"]["batch_size"]
        TT = md.data["global"]["tt"]
        encoded_features = md.td[self.input_key]

        # Minimal production check for encoded features shape
        if encoded_features.shape[0] != B * TT:
            raise ValueError(
                f"encoded_features shape {encoded_features.shape} does not match expected [{B * TT}, H] with B={B}, TT={TT}"
            )

        # Namespace state under output_key
        state_dict = md.data.get(self.output_key, {})
        state = state_dict.get("state", None)
        if state is not None:
            split_size = self.num_layers
            state = (state[:split_size], state[split_size:])
            # Minimal production check for state shape
            if state[0].shape[1] != B or state[1].shape[1] != B:
                raise ValueError(
                    f"LSTM state batch size mismatch: expected {B}, got h={state[0].shape[1]}, c={state[1].shape[1]}"
                )

        # Debug-only verbose logging
        if __debug__:
            print(
                "[METTA LSTM] Input encoded_features shape:",
                encoded_features.shape,
                "sample:",
                encoded_features.flatten()[:5],
            )
            if state is not None:
                print("[METTA LSTM] State[0] shape:", state[0].shape, "sample:", state[0].flatten()[:5])
                print("[METTA LSTM] State[1] shape:", state[1].shape, "sample:", state[1].flatten()[:5])
            else:
                print("[METTA LSTM] State: None")

        encoded_features = rearrange(encoded_features, "(b t) h -> t b h", b=B, t=TT)
        output, state = self._net(encoded_features, state)

        if __debug__:
            print("[METTA LSTM] Output shape:", output.shape, "sample:", output.flatten()[:5])
            if state is not None:
                print("[METTA LSTM] Output state[0] shape:", state[0].shape, "sample:", state[0].flatten()[:5])
                print("[METTA LSTM] Output state[1] shape:", state[1].shape, "sample:", state[1].flatten()[:5])
            else:
                print("[METTA LSTM] Output state: None")

        output = rearrange(output, "t b h -> (b t) h")
        if state is not None:
            state = tuple(s.detach() for s in state)
            state = torch.cat(state, dim=0)

        # Store state under output_key
        if self.output_key not in md.data or not isinstance(md.data[self.output_key], dict):
            md.data[self.output_key] = {}
        md.data[self.output_key]["state"] = state
        return {self.output_key: output}

    def _check_shapes(self, md: MettaDict) -> None:
        # No-op: shape validation is handled upstream
        pass
