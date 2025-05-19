import torch
import torch.nn as nn
from einops import repeat
from tensordict import TensorDict
from torch.nn import functional as F

from metta.agent.lib import nn_layer_library
from metta.agent.lib.metta_layer import LayerBase


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
        hidden_input_to_lstm = td[self._sources[0]["name"]] # Renamed to avoid confusion with LSTM's output 'hidden'
        state = td["state"]

        if state is not None:
            split_size = self.num_layers
            state = (state[:split_size], state[split_size:])

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

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B, "LSTM state batch size mismatch"
        assert hidden_input_to_lstm.shape == (B * TT, self._in_tensor_shapes[0][0]), (
            f"Hidden state shape {hidden_input_to_lstm.shape} does not match expected {(B * TT, self._in_tensor_shapes[0][0])}"
        )

        hidden_input_to_lstm = hidden_input_to_lstm.reshape(B, TT, self._in_tensor_shapes[0][0])
        hidden_input_to_lstm = hidden_input_to_lstm.transpose(0, 1)

        hidden, state = self._net(hidden_input_to_lstm, state)

        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(B * TT, self.hidden_size)

        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print(f"LSTM: Warning - output 'hidden' tensor from LSTM layer contains NaN or Inf values BEFORE writing to TensorDict!")
            # print(f"LSTM output hidden (sample): {hidden[:5]}")

        if state is not None:
            state = tuple(s.detach() for s in state)
            state = torch.cat(state, dim=0)

        td[self._name] = hidden
        td["state"] = state

        return td


class LSTMWords(LayerBase):
    def __init__(self, hidden_size, gumbel_tau=1.0, gumbel_hard=True, **cfg):
        super().__init__(**cfg)
        # self.initial_words_len = initial_words_len
        self.initial_words = [n for n in range(500)]
        self.num_words = len(self.initial_words)
        self._nn_params["embedding_dim"] = hidden_size # need a linear to map to dims different from core
        self.gumbel_tau = gumbel_tau
        self.gumbel_hard = gumbel_hard

        self.register_buffer("words", torch.tensor(self.initial_words))

    def update_words(self, words):
        pass

    def _make_net(self):
        self._out_tensor_shape = [self._nn_params["embedding_dim"]]
        return nn.Embedding(self.num_words, self._nn_params["embedding_dim"])

    def _forward(self, td: TensorDict):
        B_TT = td["_BxTT_"]
        hidden = td[self._sources[0]["name"]]  # Shape: [B*TT, hidden]

        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print(f"LSTMWords: Warning - input 'hidden' tensor to LSTMWords contains NaN or Inf values!")
            # print(f"Hidden (sample): {hidden[:5]}")

        if torch.isnan(self._net.weight).any() or torch.isinf(self._net.weight).any():
            print(f"LSTMWords: Warning - embedding layer weights (self._net.weight) contain NaN or Inf values!")
            # print(f"Embedding weights (sample): {self._net.weight[:5]}")

        # get embeddings then expand to match the batch size
        # words = repeat(self._net(self.words), "a e -> b a e", b=B_TT)
        words = self._net(self.words)

        if torch.isnan(words).any() or torch.isinf(words).any():
            print(f"LSTMWords: Warning - 'words' embedding tensor (output of self._net(self.words)) contains NaN or Inf values!")
            # print(f"Words (sample): {words[:5]}")

        words_reshaped = repeat(words, "k e -> n k e", n=B_TT)

        # Reshape inputs similar to Rev2 for bilinear calculation
        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        # hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_words)  # shape: [B*TT, num_words, hidden]
        # hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")  # shape: [N, H]
        # word_reshaped = rearrange(words, "b a e -> (b a) e")  # shape: [N, E]

        hidden_reshaped = hidden.unsqueeze(1)
        scores = F.cosine_similarity(hidden_reshaped, words_reshaped, dim=2, eps=1e-6)  # Shape: [B_TT, self.num_words]

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("LSTMWords: Warning - scores tensor contains NaN or Inf values BEFORE gumbel_softmax!")
            # Consider adding more debug prints here if needed, e.g.:
            # print(f"Scores (sample): {scores[0, :10]}")
            # print(f"Hidden (sample norms): {torch.norm(hidden_reshaped[0,:,:], dim=-1)}")
            # print(f"Words (sample norms): {torch.norm(words_reshaped[0, :10, :], dim=-1)}")

        # Apply Gumbel-Softmax
        # tau is the temperature; lower tau -> sharper distribution
        # hard=True makes the output one-hot like (or close to it)
        # tau = 1.0  # This could be a parameter or configured # Old hardcoded tau
        gumbel_output = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=-1) # Use configured tau and hard

        selected_embedding = torch.matmul(gumbel_output, words)
        td[self._name] = selected_embedding
        return td
