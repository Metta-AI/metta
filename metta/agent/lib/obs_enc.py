import einops
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import LayerBase


class ObsAttn(LayerBase):
    def __init__(
        self,
        obs_shape,
        hidden_size,
        core_num_layers,
        embed_dim,
        QV_dim,
        Q_hidden=0,
        K_hidden=0,
        V_hidden=0,
        M=200,
        **cfg,
    ):
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._lstm_h_len = hidden_size
        self._lstm_layers = core_num_layers
        self._embed_dim = embed_dim  # Dimension of attribute embeddings
        self._coord_dim = 1  # Dimension for each coordinate (x, y) and value
        self._value_dim = 1
        self._feat_dim = self._embed_dim + self._value_dim + 2 * self._coord_dim
        self._QV_dim = QV_dim  # Dimension for Query and Key projections
        self._QV_scale = self._QV_dim**0.5
        self._V_dim = hidden_size  # Dimension for Value projection, matches LSTM hidden size for output
        self.Q_hidden = Q_hidden
        self.K_hidden = K_hidden
        self.V_hidden = V_hidden
        self.M = M

    def _make_net(self):
        self._out_tensor_shape = [self._V_dim]

        # Embedding layer for attribute indices. Index 0 is used for padding.
        # Max attribute index + 1 for padding idx 0
        self._embeds = nn.Embedding(257, self._embed_dim, padding_idx=0)
        nn.init.uniform_(self._embeds.weight, -0.1, 0.1)
        # Ensure padding_idx embedding is zeros
        self._embeds.weight.data[0].fill_(0)

        self._embed_norm = nn.LayerNorm(self._embed_dim)

        if self.Q_hidden > 0:
            self._Q = nn.Sequential(
                nn.Linear(self._lstm_h_len, self.Q_hidden),
                nn.ReLU(),
                nn.Linear(self.Q_hidden, self._QV_dim),
            )
        else:
            self._Q = nn.Linear(self._lstm_h_len, self._QV_dim)

        if self.K_hidden > 0:
            self._K = nn.Sequential(
                nn.Linear(self._feat_dim, self.K_hidden),  # Input is the concatenated feature vector
                nn.ReLU(),
                nn.Linear(self.K_hidden, self._QV_dim),
            )
        else:
            self._K = nn.Linear(self._feat_dim, self._QV_dim)

        if self.V_hidden > 0:
            self._V = nn.Sequential(
                nn.Linear(self._feat_dim, self.V_hidden),  # Input is the concatenated feature vector
                nn.ReLU(),
                nn.Linear(self.V_hidden, self._V_dim),
            )
        else:
            self._V = nn.Linear(self._feat_dim, self._V_dim)

        return None

    def _forward(self, td: TensorDict):
        # [B, M, 3] the 3 vector is: coord (unit8), atr_idx, atr_val
        observations = td.get("x")

        B_TT = observations.shape[0]
        td["_B_"] = B_TT
        td["_TT_"] = 1
        td["_BxTT_"] = B_TT
        if len(observations.shape) > 3:
            B = observations.shape[0]
            TT = observations.shape[1]
            B_TT = B * TT
            td["_BxTT_"] = B_TT
            observations = einops.rearrange(observations, "b t h c -> (b t) h c")
        # M = observations.shape[1]  # Max observations, not explicitly needed if using -1 or slicing

        state_h_prev = td.get("state_h_prev", None)  # Shape: [B_TT, self.lstm_h_len]
        if state_h_prev is None:
            state_h_prev = torch.zeros(B_TT, self._lstm_h_len, device=observations.device)
        elif state_h_prev.ndim > 2:  # Assuming state_h_prev might come from LSTM [num_layers, B, H]
            state_h_prev = state_h_prev[self._lstm_layers - 1]  # Takes the last layer's hidden state
        # Ensure state_h_prev is [B_TT, H]
        if state_h_prev.shape[0] != B_TT:
            if state_h_prev.shape[0] == 1 and B_TT > 1:
                state_h_prev = state_h_prev.expand(B_TT, -1)
            # else: # Potentially raise an error or log a warning

        # Extract components from observations
        # observations shape: [B_TT, M, 3]
        # coords_byte contains x and y coordinates in a single byte
        coords_byte = observations[..., 0].to(torch.int)  # Ensure integer type for bitwise ops

        # Extract x_coords (first 4 bits) and y_coords (last 4 bits)
        # Normalize coordinates to be in [0, 1] by dividing by 15.0 (since 4 bits means 0-15 range)
        # Mettagrid must expand the coord from uint8 if we want a greater than 16x16 grid.
        x_coords = ((coords_byte >> 4) & 0x0F).float().unsqueeze(-1) / 15.0  # Shape: [B_TT, M, 1]
        y_coords = (coords_byte & 0x0F).float().unsqueeze(-1) / 15.0  # Shape: [B_TT, M, 1]

        # atr_indices are integers for embedding lookup
        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding

        # atr_values are floats (normalized 0-1)
        atr_values = observations[..., 2].float().unsqueeze(-1)  # Shape: [B_TT, M, 1]

        # Generate embeddings for attribute indices
        # self._embeds.weight.data[0] is already zero due to padding_idx and manual setting
        atr_embeds = self._embeds(atr_indices)  # [B_TT, M, embed_dim]
        atr_embeds = self._embed_norm(atr_embeds)  # Apply LayerNorm

        # Concatenate to form feature vectors
        # Feature vector: [atr_embeds, atr_values, x_coords, y_coords]
        feat_vectors = torch.cat([atr_embeds, atr_values, x_coords, y_coords], dim=-1)  # [B_TT, M, feat_dim]

        # Project Q, K, V
        Q_proj = self._Q(state_h_prev)  # Shape: [B_TT, _QV_dim]
        K_proj = self._K(feat_vectors)  # Shape: [B_TT, M, _QV_dim]
        V_proj = self._V(feat_vectors)  # Shape: [B_TT, M, _V_dim]

        # Compute attention scores
        # Q_proj: [B_TT, _QV_dim] -> [B_TT, 1, _QV_dim]
        # K_proj: [B_TT, M, _QV_dim] -> K_proj.transpose(-2, -1) gives [B_TT, _QV_dim, M]
        attn_scores = torch.matmul(Q_proj.unsqueeze(1), K_proj.transpose(-2, -1))  # Shape: [B_TT, 1, M]
        attn_scores = attn_scores.squeeze(1)  # Shape: [B_TT, M]

        # Scale scores
        attn_scores = attn_scores / (self._QV_scale)

        # Create attention mask from attribute indices (assuming 0 is padding_idx)
        # obs_mask is [B_TT, M], True for valid tokens, False for padding
        obs_mask = atr_indices != 0
        attn_scores.masked_fill_(~obs_mask, -float("inf"))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: [B_TT, M]

        # Compute weighted sum (mean) of V
        # attn_weights: [B_TT, M] -> [B_TT, 1, M]
        # V_proj: [B_TT, M, self._V_dim]
        output = torch.matmul(attn_weights.unsqueeze(1), V_proj)  # Shape: [B_TT, 1, self._V_dim]
        output = output.squeeze(1)  # Shape: [B_TT, self._V_dim]

        td[self._name] = output
