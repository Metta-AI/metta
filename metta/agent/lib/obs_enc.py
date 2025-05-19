from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import LayerBase


class ObsTokenShaper(LayerBase):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        embed_dim: int,
        M: int = 200,  # need to get this from mettagrid.yaml or env
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._embed_dim = embed_dim  # Dimension of attribute embeddings
        self._coord_dim = 1  # Dimension for each coordinate (x, y) and value
        self._value_dim = 1
        self._feat_dim = self._embed_dim + self._value_dim + 2 * self._coord_dim
        self.M = M

    def _make_net(self) -> None:
        self._out_tensor_shape = [self.M, self._feat_dim]

        # Embedding layer for attribute indices. Index 0 is used for padding.
        # Max attribute index + 1 for padding idx 0
        self._embeds = nn.Embedding(257, self._embed_dim, padding_idx=0)
        nn.init.uniform_(self._embeds.weight, -0.1, 0.1)
        # Ensure padding_idx embedding is zeros
        self._embeds.weight.data[0].fill_(0)

        # Pre-compute coordinate lookup table to avoid bitwise ops every forward pass
        coord_values = torch.arange(256)
        coords_lut = torch.stack([(coord_values >> 4) & 0x0F, coord_values & 0x0F], dim=-1).float() / 15.0
        self.register_buffer("coords_lut", coords_lut)

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), atr_idx, atr_val
        observations = td.get("x")

        B = observations.shape[0]
        TT = 1
        td["_B_"] = B
        td["_TT_"] = TT
        if len(observations.shape) > 3:
            TT = observations.shape[1]
            td["_TT_"] = TT
            observations = einops.rearrange(observations, "b t h c -> (b t) h c")
        td["_BxTT_"] = B * TT
        # M = observations.shape[1]  # Max observations, not explicitly needed if using -1 or slicing

        # Extract components from observations
        # observations shape: [B_TT, M, 3]
        # coords_byte contains x and y coordinates in a single byte
        coords_byte = observations[..., 0].to(torch.long)  # indices must be int64 for gather

        # Gather normalized x and y coordinates from the lookup table
        coords_norm = self.coords_lut[coords_byte]  # type: ignore[index]  # Shape: [B_TT, M, 2]
        x_coords = coords_norm[..., 0:1]  # Shape: [B_TT, M, 1]
        y_coords = coords_norm[..., 1:2]  # Shape: [B_TT, M, 1]

        # atr_indices are integers for embedding lookup
        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding

        # atr_values are floats (normalized 0-1)
        atr_values = observations[..., 2].float().unsqueeze(-1)  # Shape: [B_TT, M, 1]

        # Generate embeddings for attribute indices
        # self._embeds.weight.data[0] is already zero due to padding_idx and manual setting
        atr_embeds = self._embeds(atr_indices)  # [B_TT, M, embed_dim]

        # Concatenate to form feature vectors
        # Feature vector: [atr_embeds, atr_values, x_coords, y_coords]
        feat_vectors = torch.cat([atr_embeds, atr_values, x_coords, y_coords], dim=-1)  # [B_TT, M, feat_dim]

        obs_mask = atr_indices == 0

        td[self._name] = feat_vectors
        td["obs_mask"] = obs_mask
        return td


class ObsVanillaAttn(LayerBase):
    """Future work can go beyond a single layer and single head. Also adding a GRU before the out projection."""

    def __init__(
        self,
        out_dim: int,
        attn_num_heads: int = 1,
        use_mask: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask
        self._attn_num_heads = attn_num_heads

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]

        self._out_tensor_shape = [self._M, self._out_dim]

        self._layer_norm_1 = nn.LayerNorm(self._feat_dim)

        # this could be more robust to varied vdim etc.
        self._attn = nn.MultiheadAttention(self._feat_dim, self._attn_num_heads, batch_first=True)

        self._layer_norm_2 = nn.LayerNorm(self._feat_dim)

        # we could replace this by adjusting vdim. Not excatly the same close.
        if self._feat_dim != self._out_dim:
            self._out_proj = nn.Linear(self._feat_dim, self._out_dim)
        else:
            self._out_proj = nn.Identity()

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x = td[self._sources[0]["name"]]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]
        x = self._layer_norm_1(x)
        x = self._attn(x, x, x, key_padding_mask=key_mask)
        x = self._layer_norm_2(x[0])
        x = self._out_proj(x)
        td[self._name] = x
        return td


class ObsLSTMCrossAttn(LayerBase):
    """Use LSTM as a classifier token. We take the mean between LSTM state and an initially random
    but learnable token and use that as a query to attend over the feature vectors. This layer collapses
    input sequence length to 1 so it also squeezes the output to [B, out_dim]."""

    def __init__(
        self,
        out_dim: int,
        hidden_size: int,
        core_num_layers: int,
        attn_num_heads: int = 1,
        use_mask: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._hidden_size = hidden_size
        self._core_num_layers = core_num_layers
        self._attn_num_heads = attn_num_heads
        self._use_mask = use_mask

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]

        self._out_tensor_shape = [self._out_dim]

        self._cls_token = nn.Parameter(torch.randn(1, 1, self._hidden_size))

        self._layer_norm_1 = nn.LayerNorm(self._hidden_size)

        if self._hidden_size != self._feat_dim:
            self._feat_proj = nn.Linear(self._hidden_size, self._feat_dim)
        else:
            self._feat_proj = nn.Identity()

        self._attn = nn.MultiheadAttention(self._feat_dim, self._attn_num_heads, batch_first=True)

        self._layer_norm_2 = nn.LayerNorm(self._feat_dim)

        if self._feat_dim != self._out_dim:
            self._out_proj = nn.Linear(self._feat_dim, self._out_dim)
        else:
            self._out_proj = nn.Identity()

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x = td[self._sources[0]["name"]]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]
        B_TT = x.shape[0]

        state_h_prev = td.get("state_h_prev", None)  # Shape: [B_TT, self.lstm_h_len]
        if state_h_prev is None:
            state_h_prev = self._cls_token.expand(B_TT, -1, -1)
        elif state_h_prev.ndim > 2:
            state_h_prev = state_h_prev[self._core_num_layers - 1]  # Takes the last layer's hidden state
        else:
            AttributeError("ObsLSTMCrossAttn has not been tested for LSTM layers < 3")

        # Ensure state_h_prev is [B_TT, H]
        if state_h_prev.shape[0] != B_TT:
            if state_h_prev.shape[0] == 1 and B_TT > 1:
                state_h_prev = state_h_prev.expand(B_TT, -1)

        query = self._layer_norm_1(state_h_prev)
        query = self._feat_proj(query)
        x = self._attn(query, x, x, key_padding_mask=key_mask)
        x = self._layer_norm_2(x[0])
        x = self._out_proj(x)
        x = einops.rearrange(x, "b m h -> b (m h)")
        td[self._name] = x
        return td


class ObsSlotAttn(LayerBase):
    """Future work: replace slot nn.Parameter with nn.Embedding"""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        num_iterations: int,
        slot_init_mu: float = 0.0,
        slot_init_sigma: float = 1.0,
        mlp_hidden_size: Optional[int] = None,
        epsilon: float = 1e-8,
        use_mask: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._num_slots = num_slots
        self._slot_dim = slot_dim
        self._num_iterations = num_iterations
        self._slot_init_mu = slot_init_mu
        self._slot_init_sigma = slot_init_sigma
        self._mlp_hidden_size = mlp_hidden_size if mlp_hidden_size is not None else slot_dim
        self._epsilon = epsilon
        self._use_mask = use_mask

    def _make_net(self) -> None:
        # Expected input shape: [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]

        self._out_tensor_shape = [self._num_slots, self._slot_dim]

        # Slot initialization
        # Instead of nn.Parameter for slots initial values to allow dynamic batch size,
        # we'll initialize them in the forward pass if not provided or make them learnable fixed params.
        # For simplicity, let's use learnable parameters for initial slot means and stds.
        self.slots_mu = nn.Parameter(torch.randn(1, self._num_slots, self._slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, self._num_slots, self._slot_dim))
        # Initialize with user-defined mean and sigma
        nn.init.normal_(self.slots_mu, mean=self._slot_init_mu, std=self._slot_init_sigma)

        # LayerNorm for inputs and slots
        self.norm_input = nn.LayerNorm(self._feat_dim)
        self.norm_slots = nn.LayerNorm(self._slot_dim)
        self.norm_mlp_input = nn.LayerNorm(self._slot_dim)

        # Projections for K, V from input features
        self.k_proj = nn.Linear(self._feat_dim, self._slot_dim, bias=False)
        self.v_proj = nn.Linear(self._feat_dim, self._slot_dim, bias=False)

        # Projection for Q from slots
        self.q_proj = nn.Linear(self._slot_dim, self._slot_dim, bias=False)

        # Slot update mechanism (GRU cell or MLP)
        # Using a GRU cell for slot updates
        self.gru = nn.GRUCell(input_size=self._slot_dim, hidden_size=self._slot_dim)

        # MLP for refining the update from GRU
        self.mlp = nn.Sequential(
            nn.Linear(self._slot_dim, self._mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self._mlp_hidden_size, self._slot_dim),
        )
        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        inputs = td[self._sources[0]["name"]]  # Shape: [B_TT, M, feat_dim]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]  # Shape: [B_TT, M], True for valid tokens

        B_TT = inputs.shape[0]

        # Initialize slots
        # Sample initial slots from the learned Gaussian distribution
        mu = self.slots_mu.expand(B_TT, -1, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(B_TT, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)  # Shape: [B_TT, num_slots, slot_dim]

        # Normalize inputs
        inputs_norm = self.norm_input(inputs)

        # Project K, V from input features
        k = self.k_proj(inputs_norm)  # Shape: [B_TT, M, slot_dim]
        v = self.v_proj(inputs_norm)  # Shape: [B_TT, M, slot_dim]

        # Iterative attention
        for _ in range(self._num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Project Q from slots
            q = self.q_proj(slots)  # Shape: [B_TT, num_slots, slot_dim]

            # Attention mechanism
            # q: [B_TT, num_slots, slot_dim]
            # k: [B_TT, M, slot_dim] -> k.transpose(-1, -2) is [B_TT, slot_dim, M]
            attn_logits = torch.matmul(q, k.transpose(-1, -2))  # Shape: [B_TT, num_slots, M]
            attn_logits = attn_logits / (self._slot_dim**0.5)

            if key_mask is not None:
                # obs_mask: [B_TT, M], need to unsqueeze for broadcasting: [B_TT, 1, M]
                # We want to mask out invalid observations for each slot.
                # Attention logits for masked positions should be -inf.
                attn_logits.masked_fill_(~key_mask.unsqueeze(1), -float("inf"))

            attn = torch.softmax(attn_logits, dim=-1)  # Shape: [B_TT, num_slots, M]

            # Weighted mean of values
            # attn: [B_TT, num_slots, M]
            # v:    [B_TT, M, slot_dim]
            # updates: [B_TT, num_slots, slot_dim]
            updates = torch.matmul(attn, v)

            # Slot update using GRU
            # GRU expects input [batch_size, input_size] and hidden [batch_size, hidden_size]
            # Here, batch_size effectively becomes B_TT * num_slots
            # updates: [B_TT, num_slots, slot_dim] -> [(B_TT * num_slots), slot_dim]
            # slots (hidden state): [B_TT, num_slots, slot_dim] -> [(B_TT * num_slots), slot_dim]
            updates_reshaped = einops.rearrange(updates, "b n d -> (b n) d")
            slots_reshaped = einops.rearrange(slots_prev, "b n d -> (b n) d")  # use slots_prev for GRU hidden state

            slots_updated_gru = self.gru(updates_reshaped, slots_reshaped)
            slots_updated_gru = einops.rearrange(slots_updated_gru, "(b n) d -> b n d", n=self._num_slots)

            # MLP for further refinement
            slots_updated_mlp = self.mlp(self.norm_mlp_input(slots_updated_gru))
            slots = slots_updated_gru + slots_updated_mlp  # Apply residual connection

        td[self._name] = slots  # Output shape: [B_TT, num_slots, slot_dim]
        return td
