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
        atr_embed_dim: int,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._atr_embed_dim = atr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._feat_dim = self._atr_embed_dim + self._value_dim
        self.M = obs_shape[0]

    def _make_net(self) -> None:
        self._out_tensor_shape = [self.M, self._feat_dim]

        self._atr_embeds = nn.Embedding(257, self._atr_embed_dim, padding_idx=0)
        nn.init.uniform_(self._atr_embeds.weight, -0.1, 0.1)

        self._coord_embeds = nn.Embedding(256, self._atr_embed_dim)
        nn.init.uniform_(self._coord_embeds.weight, -0.1, 0.1)

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), atr_idx, atr_val
        observations = td.get("x")

        B = observations.shape[0]
        TT = 1
        td["_B_"] = B
        td["_TT_"] = TT
        if observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = observations.shape[1]
            td["_TT_"] = TT
            observations = einops.rearrange(observations, "b t h c -> (b t) h c")
            # observations = observations.flatten(0, 1)
        td["_BxTT_"] = B * TT

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range)
        x_coord_indices = (coords_byte >> 4) & 0x0F  # Shape: [B_TT, M]
        y_coord_indices = coords_byte & 0x0F  # Shape: [B_TT, M]

        # Combine x and y indices to a single index for embedding lookup (0-255 range)
        # Assuming 16 possible values for x (0-15)
        combined_coord_indices = y_coord_indices * 16 + x_coord_indices
        coord_pair_embedding = self._coord_embeds(combined_coord_indices.long())  # [B_TT, M, 4]

        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_embeds = self._atr_embeds(atr_indices)  # [B_TT, M, embed_dim]

        combined_embeds = atr_embeds + coord_pair_embedding

        # atr_values are floats and we're assuming normalized 0-1 from the env but this won't break if not
        atr_values = observations[..., 2].float().unsqueeze(-1)  # Shape: [B_TT, M, 1]

        # Assemble feature vectors
        # feat_vectors will have shape [B_TT, M, _feat_dim] where _feat_dim = _embed_dim + _value_dim
        feat_vectors = torch.empty(
            (*atr_embeds.shape[:-1], self._feat_dim),
            dtype=atr_embeds.dtype,
            device=atr_embeds.device,
        )
        # Combined embedding portion
        feat_vectors[..., : self._atr_embed_dim] = combined_embeds
        feat_vectors[..., self._atr_embed_dim : self._atr_embed_dim + self._value_dim] = atr_values

        obs_mask = atr_indices == 0  # important! true means 0 ie mask me

        td[self._name] = feat_vectors
        td["obs_mask"] = obs_mask
        return td


class ObsVanillaAttn(LayerBase):
    """Future work can go beyond just using the feat dim as the attn qv dim, a single layer and single head,
    adding a GRU before the out projection..."""

    def __init__(
        self,
        out_dim: int,
        use_mask: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]
        self._scale = self._feat_dim**-0.5

        self._out_tensor_shape = [self._M, self._out_dim]

        self._layer_norm_1 = nn.LayerNorm(self._feat_dim)

        # QKV projection parameters
        self.W_qkv = nn.Parameter(torch.empty(self._feat_dim, 3 * self._feat_dim))
        self.b_qkv = nn.Parameter(torch.empty(3 * self._feat_dim))
        nn.init.xavier_uniform_(self.W_qkv)
        nn.init.zeros_(self.b_qkv)

        self._layer_norm_2 = nn.LayerNorm(self._feat_dim)

        if self._feat_dim != self._out_dim:
            self._out_proj = nn.Linear(self._feat_dim, self._out_dim)
        else:
            self._out_proj = nn.Identity()

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]  # True for elements to be masked

        x_norm1 = self._layer_norm_1(x_features)

        # QKV projection
        # x_norm1: [B, M, feat_dim], W_qkv: [feat_dim, 3 * feat_dim]
        # qkv: [B, M, 3 * feat_dim]
        qkv = torch.einsum("bmd,df->bmf", x_norm1, self.W_qkv) + self.b_qkv
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each [B, M, feat_dim]

        # Simplified single-head attention
        # Attention scores: [B, M, M]
        attn_scores = torch.einsum("bmd,bnd->bmn", q, k) * self._scale

        if key_mask is not None:
            # key_mask: [B, M] -> [B, 1, M] for broadcasting
            # True in key_mask means mask out, so fill with -inf
            attn_scores.masked_fill_(key_mask.unsqueeze(1), -torch.finfo(attn_scores.dtype).max)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M, M]

        # Weighted sum of V: [B, M, feat_dim]
        attn_output = torch.einsum("bmn,bnd->bmd", attn_weights, v)

        x_res = x_features + attn_output

        x_norm2 = self._layer_norm_2(x_res)

        output = self._out_proj(x_norm2)

        td[self._name] = output
        return td


class ObsCrossAttn(LayerBase):
    """Use LSTM as a classifier token. If LSTM hidden is None, use a learnable token in its stead.
    This layer collapses also input sequence length to 1 so it also squeezes the output to [B, out_dim]. This makes it
    the only of the above layers that can feed directly into the LSTM (otherwise, you need to decide how you'd like to
    collapse your sequence to a single token)."""

    def __init__(
        self,
        out_dim: int,
        hidden_size: int,
        core_num_layers: int,
        use_mask: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._hidden_size = hidden_size
        self._core_num_layers = core_num_layers
        self._use_mask = use_mask

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]

        self._out_tensor_shape = [self._out_dim]

        self._q_token = nn.Parameter(torch.randn(1, 1, self._hidden_size))

        self._layer_norm_1 = nn.LayerNorm(self._hidden_size)

        if self._hidden_size != self._feat_dim:
            self._feat_proj = nn.Linear(self._hidden_size, self._feat_dim)
        else:
            self._feat_proj = nn.Identity()

        self.q_proj = nn.Linear(self._feat_dim, self._feat_dim, bias=False)
        self.k_proj = nn.Linear(self._feat_dim, self._feat_dim, bias=False)
        self.v_proj = nn.Linear(self._feat_dim, self._feat_dim, bias=False)

        self._layer_norm_2 = nn.LayerNorm(self._feat_dim)

        if self._feat_dim != self._out_dim:
            self._out_proj = nn.Linear(self._feat_dim, self._out_dim)
        else:
            self._out_proj = nn.Identity()

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]
        B_TT = td["_BxTT_"]

        # state_h_prev = td.get("state", None)  # Variable is actually all of the state -> goes to h_prev in lines below
        # state_h_prev = None
        # if state_h_prev is None or state_h_prev.shape[0] != B_TT:
        #     state_h_prev = self._q_token.expand(B_TT, -1, -1)
        # else:
        #     state_h_prev = state_h_prev[self._core_num_layers - 1].unsqueeze(1)  # Takes the last layer's hidden state

        # # Ensure state_h_prev is [B_TT, 1, H] during training.
        # if state_h_prev.shape[0] != B_TT:
        #     TT = td["_TT_"]
        #     state_h_prev = einops.repeat(state_h_prev, "b one h -> (b t) one h", t=TT)
        # query = self._layer_norm_1(state_h_prev)

        query = self._q_token.expand(B_TT, -1, -1)

        query = self._feat_proj(query)

        q_p = self.q_proj(query)  # q_p is now [B_TT, 1, _feat_dim]
        k_p = self.k_proj(x_features)  # [B_TT, M, _feat_dim]
        v_p = self.v_proj(x_features)  # [B_TT, M, _feat_dim]

        # Calculate attention scores: Q_projected @ K_projected.T
        # q_p: [B_TT, 1, _feat_dim], k_p: [B_TT, M, _feat_dim]. that means linear attention, yay.
        attn_scores = torch.einsum("bqd,bkd->bqk", q_p, k_p)  # Result: [B_TT, 1, M]

        # Scale scores
        attn_scores = attn_scores / (self._feat_dim**0.5)

        # Apply mask
        if key_mask is not None:
            # key_mask shape: [B_TT, M] -> unsqueeze to [B_TT, 1, M] for broadcasting
            key_mask_expanded = key_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(key_mask_expanded, -float("inf"))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Result: [B_TT, 1, M]

        # Calculate output: Weights @ V_projected
        x = torch.einsum("bqk,bkd->bqd", attn_weights, v_p)  # Result: [B_TT, 1, _feat_dim]

        x = self._layer_norm_2(x)

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
        self._use_mask = use_mask

    def _make_net(self) -> None:
        # Expected input shape: [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]

        self._out_tensor_shape = [self._num_slots, self._slot_dim]

        # Slot initialization with a learnable mean and sigma
        self.slots_mu = nn.Parameter(torch.randn(1, self._num_slots, self._slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, self._num_slots, self._slot_dim))
        nn.init.normal_(self.slots_mu, mean=self._slot_init_mu, std=self._slot_init_sigma)

        self.norm_input = nn.LayerNorm(self._feat_dim)
        self.norm_slots = nn.LayerNorm(self._slot_dim)
        self.norm_mlp_input = nn.LayerNorm(self._slot_dim)

        self.k_proj = nn.Linear(self._feat_dim, self._slot_dim, bias=False)
        self.v_proj = nn.Linear(self._feat_dim, self._slot_dim, bias=False)
        self.q_proj = nn.Linear(self._slot_dim, self._slot_dim, bias=False)

        self.gru = nn.GRUCell(input_size=self._slot_dim, hidden_size=self._slot_dim)

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
            key_mask = td["obs_mask"]  # Shape: [B_TT, M], True for tokens to be masked (padding)

        B_TT = inputs.shape[0]

        # Initialize slots
        # Sample initial slots from the learned Gaussian distribution
        mu = self.slots_mu.expand(B_TT, -1, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(B_TT, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)  # Shape: [B_TT, num_slots, slot_dim]

        inputs_norm = self.norm_input(inputs)

        k = self.k_proj(inputs_norm)  # Shape: [B_TT, M, slot_dim]
        v = self.v_proj(inputs_norm)  # Shape: [B_TT, M, slot_dim]

        for _ in range(self._num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.q_proj(slots)  # Shape: [B_TT, num_slots, slot_dim]

            attn_logits = torch.einsum("ijk,ilk->ijl", q, k)  # Shape: [B_TT, num_slots, M]
            attn_logits = attn_logits / (self._slot_dim**0.5)

            if key_mask is not None:
                # obs_mask: [B_TT, M], need to unsqueeze for broadcasting: [B_TT, 1, M]
                attn_logits.masked_fill_(key_mask.unsqueeze(1), -float("inf"))

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

            slots_updated_mlp = self.mlp(self.norm_mlp_input(slots_updated_gru))
            slots = slots_updated_gru + slots_updated_mlp  # Apply residual connection

        td[self._name] = slots  # Output shape: [B_TT, num_slots, slot_dim]
        return td
