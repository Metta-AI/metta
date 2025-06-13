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
        feature_normalizations: list[float],
        use_max_n_dense: Optional[bool] = None,  # delete
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._atr_embed_dim = atr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._feat_dim = self._atr_embed_dim + self._value_dim
        self.M = obs_shape[0]
        self._feature_normalizations = list(feature_normalizations)
        self._max_embeds = 256

    def _make_net(self) -> None:
        self._out_tensor_shape = [self.M, self._feat_dim]

        self._atr_embeds = nn.Embedding(self._max_embeds, self._atr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._atr_embeds.weight, std=0.02)

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._atr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        # Create a tensor for feature normalizations
        # We need to handle the case where atr_idx might be 0 (padding) or larger than defined normalizations.
        # Assuming max atr_idx is 256 (same as atr_embeds size - 1 for padding_idx).
        # Initialize with 1.0 to avoid division by zero for unmapped indices.
        norm_tensor = torch.ones(self._max_embeds, dtype=torch.float32)
        for i, val in enumerate(self._feature_normalizations):
            if i < len(norm_tensor):  # Ensure we don't go out of bounds
                norm_tensor[i] = val
            else:
                raise ValueError(f"Feature normalization {val} is out of bounds for Embedding layer size {i}")
        self.register_buffer("_norm_factors", norm_tensor)

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
        M = observations.shape[1]
        td["_BxTT_"] = B * TT

        # transition_idx = atr_values.argmax(dim=0)

        coords = observations[..., 0]
        obs_mask = coords == 255  # important! true means mask me

        # 1) find each row’s flip‐point
        flip_pts = obs_mask.int().argmax(dim=1)  # shape [B], on GPU

        # 2) find the global max flip‐point as a 0‐d tensor (still on GPU)
        max_flip = flip_pts.max()  # e.g. tensor(3, device='cuda')
        if max_flip == 0:
            max_flip = max_flip + M  # hack to avoid 0. should instead grab

        # 3) build a 1‐D “positions” row [0,1,2,…,L−1]
        positions = torch.arange(M, device=obs_mask.device)

        # 4) make a boolean column mask: keep all columns strictly before max_flip
        keep_cols = positions < max_flip  # shape [L], dtype=torch.bool

        # 5) now “slice” your batch in one go, on the GPU:
        observations = observations[:, keep_cols]  # shape [B, max_flip]
        obs_mask = obs_mask[:, keep_cols]

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

        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        # Gather normalization factors based on atr_indices
        norm_factors = self._norm_factors[atr_indices]  # Shape: [B_TT, M]

        # Normalize atr_values
        # no epsilon to prevent division by zero - we want to fail if we have a bad normalization
        normalized_atr_values = atr_values / (norm_factors)
        normalized_atr_values = normalized_atr_values.unsqueeze(-1)  # Shape: [B_TT, M, 1]

        # Assemble feature vectors
        # feat_vectors will have shape [B_TT, M, _feat_dim] where _feat_dim = _embed_dim + _value_dim
        feat_vectors = torch.empty(
            (*atr_embeds.shape[:-1], self._feat_dim),
            dtype=atr_embeds.dtype,
            device=atr_embeds.device,
        )
        # Combined embedding portion
        feat_vectors[..., : self._atr_embed_dim] = combined_embeds
        feat_vectors[..., self._atr_embed_dim : self._atr_embed_dim + self._value_dim] = normalized_atr_values

        td[self._name] = feat_vectors
        td["obs_mask"] = obs_mask
        return td


class ObsTokenShaperValueEmbed(LayerBase):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        atr_embed_dim: int,
        feature_normalizations: list[float],
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._atr_embed_dim = atr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._feat_dim = self._atr_embed_dim + self._value_dim
        self.M = obs_shape[0]
        self._feature_normalizations = list(feature_normalizations)
        self._max_embeds = 256

    def _make_net(self) -> None:
        self._out_tensor_shape = [self.M, self._feat_dim]

        self._atr_embeds = nn.Embedding(self._max_embeds, self._atr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._atr_embeds.weight, std=0.02)

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._atr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._val_embeds = nn.Embedding(self._max_embeds, self._atr_embed_dim)
        nn.init.trunc_normal_(self._val_embeds.weight, std=0.02)

        # Create a tensor for feature normalizations
        # We need to handle the case where atr_idx might be 0 (padding) or larger than defined normalizations.
        # Assuming max atr_idx is 256 (same as atr_embeds size - 1 for padding_idx).
        # Initialize with 1.0 to avoid division by zero for unmapped indices.
        norm_tensor = torch.ones(self._max_embeds, dtype=torch.float32)
        for i, val in enumerate(self._feature_normalizations):
            if i < len(norm_tensor):  # Ensure we don't go out of bounds
                norm_tensor[i] = val
            else:
                raise ValueError(f"Feature normalization {val} is out of bounds for Embedding layer size {i}")
        self.register_buffer("_norm_factors", norm_tensor)

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

        # The attribute value is treated as a categorical variable for embedding.
        val_indices = observations[..., 2].long()
        # Clip values to be within the embedding range to avoid errors, e.g. if a value is 256 for _max_embeds=256
        val_indices = torch.clamp(val_indices, 0, self._max_embeds - 1)
        val_embeds = self._val_embeds(val_indices)

        combined_embeds = atr_embeds + coord_pair_embedding + val_embeds

        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        # Gather normalization factors based on atr_indices
        norm_factors = self._norm_factors[atr_indices]  # Shape: [B_TT, M]

        # Normalize atr_values
        # no epsilon to prevent division by zero - we want to fail if we have a bad normalization
        normalized_atr_values = atr_values / (norm_factors)
        normalized_atr_values = normalized_atr_values.unsqueeze(-1)  # Shape: [B_TT, M, 1]

        # Assemble feature vectors
        # feat_vectors will have shape [B_TT, M, _feat_dim] where _feat_dim = _embed_dim + _value_dim
        feat_vectors = torch.empty(
            (*atr_embeds.shape[:-1], self._feat_dim),
            dtype=atr_embeds.dtype,
            device=atr_embeds.device,
        )
        # Combined embedding portion
        feat_vectors[..., : self._atr_embed_dim] = combined_embeds
        feat_vectors[..., self._atr_embed_dim : self._atr_embed_dim + self._value_dim] = normalized_atr_values

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
        num_layers: int = 1,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask
        self._num_layers = num_layers

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._M = self._in_tensor_shapes[0][0]
        self._feat_dim = self._in_tensor_shapes[0][1]
        self._scale = self._feat_dim**-0.5

        self._out_tensor_shape = [self._M, self._out_dim]

        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(self._feat_dim) for _ in range(self._num_layers)])
        self.W_qkvs = nn.ParameterList()
        self.b_qkvs = nn.ParameterList()
        for _ in range(self._num_layers):
            W_qkv = nn.Parameter(torch.empty(self._feat_dim, 3 * self._feat_dim))
            b_qkv = nn.Parameter(torch.empty(3 * self._feat_dim))
            nn.init.xavier_uniform_(W_qkv)
            nn.init.zeros_(b_qkv)
            self.W_qkvs.append(W_qkv)
            self.b_qkvs.append(b_qkv)

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

        x = x_features
        for i in range(self._num_layers):
            x_res = x
            x_norm = self.layer_norms_1[i](x)

            # QKV projection
            # x_norm: [B, M, feat_dim], W_qkv: [feat_dim, 3 * feat_dim]
            # qkv: [B, M, 3 * feat_dim]
            qkv = torch.einsum("bmd,df->bmf", x_norm, self.W_qkvs[i]) + self.b_qkvs[i]
            q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each [B, M, feat_dim]

            # Simplified single-head attention
            # Attention scores: [B, M, M]
            attn_scores = torch.einsum("bmd,bnd->bmn", q, k) * self._scale

            if key_mask is not None:
                # key_mask: [B, M] -> [B, 1, M] for broadcasting
                # True in key_mask means mask out, so fill with -inf
                mask_value = -torch.finfo(attn_scores.dtype).max
                attn_scores = attn_scores + key_mask.unsqueeze(1).to(attn_scores.dtype) * mask_value

            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M, M]

            # Weighted sum of V: [B, M, feat_dim]
            attn_output = torch.einsum("bmn,bnd->bmd", attn_weights, v)

            x = x_res + attn_output

        x_norm2 = self._layer_norm_2(x)

        output = self._out_proj(x_norm2)

        td[self._name] = output
        return td


class ObsCrossAttn(LayerBase):
    """
    Performs cross-attention between learnable query tokens and input features.

    This layer uses a set of `num_query_tokens` learnable parameters as queries to attend
    to the input features `x_features` (typically coming from an observation encoder).

    !!! Note About Output Shape: !!!
    The output shape depends on the `num_query_tokens` parameter:
    - If `num_query_tokens == 1`, the output tensor shape will be `[B, out_dim]`,
      where B is the batch-time dimension (B_TT).
    - If `num_query_tokens > 1`, the output tensor shape will be `[B, num_query_tokens, out_dim]`.

    Key Functionality:
    1. Initializes `num_query_tokens` learnable query tokens.
    2. Projects these query tokens to `qk_dim` (query projection).
    3. Projects input features to `qk_dim` (key projection) and `v_dim` (value projection).
    4. Computes scaled dot-product attention scores between projected queries and keys.
    5. Applies softmax to get attention weights.
    6. Computes the weighted sum of projected values using these weights.
    7. Passes the result through a LayerNorm and an optional output MLP.

    Args:
        out_dim (int): The final output dimension for each query token's processed features.
        use_mask (bool, optional): If True, uses an observation mask (`obs_mask` from the
            input TensorDict) to mask attention scores for padded elements in `x_features`.
            Defaults to False.
        num_query_tokens (int, optional): The number of learnable query tokens to use.
            Defaults to 1.
        query_token_dim (Optional[int], optional): The embedding dimension for the initial
            learnable query tokens. If None, defaults to the input feature dimension (`feat_dim`).
            Defaults to None.
        qk_dim (Optional[int], optional): The dimension for query and key projections in the
            attention mechanism. If None, defaults to `feat_dim`. Defaults to None.
        v_dim (Optional[int], optional): The dimension for value projection in the attention
            mechanism. If None, defaults to `feat_dim`. Defaults to None.
        mlp_out_hidden_dim (Optional[int], optional): If provided, an MLP is used as the final
            output projection. This defines the hidden layer size of that MLP.
            The MLP structure is Linear(v_dim, mlp_out_hidden_dim) -> ReLU -> Linear(mlp_out_hidden_dim, out_dim).
            If None and v_dim == out_dim, an nn.Identity() is used. If None and v_dim != out_dim,
            a ValueError is raised (as mlp_out_hidden_dim is required to define the MLP).
            Defaults to None.
        **cfg: Additional configuration for LayerBase.

    Input TensorDict:
        - `x_features` (from `self._sources[0]["name"]`): Tensor of shape `[B_TT, M, feat_dim]`
          containing the input features.
        - `obs_mask` (optional, if `use_mask` is True): Tensor of shape `[B_TT, M]` indicating
          elements to be masked (True for masked).
        - `_BxTT_`: Batch-time dimension.

    Output TensorDict:
        - `self._name`: Output tensor. Shape is `[B_TT, out_dim]` if `num_query_tokens == 1`,
          or `[B_TT, num_query_tokens, out_dim]` if `num_query_tokens > 1`.
    """

    def __init__(
        self,
        out_dim: int,
        use_mask: bool = False,
        num_query_tokens: int = 1,
        query_token_dim: Optional[int] = None,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        mlp_out_hidden_dim: Optional[int] = None,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask
        self._num_query_tokens = num_query_tokens
        self._query_token_dim = query_token_dim
        self._qk_dim = qk_dim
        self._v_dim = v_dim
        self._mlp_out_hidden_dim = mlp_out_hidden_dim
        self._qk_dim_sqrt = self._qk_dim**0.5

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._feat_dim = self._in_tensor_shapes[0][1]

        if self._qk_dim is None:
            self._qk_dim = self._feat_dim
        if self._v_dim is None:
            self._v_dim = self._feat_dim
        if self._query_token_dim is None:
            self._query_token_dim = self._feat_dim

        if self._num_query_tokens == 1:
            self._out_tensor_shape = [self._out_dim]
        else:
            self._out_tensor_shape = [self._num_query_tokens, self._out_dim]

        self._q_token = nn.Parameter(torch.randn(1, self._num_query_tokens, self._query_token_dim))
        nn.init.trunc_normal_(self._q_token, std=0.02)

        self._layer_norm_1 = nn.LayerNorm(self._feat_dim)

        self.q_proj = nn.Linear(self._query_token_dim, self._qk_dim, bias=False)
        self.k_proj = nn.Linear(self._feat_dim, self._qk_dim, bias=False)
        self.v_proj = nn.Linear(self._feat_dim, self._v_dim, bias=False)

        self._layer_norm_2 = nn.LayerNorm(self._v_dim)

        self._out_proj = nn.Identity()
        if self._v_dim != self._out_dim or self._mlp_out_hidden_dim is not None:
            if self._mlp_out_hidden_dim is None:
                raise ValueError("mlp_out_hidden_dim must be provided if v_dim != out_dim")
            self._out_proj = nn.Sequential(
                nn.Linear(self._v_dim, self._mlp_out_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._mlp_out_hidden_dim, self._out_dim),
            )

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]
        B_TT = td["_BxTT_"]

        # query_token_unprojected will have shape [B_TT, num_query_tokens, _feat_dim]
        query_token_unprojected = self._q_token.expand(B_TT, -1, -1)

        q_p = self.q_proj(query_token_unprojected)  # q_p is now [B_TT, num_query_tokens, _actual_qk_dim]
        k_p = self.k_proj(x_features)  # [B_TT, M, _actual_qk_dim]
        v_p = self.v_proj(x_features)  # [B_TT, M, _actual_v_dim]

        # Calculate attention scores: Q_projected @ K_projected.T
        # q_p: [B_TT, num_query_tokens, _actual_qk_dim], k_p: [B_TT, M, _actual_qk_dim].
        # attn_scores will have shape [B_TT, num_query_tokens, M]
        attn_scores = torch.einsum("bqd,bkd->bqk", q_p, k_p)

        # Scale scores
        attn_scores = attn_scores / self._qk_dim_sqrt

        # Apply mask
        if key_mask is not None:
            mask_value = -torch.finfo(attn_scores.dtype).max
            attn_scores = attn_scores + key_mask.unsqueeze(1).to(attn_scores.dtype) * mask_value

        # Softmax to get attention weights
        # attn_weights will have shape [B_TT, num_query_tokens, M]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Calculate output: Weights @ V_projected
        # x will have shape [B_TT, num_query_tokens, _actual_v_dim]
        x = torch.einsum("bqk,bkd->bqd", attn_weights, v_p)

        x = self._layer_norm_2(x)

        # x shape: [B_TT, num_query_tokens, _actual_v_dim]
        # _out_proj maps last dim from _actual_v_dim to _out_dim
        # x after _out_proj: [B_TT, num_query_tokens, _out_dim]
        x = self._out_proj(x)

        if self._num_query_tokens == 1:
            # Reshape [B_TT, 1, self._out_dim] to [B_TT, self._out_dim]
            # This explicitly removes the middle dimension of size 1.
            x = einops.rearrange(x, "btt 1 d -> btt d")
        # Else (num_query_tokens > 1), x is already [B_TT, self._num_query_tokens, self._out_dim]
        # and this shape is consistent with self._out_tensor_shape (plus batch dim).

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
                mask_value = -torch.finfo(attn_logits.dtype).max
                attn_logits = attn_logits + key_mask.unsqueeze(1).to(attn_logits.dtype) * mask_value

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


class ObsTransformerEncoder(LayerBase):
    """
    Applies a multi-layer Transformer Encoder to a sequence of observation tokens,
    using a prepended CLS (class) token to aggregate information.

    This layer implements the core logic of a Vision Transformer (ViT) encoder:
    1. A learnable CLS token is prepended to the input sequence of tokens.
    2. The combined sequence is processed by a standard `nn.TransformerEncoder`.
    3. The output representation of the CLS token is extracted and returned as the
       final output of the layer.

    This is designed to produce a single vector representation that summarizes the
    entire input sequence. Positional encodings are expected to be handled by
    upstream layers (e.g., within the token shaper).

    Args:
        out_dim (int): The dimension of the final output vector.
        num_layers (int, optional): The number of sub-encoder-layers in the encoder.
            Defaults to 3.
        num_heads (int, optional): The number of heads in the multiheadattention models.
            Defaults to 6.
        mlp_ratio (float, optional): The ratio of the feedforward layer's size to the
            embedding size. Defaults to 3.0.
        use_mask (bool, optional): If True, an observation mask (`obs_mask` from the
            input TensorDict) is used to ignore padded tokens in the input sequence.
            Defaults to False.
        **cfg: Additional configuration for LayerBase.

    Input TensorDict:
        - `x_features` (from `self._sources[0]["name"]`): Tensor of shape
          `[B_TT, M, feat_dim]` containing the input token features.
        - `obs_mask` (optional, if `use_mask` is True): Tensor of shape `[B_TT, M]`
          indicating elements to be masked (True for masked).
        - `_BxTT_`: Batch-time dimension.

    Output TensorDict:
        - `self._name`: Output tensor of shape `[B_TT, out_dim]`.
    """

    def __init__(
        self,
        out_dim: int,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        use_mask: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._mlp_ratio = mlp_ratio
        self._use_mask = use_mask

    def _make_net(self) -> None:
        self._feat_dim = self._in_tensor_shapes[0][1]
        self._out_tensor_shape = [self._out_dim]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._feat_dim,
            nhead=self._num_heads,
            dim_feedforward=int(self._feat_dim * self._mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self._num_layers)

        self.final_norm = nn.LayerNorm(self._feat_dim)

        if self._feat_dim == self._out_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(self._feat_dim, self._out_dim)

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]  # [B_TT, M, feat_dim]
        B_TT = x_features.shape[0]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B_TT, -1, -1)  # [B_TT, 1, feat_dim]
        tokens = torch.cat([cls_tokens, x_features], dim=1)  # [B_TT, M + 1, feat_dim]

        # Create mask if necessary
        mask = None
        if self._use_mask:
            obs_mask = td["obs_mask"]  # [B_TT, M], True means mask
            # CLS token should not be masked
            cls_mask = torch.zeros(B_TT, 1, dtype=torch.bool, device=obs_mask.device)
            mask = torch.cat([cls_mask, obs_mask], dim=1)  # [B_TT, M + 1]

        # Pass through transformer
        transformer_output = self.transformer(tokens, src_key_padding_mask=mask)

        # Extract CLS token output
        cls_output = transformer_output[:, 0]  # [B_TT, feat_dim]

        # Final normalization and projection
        cls_output = self.final_norm(cls_output)
        output = self.output_proj(cls_output)  # [B_TT, out_dim]

        td[self._name] = output
        return td


class ObsTokenCatFourier(LayerBase):
    """Concatenate attr embeds w coord embeds. Coord embeds are Fourier features."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        atr_embed_dim: int,
        feature_normalizations: list[float],
        num_freqs: int = 3,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._atr_embed_dim = atr_embed_dim  # Dimension of attribute embeddings
        self._num_freqs = num_freqs  # fourier feature frequencies
        self._coord_embed_dim = 4 * self._num_freqs  # x, y, sin, cos for each freq
        self._value_dim = 1
        self._feat_dim = self._atr_embed_dim + self._coord_embed_dim + self._value_dim
        self.M = obs_shape[0]
        self._feature_normalizations = list(feature_normalizations)
        self._max_embeds = 256
        self._mu = 11.0  # hardcoding 11 as the max coord value for now (range 0-10). can grab from mettagrid_env.py

    def _make_net(self) -> None:
        self._out_tensor_shape = [self.M, self._feat_dim]

        self._atr_embeds = nn.Embedding(self._max_embeds, self._atr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._atr_embeds.weight, std=0.02)

        # Create a tensor for feature normalizations
        # We need to handle the case where atr_idx might be 0 (padding) or larger than defined normalizations.
        # Assuming max atr_idx is 256 (same as atr_embeds size - 1 for padding_idx).
        # Initialize with 1.0 to avoid division by zero for unmapped indices.
        norm_tensor = torch.ones(self._max_embeds, dtype=torch.float32)
        for i, val in enumerate(self._feature_normalizations):
            if i < len(norm_tensor):  # Ensure we don't go out of bounds
                norm_tensor[i] = val
            else:
                raise ValueError(f"Feature normalization {val} is out of bounds for Embedding layer size {i}")
        self.register_buffer("_norm_factors", norm_tensor)
        self.register_buffer("frequencies", 2.0 ** torch.arange(self._num_freqs))

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

        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_embeds = self._atr_embeds(atr_indices)  # [B_TT, M, embed_dim]

        # Assemble feature vectors
        # Pre-allocating the tensor and filling it avoids multiple `torch.cat` calls,
        # which can be more efficient on GPU.
        feat_vectors = torch.empty(
            (*atr_embeds.shape[:-1], self._feat_dim),
            dtype=atr_embeds.dtype,
            device=atr_embeds.device,
        )
        feat_vectors[..., : self._atr_embed_dim] = atr_embeds

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).float()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).float()  # Shape: [B_TT, M]

        # Normalize coordinates to [-1, 1] based on the data range [0, 10]
        x_coords_norm = x_coord_indices / (self._mu - 1.0) * 2.0 - 1.0
        y_coords_norm = y_coord_indices / (self._mu - 1.0) * 2.0 - 1.0

        # Expand dims for broadcasting with frequencies
        x_coords_norm = x_coords_norm.unsqueeze(-1)  # [B_TT, M, 1]
        y_coords_norm = y_coords_norm.unsqueeze(-1)  # [B_TT, M, 1]

        # Get frequencies and reshape for broadcasting
        # self.frequencies is [f], reshape to [1, 1, f]
        frequencies = self.get_buffer("frequencies").view(1, 1, -1)

        # Compute scaled coordinates for Fourier features
        x_scaled = x_coords_norm * frequencies
        y_scaled = y_coords_norm * frequencies

        # Compute and place Fourier features directly into the feature vector
        offset = self._atr_embed_dim
        feat_vectors[..., offset : offset + self._num_freqs] = torch.cos(x_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.sin(x_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.cos(y_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.sin(y_scaled)

        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        # Gather normalization factors based on atr_indices
        norm_factors = self._norm_factors[atr_indices]  # Shape: [B_TT, M]

        # Normalize atr_values
        # no epsilon to prevent division by zero - we want to fail if we have a bad normalization
        normalized_atr_values = atr_values / (norm_factors)
        normalized_atr_values = normalized_atr_values.unsqueeze(-1)  # Shape: [B_TT, M, 1]

        # Place normalized attribute values in the feature vector
        feat_vectors[..., self._atr_embed_dim + self._coord_embed_dim :] = normalized_atr_values

        obs_mask = atr_indices == 0  # important! true means 0 ie mask me

        td[self._name] = feat_vectors
        td["obs_mask"] = obs_mask
        return td
