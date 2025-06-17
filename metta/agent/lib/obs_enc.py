from typing import Optional

import einops
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import LayerBase


class ObsVanillaAttn(LayerBase):
    """Future work can go beyond just using the feat dim as the attn qv dim, a single layer and single head,
    adding a GRU before the out projection..."""

    def __init__(
        self,
        out_dim: int,
        use_mask: bool = False,
        num_layers: int = 1,
        num_heads: int = 1,
        use_cls_token: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._use_cls_token = use_cls_token

    def _make_net(self) -> None:
        # we expect input shape to be [B, M, feat_dim]
        self._feat_dim = self._in_tensor_shapes[0][1]

        if self._feat_dim % self._num_heads != 0:
            raise ValueError(f"feat_dim ({self._feat_dim}) must be divisible by num_heads ({self._num_heads})")

        self._scale = (self._feat_dim // self._num_heads) ** -0.5

        self._out_tensor_shape = [0, self._out_dim]
        if self._use_cls_token:
            self._out_tensor_shape = [self._out_dim]
            self._cls_token = nn.Parameter(torch.randn(1, 1, self._feat_dim))

        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(self._feat_dim) for _ in range(self._num_layers)])
        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.out_projs = nn.ModuleList()

        for _ in range(self._num_layers):
            self.q_projs.append(nn.Linear(self._feat_dim, self._feat_dim, bias=False))
            self.k_projs.append(nn.Linear(self._feat_dim, self._feat_dim, bias=False))
            self.v_projs.append(nn.Linear(self._feat_dim, self._feat_dim, bias=False))
            self.out_projs.append(nn.Linear(self._feat_dim, self._feat_dim))

        self._layer_norm_2 = nn.LayerNorm(self._feat_dim)

        if self._feat_dim != self._out_dim:
            self._final_proj = nn.Linear(self._feat_dim, self._out_dim)
        else:
            self._final_proj = nn.Identity()

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]
        if self._use_cls_token:
            x_features = torch.cat([self._cls_token.expand(x_features.shape[0], -1, -1), x_features], dim=1)

        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]  # True for elements to be masked
            if self._use_cls_token:
                key_mask = torch.cat([torch.zeros(key_mask.shape[0], 1, device=key_mask.device), key_mask], dim=1)

        x = x_features
        for i in range(self._num_layers):
            x_res = x
            x_norm = self.layer_norms_1[i](x)

            q = self.q_projs[i](x_norm)
            k = self.k_projs[i](x_norm)
            v = self.v_projs[i](x_norm)

            # Reshape for multi-head
            q = einops.rearrange(q, "b m (h d) -> b h m d", h=self._num_heads)
            k = einops.rearrange(k, "b m (h d) -> b h m d", h=self._num_heads)
            v = einops.rearrange(v, "b m (h d) -> b h m d", h=self._num_heads)

            # Attention scores: [B, num_heads, M, M]
            attn_scores = torch.einsum("bhmd,bhnd->bhmn", q, k) * self._scale

            if key_mask is not None:
                # key_mask: [B, M] -> [B, 1, 1, M] for broadcasting
                mask_value = -torch.finfo(attn_scores.dtype).max
                attn_scores = attn_scores + key_mask.unsqueeze(1).unsqueeze(1).to(attn_scores.dtype) * mask_value

            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, M, M]

            # Weighted sum of V: [B, num_heads, M, head_dim]
            attn_output = torch.einsum("bhmn,bhnd->bhmd", attn_weights, v)

            # Combine heads: [B, M, feat_dim]
            attn_output = einops.rearrange(attn_output, "b h m d -> b m (h d)")

            attn_output = self.out_projs[i](attn_output)

            x = x_res + attn_output

        x = self._layer_norm_2(x)

        x = self._final_proj(x)

        if self._use_cls_token:
            x = x[:, 0]

        td[self._name] = x
        return td


class ObsLatentAttn(LayerBase):
    """
    Performs multi-layer cross-attention between learnable query tokens and input features.

    !!! Note About Output Shape: !!!
    The output shape depends on the `_use_cls_token` parameter:
    - If `_use_cls_token == True`, the output tensor shape will be `[B_TT, out_dim]`.
    - If `_use_cls_token == False`, the output tensor shape will be `[B_TT, num_query_tokens, out_dim]`.
    So, if true, it's setup to pass directly to the LSTM. But that also means that it will be in an invalid shape to
    pass to another attention layer. In other words, if _use_cls_token == True, then this should be the last layer of
    the encoder (because why else use the cls token?).

    Key Functionality (per layer):
    1. Multi-Head Cross-Attention: The current query tokens attend to the full sequence of
       input features (keys and values).
    2. Residual Connection and Layer Normalization.
    3. Feed-Forward Network (MLP): A position-wise MLP is applied to each query token.
    4. Another Residual Connection and Layer Normalization.

    Args:
        out_dim (int): The final output dimension for each query token's processed features.
        use_mask (bool, optional): If True, uses an observation mask (`obs_mask` from the
            input TensorDict) to mask attention scores for padded elements in `x_features`.
            Defaults to False.
        num_query_tokens (int, optional): The number of learnable query tokens to use.
            Defaults to 1.
        num_heads (int, optional): The number of attention heads. Defaults to 1.
        num_layers (int, optional): The number of cross-attention blocks. Defaults to 1.
        query_token_dim (Optional[int], optional): The embedding dimension for the initial
            learnable query tokens and the hidden dimension throughout the layers.
            If None, defaults to the input feature dimension (`feat_dim`). Defaults to None.
        qk_dim (Optional[int], optional): The dimension for query and key projections in the
            attention mechanism. If None, defaults to `query_token_dim`. Defaults to None.
        v_dim (Optional[int], optional): The dimension for value projection. For multi-layer
            architectures, this must be equal to `query_token_dim` to allow for residual
            connections. If None, defaults to `query_token_dim`. Defaults to None.
        mlp_ratio (float, optional): Determines the hidden dimension of the per-layer feed-forward
            network as `mlp_ratio * query_token_dim`. Defaults to 4.0.
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
        num_heads: int = 1,
        num_layers: int = 1,
        query_token_dim: Optional[int] = None,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        use_cls_token: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask
        self._num_query_tokens = num_query_tokens
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._query_token_dim = query_token_dim
        assert self._num_query_tokens > 0, "num_query_tokens must be greater than 0"
        self._qk_dim = qk_dim
        self._v_dim = v_dim
        self._mlp_ratio = mlp_ratio
        self._use_cls_token = use_cls_token  # simply output one latent token (the same one)

    def _make_net(self) -> None:
        self._out_tensor_shape = [self._num_query_tokens, self._out_dim]
        if self._use_cls_token:
            self._out_tensor_shape = [self._out_dim]

        # we expect input shape to be [B, M, feat_dim] where we don't know M
        self._feat_dim = self._in_tensor_shapes[0][1]

        if self._query_token_dim is None:
            self._query_token_dim = self._feat_dim
        if self._qk_dim is None:
            self._qk_dim = self._query_token_dim
        if self._v_dim is None:
            self._v_dim = self._query_token_dim

        if self._qk_dim % self._num_heads != 0:
            raise ValueError(f"qk_dim ({self._qk_dim}) must be divisible by num_heads ({self._num_heads})")
        if self._v_dim % self._num_heads != 0:
            raise ValueError(f"v_dim ({self._v_dim}) must be divisible by num_heads ({self._num_heads})")
        if self._num_layers > 1 and self._query_token_dim != self._v_dim:
            raise ValueError(
                f"For multi-layer cross attention (num_layers > 1), query_token_dim ({self._query_token_dim}) must"
                f"equal v_dim ({self._v_dim}) for residual connections."
            )

        self._scale = (self._qk_dim // self._num_heads) ** -0.5

        self._q_token = nn.Parameter(torch.randn(1, self._num_query_tokens, self._query_token_dim))
        nn.init.trunc_normal_(self._q_token, std=0.02)

        self.norm_kv = nn.LayerNorm(self._feat_dim)
        self.k_proj = nn.Linear(self._feat_dim, self._qk_dim, bias=False)
        self.v_proj = nn.Linear(self._feat_dim, self._v_dim, bias=False)

        self.layers = nn.ModuleList([])
        for _ in range(self._num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(self._query_token_dim, self._qk_dim, bias=False),
                        "attn_out_proj": nn.Linear(self._v_dim, self._query_token_dim),
                        "norm1": nn.LayerNorm(self._query_token_dim),
                        "norm2": nn.LayerNorm(self._query_token_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(self._query_token_dim, int(self._query_token_dim * self._mlp_ratio)),
                            nn.GELU(),
                            nn.Linear(int(self._query_token_dim * self._mlp_ratio), self._query_token_dim),
                        ),
                    }
                )
            )

        self.final_norm = nn.LayerNorm(self._query_token_dim)

        if self._query_token_dim == self._out_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(self._query_token_dim, self._out_dim)

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]
        key_mask = None
        if self._use_mask:
            key_mask = td["obs_mask"]
        B_TT = td["_BxTT_"]

        queries = self._q_token.expand(B_TT, -1, -1)

        kv_norm = self.norm_kv(x_features)
        k_p = self.k_proj(kv_norm)
        v_p = self.v_proj(kv_norm)

        k_p = einops.rearrange(k_p, "b m (h d) -> b h m d", h=self._num_heads)
        v_p = einops.rearrange(v_p, "b m (h d) -> b h m d", h=self._num_heads)

        for layer in self.layers:
            # Attention block
            queries_res = queries
            queries_norm = layer["norm1"](queries)
            q_p = layer["q_proj"](queries_norm)
            q_p = einops.rearrange(q_p, "b q (h d) -> b h q d", h=self._num_heads)

            attn_scores = torch.einsum("bhqd,bhkd->bhqk", q_p, k_p) * self._scale

            if key_mask is not None:
                mask_value = -torch.finfo(attn_scores.dtype).max
                # key_mask: [B_TT, M] -> [B_TT, 1, 1, M] for broadcasting
                attn_scores = attn_scores + key_mask.unsqueeze(1).unsqueeze(1).to(attn_scores.dtype) * mask_value

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v_p)
            attn_output = einops.rearrange(attn_output, "b h q d -> b q (h d)")
            attn_output = layer["attn_out_proj"](attn_output)

            queries = queries_res + attn_output

            # MLP block
            queries_res = queries
            queries_norm = layer["norm2"](queries)
            mlp_output = layer["mlp"](queries_norm)
            queries = queries_res + mlp_output

        x = self.final_norm(queries)
        x = self.output_proj(x)

        if self._use_cls_token:
            # Select first query token from [B_TT, num_query_tokens, self._out_dim] to [B_TT, self._out_dim]
            x = x[:, 0]

        td[self._name] = x
        return td


class ObsLatentAttnBisect(LayerBase):
    def __init__(
        self,
        out_dim: int,
        use_mask: bool = False,
        num_query_tokens: int = 1,
        num_heads: int = 1,
        num_layers: int = 1,
        query_token_dim: Optional[int] = None,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        use_cls_token: bool = False,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._out_dim = out_dim
        self._use_mask = use_mask
        self._num_query_tokens = num_query_tokens
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._query_token_dim = query_token_dim
        assert self._num_query_tokens > 0, "num_query_tokens must be greater than 0"
        self._qk_dim = qk_dim
        self._v_dim = v_dim
        self._mlp_ratio = mlp_ratio
        self._use_cls_token = use_cls_token  # simply output one latent token (the same one)

    def _make_net(self) -> None:
        self._out_tensor_shape = [self._out_dim]
        # self._out_tensor_shape = [self._num_query_tokens, self._out_dim]
        # if self._use_cls_token:
        #     self._out_tensor_shape = [self._out_dim]

        # # we expect input shape to be [B, M, feat_dim] where we don't know M
        # self._feat_dim = self._in_tensor_shapes[0][1]

        # if self._query_token_dim is None:
        #     self._query_token_dim = self._feat_dim
        # if self._qk_dim is None:
        #     self._qk_dim = self._query_token_dim
        # if self._v_dim is None:
        #     self._v_dim = self._query_token_dim

        # if self._qk_dim % self._num_heads != 0:
        #     raise ValueError(f"qk_dim ({self._qk_dim}) must be divisible by num_heads ({self._num_heads})")
        # if self._v_dim % self._num_heads != 0:
        #     raise ValueError(f"v_dim ({self._v_dim}) must be divisible by num_heads ({self._num_heads})")
        # if self._num_layers > 1 and self._query_token_dim != self._v_dim:
        #     raise ValueError(
        #         f"For multi-layer cross attention (num_layers > 1), query_token_dim ({self._query_token_dim}) must"
        #         f"equal v_dim ({self._v_dim}) for residual connections."
        #     )

        # self._scale = (self._qk_dim // self._num_heads) ** -0.5

        # self._q_token = nn.Parameter(torch.randn(1, self._num_query_tokens, self._query_token_dim))
        # nn.init.trunc_normal_(self._q_token, std=0.02)

        # self.norm_kv = nn.LayerNorm(self._feat_dim)
        # self.k_proj = nn.Linear(self._feat_dim, self._qk_dim, bias=False)
        # self.v_proj = nn.Linear(self._feat_dim, self._v_dim, bias=False)

        # self.q_proj = nn.Linear(self._query_token_dim, self._qk_dim, bias=False)
        # self.attn_out_proj = nn.Linear(self._v_dim, self._query_token_dim)
        # self.norm1 = nn.LayerNorm(self._query_token_dim)
        # self.norm2 = nn.LayerNorm(self._query_token_dim)
        # self.MLP = nn.Sequential(
        #     nn.Linear(self._query_token_dim, int(self._query_token_dim * self._mlp_ratio)),
        #     nn.GELU(),
        #     nn.Linear(int(self._query_token_dim * self._mlp_ratio), self._query_token_dim),
        # )

        # # self.layers = nn.ModuleList([])
        # # for _ in range(self._num_layers):
        # #     self.layers.append(
        # #         nn.ModuleDict(
        # #             {
        # #                 "q_proj": nn.Linear(self._query_token_dim, self._qk_dim, bias=False),
        # #                 "attn_out_proj": nn.Linear(self._v_dim, self._query_token_dim),
        # #                 "norm1": nn.LayerNorm(self._query_token_dim),
        # #                 "norm2": nn.LayerNorm(self._query_token_dim),
        # #                 "mlp": nn.Sequential(
        # #                     nn.Linear(self._query_token_dim, int(self._query_token_dim * self._mlp_ratio)),
        # #                     nn.GELU(),
        # #                     nn.Linear(int(self._query_token_dim * self._mlp_ratio), self._query_token_dim),
        # #                 ),
        # #             }
        # #         )
        # #     )

        # self.final_norm = nn.LayerNorm(self._query_token_dim)

        # if self._query_token_dim == self._out_dim:
        #     self.output_proj = nn.Identity()
        # else:
        #     self.output_proj = nn.Linear(self._query_token_dim, self._out_dim)
        self.pre = torch.randn(1, 128)
        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        x_features = td[self._sources[0]["name"]]
        # key_mask = None
        # if self._use_mask:
        #     key_mask = td["obs_mask"]
        # B_TT = td["_BxTT_"]

        # queries = self._q_token.expand(B_TT, -1, -1)

        # kv_norm = self.norm_kv(x_features)
        # k_p = self.k_proj(kv_norm)
        # v_p = self.v_proj(kv_norm)

        # k_p = einops.rearrange(k_p, "b m (h d) -> b h m d", h=self._num_heads)
        # v_p = einops.rearrange(v_p, "b m (h d) -> b h m d", h=self._num_heads)

        # # for layer in self.layers:
        # # Attention block
        # queries_res = queries
        # queries_norm = self.norm1(queries)
        # q_p = self.q_proj(queries_norm)
        # q_p = einops.rearrange(q_p, "b q (h d) -> b h q d", h=self._num_heads)

        # attn_scores = torch.einsum("bhqd,bhkd->bhqk", q_p, k_p) * self._scale

        # if key_mask is not None:
        #     mask_value = -torch.finfo(attn_scores.dtype).max
        #     # key_mask: [B_TT, M] -> [B_TT, 1, 1, M] for broadcasting
        #     attn_scores = attn_scores + key_mask.unsqueeze(1).unsqueeze(1).to(attn_scores.dtype) * mask_value

        # attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v_p)
        # attn_output = einops.rearrange(attn_output, "b h q d -> b q (h d)")
        # attn_output = self.attn_out_proj(attn_output)

        # queries = queries_res + attn_output

        # # MLP block
        # queries_res = queries
        # queries_norm = self.norm2(queries)
        # mlp_output = self.MLP(queries_norm)
        # queries = queries_res + mlp_output

        # x = self.final_norm(queries)
        # x = self.output_proj(x)

        # if self._use_cls_token:
        #     # Select first query token from [B_TT, num_query_tokens, self._out_dim] to [B_TT, self._out_dim]
        #     x = x[:, 0]

        B_TT = x_features.shape[0]


        td[self._name] = self.pre.expand(B_TT, -1).to(x_features.device)
        return td


class ObsSlotAttn(LayerBase):
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
