class ObsLatentAttnBuiltIn(LayerBase):
    """
    Performs multi-layer cross-attention between learnable query tokens and input features using PyTorch's built-in MultiheadAttention.

    !!! Note About Output Shape: !!!
    The output shape depends on the `_use_cls_token` parameter:
    - If `_use_cls_token == True`, the output tensor shape will be `[B_TT, out_dim]`.
    - If `_use_cls_token == False`, the output tensor shape will be `[B_TT, num_query_tokens, out_dim]`.
    So, if true, it's setup to pass directly to the LSTM. But that also means that it will be in an invalid shape to
    pass to another attention layer. In other words, if _use_cls_token == True, then this should be the last layer of
    the encoder (because why else use the cls token?).

    Key Functionality (per layer):
    1. Multi-Head Cross-Attention: The current query tokens attend to the full sequence of
       input features (keys and values) using PyTorch's built-in MultiheadAttention.
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
        - `self._name`: Output tensor. Shape is `[B_TT, out_dim]` if `_use_cls_token == True`,
          or `[B_TT, num_query_tokens, out_dim]` if `_use_cls_token == False`.
    """

    def __init__(
        self,
        out_dim: int,
        use_mask: bool = False,
        num_query_tokens: int = 1,
        num_heads: int = 1,
        num_layers: int = 1,
        query_token_dim: Optional[int] = None,
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
        self._mlp_ratio = mlp_ratio
        self._use_cls_token = use_cls_token

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
        flip_pts = td["flip_pts"]
        B_TT = td["_BxTT_"]

        obs_per_batch = torch.unbind(x_features, dim=0)

        result = []
        for obs, flip_idx in zip(obs_per_batch, flip_pts, strict=False):
            # Slice from 0 to flip_idx
            sliced_obs = obs[:flip_idx]
            result.append(sliced_obs)

        x_features = torch.nested.nested_tensor(
            result, dtype=torch.float, device=x_features.device, layout=torch.jagged
        )

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

            attn_output = F.scaled_dot_product_attention(q_p, k_p, v_p)
            attn_output = einops.rearrange(attn_output, "b h q d -> b q (h d)")
            attn_output = layer["attn_out_proj"](attn_output)

            # reesidgual konnectshun
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
