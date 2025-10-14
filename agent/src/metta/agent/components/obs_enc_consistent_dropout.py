from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous
from torchrl.modules import ConsistentDropout

from metta.agent.components.component_config import ComponentConfig


class ObsPerceiverLatentConsistentDropoutConfig(ComponentConfig):
    """Cross-attention encoder with consistent dropout for RL policy gradients.

    Consistent dropout ensures the same dropout mask is used during both
    rollout and gradient computation, preventing bias in policy gradients.
    """

    in_key: str
    out_key: str
    feat_dim: int
    latent_dim: int
    num_latents: int = 16
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: float = 4.0
    use_mask: bool = True
    pool: Literal["mean", "first", "none"] = "mean"
    name: str = "obs_perceiver_latent_consistent_dropout"
    dropout_p: float = 0.2

    def make_component(self, env=None):
        return ObsPerceiverLatentConsistentDropout(config=self)


class ObsPerceiverLatentConsistentDropout(nn.Module):
    """Cross-attention encoder with consistent dropout that maps input tokens to a fixed set of latent slots.

    Consistent dropout maintains the same mask across rollout and training,
    preventing biased policy gradients in RL settings.
    """

    def __init__(self, config: ObsPerceiverLatentConsistentDropoutConfig) -> None:
        super().__init__()
        self.config = config
        self._feat_dim = config.feat_dim
        self._latent_dim = config.latent_dim
        self._num_latents = config.num_latents
        self._num_heads = config.num_heads
        self._num_layers = config.num_layers
        self._mlp_ratio = config.mlp_ratio
        self._use_mask = config.use_mask
        self._pool = config.pool
        self._dropout_p = config.dropout_p

        if self._feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if self._latent_dim % self._num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.latents = nn.Parameter(torch.randn(1, self._num_latents, self._latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.token_norm = nn.LayerNorm(self._feat_dim)
        self.k_proj = nn.Linear(self._feat_dim, self._latent_dim, bias=False)
        self.v_proj = nn.Linear(self._feat_dim, self._latent_dim, bias=False)

        self.layers = nn.ModuleList([])
        self.attn_dropouts = nn.ModuleList([])
        self.mlp_dropouts = nn.ModuleList([])

        for _i in range(self._num_layers):
            layer_dict = nn.ModuleDict(
                {
                    "latent_norm": nn.LayerNorm(self._latent_dim),
                    "q_proj": nn.Linear(self._latent_dim, self._latent_dim, bias=False),
                    "attn_out_proj": nn.Linear(self._latent_dim, self._latent_dim),
                    "mlp_norm": nn.LayerNorm(self._latent_dim),
                    "mlp": nn.Sequential(
                        nn.Linear(self._latent_dim, int(self._latent_dim * self._mlp_ratio)),
                        nn.GELU(),
                        nn.Linear(int(self._latent_dim * self._mlp_ratio), self._latent_dim),
                    ),
                }
            )
            self.layers.append(layer_dict)

            if self._dropout_p > 0:
                self.attn_dropouts.append(ConsistentDropout(p=self._dropout_p))
                self.mlp_dropouts.append(ConsistentDropout(p=self._dropout_p))
            else:
                self.attn_dropouts.append(None)
                self.mlp_dropouts.append(None)

        self.final_norm = nn.LayerNorm(self._latent_dim)

    def get_agent_experience_spec(self) -> Composite:
        """Return experience spec for dropout masks that need to be stored in replay buffer."""
        spec = Composite()

        if self._dropout_p > 0:
            for i in range(self._num_layers):
                attn_mask_key = (self.config.name, f"attn_dropout_mask_{i}")
                mlp_mask_key = (self.config.name, f"mlp_dropout_mask_{i}")

                spec[attn_mask_key] = UnboundedContinuous(
                    shape=torch.Size([self._num_latents, self._latent_dim]),
                    dtype=torch.float32,
                )
                spec[mlp_mask_key] = UnboundedContinuous(
                    shape=torch.Size([self._num_latents, self._latent_dim]),
                    dtype=torch.float32,
                )

        return spec

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass with consistent dropout masks stored in tensordict."""
        x_features = td[self.config.in_key]
        key_mask = td.get("obs_mask") if self._use_mask else None
        tokens_norm = self.token_norm(x_features)
        k = self.k_proj(tokens_norm)
        v = self.v_proj(tokens_norm)

        k = einops.rearrange(k, "b m (h d) -> b h m d", h=self._num_heads)
        v = einops.rearrange(v, "b m (h d) -> b h m d", h=self._num_heads)

        attn_bias = None
        if key_mask is not None:
            mask_value = -torch.finfo(k.dtype).max
            attn_bias = einops.rearrange(key_mask.to(torch.bool), "b m -> b 1 1 m").to(k.dtype) * mask_value

        latents = self.latents.expand(x_features.shape[0], -1, -1)

        for i, layer in enumerate(self.layers):
            residual = latents
            q = layer["q_proj"](layer["latent_norm"](latents))
            q = einops.rearrange(q, "b n (h d) -> b h n d", h=self._num_heads)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            attn_output = einops.rearrange(attn_output, "b h n d -> b n (h d)")
            attn_output = layer["attn_out_proj"](attn_output)

            if self.attn_dropouts[i] is not None:
                mask_key_attn = (self.config.name, f"attn_dropout_mask_{i}")
                attn_mask = td.get(mask_key_attn, None)
                if self.training:
                    attn_output, attn_mask = self.attn_dropouts[i](attn_output, mask=attn_mask)
                    td[mask_key_attn] = attn_mask
                else:
                    attn_output = self.attn_dropouts[i](attn_output, mask=attn_mask)

            latents = residual + attn_output

            mlp_output = layer["mlp"](layer["mlp_norm"](latents))

            if self.mlp_dropouts[i] is not None:
                mask_key_mlp = (self.config.name, f"mlp_dropout_mask_{i}")
                mlp_mask = td.get(mask_key_mlp, None)
                if self.training:
                    mlp_output, mlp_mask = self.mlp_dropouts[i](mlp_output, mask=mlp_mask)
                    td[mask_key_mlp] = mlp_mask
                else:
                    mlp_output = self.mlp_dropouts[i](mlp_output, mask=mlp_mask)

            latents = latents + mlp_output

        latents = self.final_norm(latents)

        if self._pool == "mean":
            latents = latents.mean(dim=1)
        elif self._pool == "first":
            latents = latents[:, 0]
        elif self._pool == "none":
            latents = einops.rearrange(latents, "b n d -> b (n d)")
        else:
            raise ValueError("unsupported pool mode")

        td[self.config.out_key] = latents
        return td
