from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


def _zero_masked_features(td: TensorDict, features: torch.Tensor) -> torch.Tensor:
    mask = td.get("obs_mask")
    if mask is not None:
        mask_bool = mask.to(torch.bool)
        features = features.masked_fill(einops.rearrange(mask_bool, "... -> ... 1"), 0.0)
    return features


class ObsAttrCoordEmbedConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_attr_coord_embed"
    attr_embed_dim: int

    def make_component(self, env=None):
        return ObsAttrCoordEmbed(config=self)


class ObsAttrCoordEmbed(nn.Module):
    """Embeds attr index as, separately embeds coords, then adds them together. Finally concatenate attr value to the
    end of the embedding. Learnable coord embeddings have performed worse than Fourier features as of 6/16/2025."""

    def __init__(
        self,
        config: ObsAttrCoordEmbedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self._attr_embed_dim = self.config.attr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._feat_dim = self._attr_embed_dim + self._value_dim
        self._max_embeds = 256

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)
        return None

    def forward(self, td: TensorDict) -> TensorDict:
        observations = td[self.config.in_key]

        coord_indices = observations[..., 0].long()
        coord_pair_embedding = self._coord_embeds(coord_indices)

        attr_indices = observations[..., 1].long()
        attr_embeds = self._attr_embeds(attr_indices)
        combined_embeds = attr_embeds + coord_pair_embedding

        attr_values = observations[..., 2].float()
        attr_values = einops.rearrange(attr_values, "... -> ... 1")

        feat_vectors = torch.empty(
            (*attr_embeds.shape[:-1], self._feat_dim),
            dtype=attr_embeds.dtype,
            device=attr_embeds.device,
        )
        feat_vectors[..., : self._attr_embed_dim] = combined_embeds
        feat_vectors[..., self._attr_embed_dim : self._attr_embed_dim + self._value_dim] = attr_values

        td[self.config.out_key] = _zero_masked_features(td, feat_vectors)

        return td


class ObsAttrEmbedFourierConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_attr_embed_fourier"
    attr_embed_dim: int = 12
    num_freqs: int = 6

    def make_component(self, env=None):
        return ObsAttrEmbedFourier(config=self)


class ObsAttrEmbedFourier(nn.Module):
    """An alternate to ObsAttrCoordEmbed that concatenates attr embeds w coord representation as Fourier features.

    The output feature dimension is calculated as:
    `output_dim = attr_embed_dim + (4 * num_freqs) + 1`

    Where:
    - `attr_embed_dim` is the dimension of the attribute embedding.
    - `num_freqs` is the number of frequencies for the Fourier features. The coordinate
      representation dimension is `4 * num_freqs` because we have sin and cos for
      both x and y coordinates for each frequency.
    - `1` is for the scalar attribute value that is concatenated at the end.
    """

    def __init__(self, config: ObsAttrEmbedFourierConfig) -> None:
        super().__init__()
        self.config = config
        self._attr_embed_dim = self.config.attr_embed_dim  # Dimension of attribute embeddings
        self._num_freqs = self.config.num_freqs  # fourier feature frequencies
        self._coord_rep_dim = 4 * self._num_freqs  # x, y, sin, cos for each freq
        self._value_dim = 1
        self._feat_dim = self._attr_embed_dim + self._coord_rep_dim + self._value_dim
        self._max_embeds = 256
        self._mu = 11.0  # hardcoding 11 as the max coord value for now (range 0-10). can grab from mettagrid_env.py

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        self.register_buffer("frequencies", 2.0 ** torch.arange(self._num_freqs))
        return None

    def forward(self, td: TensorDict) -> TensorDict:
        observations = td[self.config.in_key]

        # Attribute embeddings (pad idx 255 stays zeroed by nn.Embedding)
        attr_indices = observations[..., 1].long()
        attr_embeds = self._attr_embeds(attr_indices)

        # Preallocate output to avoid repeated torch.cat
        feat_vectors = torch.empty(
            (*attr_embeds.shape[:-1], self._feat_dim),
            dtype=attr_embeds.dtype,
            device=attr_embeds.device,
        )
        feat_vectors[..., : self._attr_embed_dim] = attr_embeds

        # coords_byte packs x/y into the high/low nibble
        coords_byte = observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).float()
        y_coord_indices = (coords_byte & 0x0F).float()

        # Normalize to [-1, 1] using known grid range (0-10)
        x_coords_norm = x_coord_indices / (self._mu - 1.0) * 2.0 - 1.0
        y_coords_norm = y_coord_indices / (self._mu - 1.0) * 2.0 - 1.0

        # Broadcast with frequency tensor
        x_coords_norm = einops.rearrange(x_coords_norm, "... -> ... 1")
        y_coords_norm = einops.rearrange(y_coords_norm, "... -> ... 1")
        frequencies = self.get_buffer("frequencies").view(1, 1, -1)
        x_scaled = x_coords_norm * frequencies
        y_scaled = y_coords_norm * frequencies

        # Populate Fourier blocks: [cos(x), sin(x), cos(y), sin(y)]
        offset = self._attr_embed_dim
        feat_vectors[..., offset : offset + self._num_freqs] = torch.cos(x_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.sin(x_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.cos(y_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.sin(y_scaled)

        # Append scalar attribute value
        feat_vectors[..., self._attr_embed_dim + self._coord_rep_dim :] = einops.rearrange(
            observations[..., 2].float(), "... -> ... 1"
        )

        td[self.config.out_key] = _zero_masked_features(td, feat_vectors)

        return td


class ObsAttrCoordValueEmbedConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_attr_coord_value_embed"
    attr_embed_dim: int = 12

    def make_component(self, env=None):
        return ObsAttrCoordValueEmbed(config=self)


class ObsAttrCoordValueEmbed(nn.Module):
    """An experiment that embeds attr value as a categorical variable. Using a normalization layer is not
    recommended."""

    def __init__(self, config: ObsAttrCoordValueEmbedConfig) -> None:
        super().__init__()
        self.config = config
        self._attr_embed_dim = self.config.attr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._max_embeds = 256

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._val_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._val_embeds.weight, std=0.02)

        return None

    def forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), attr_idx, attr_val
        observations = td[self.config.in_key]

        # The first element of an observation is the coordinate, which can be used as a direct index for embedding.
        coord_indices = observations[..., 0].long()
        coord_pair_embedding = self._coord_embeds(coord_indices)

        attr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        attr_embeds = self._attr_embeds(attr_indices)  # [B_TT, M, embed_dim]

        # The attribute value is treated as a categorical variable for embedding.
        val_indices = observations[..., 2].long()
        # Clip values to be within the embedding range to avoid errors, e.g. if a value is 256 for _max_embeds=256
        val_indices = torch.clamp(val_indices, 0, self._max_embeds - 1)
        val_embeds = self._val_embeds(val_indices)

        combined_embeds = attr_embeds + coord_pair_embedding + val_embeds

        td[self.config.out_key] = _zero_masked_features(td, combined_embeds)
        return td


class ObsPerceiverLatentConfig(ComponentConfig):
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
    name: str = "obs_perceiver_latent"

    def make_component(self, env=None):
        return ObsPerceiverLatent(config=self)


class ObsPerceiverLatent(nn.Module):
    """Cross-attention encoder that maps input tokens to a fixed set of latent slots."""

    def __init__(self, config: ObsPerceiverLatentConfig) -> None:
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
        for _ in range(self._num_layers):
            self.layers.append(
                nn.ModuleDict(
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
            )

        self.final_norm = nn.LayerNorm(self._latent_dim)

    def forward(self, td: TensorDict) -> TensorDict:
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

        for layer in self.layers:
            residual = latents
            q = layer["q_proj"](layer["latent_norm"](latents))
            q = einops.rearrange(q, "b n (h d) -> b h n d", h=self._num_heads)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            attn_output = einops.rearrange(attn_output, "b h n d -> b n (h d)")
            latents = residual + layer["attn_out_proj"](attn_output)

            latents = latents + layer["mlp"](layer["mlp_norm"](latents))

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
