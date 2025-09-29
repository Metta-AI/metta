"""Swin-style observation encoder."""

from __future__ import annotations

from typing import Literal, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class ObsSwinEncoderConfig(ComponentConfig):
    """Configuration for the Swin-like observation encoder."""

    in_key: str
    out_key: str
    tokens_key: str
    token_feat_dim: int
    name: str = "obs_swin_encoder"
    embed_dim: int = 96
    depth: int = 2
    num_heads: int = 4
    window_size: int = 4
    patch_size: int = 4
    mlp_ratio: float = 4.0
    use_shifted_window: bool = True
    drop_path: float = 0.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    pool: Literal["mean", "first", "flatten"] = "mean"
    out_dim: int = 256
    mask_key: Optional[str] = "obs_mask"

    def make_component(self, env=None):
        return ObsSwinEncoder(config=self, env=env)


class DropPath(nn.Module):
    """Stochastic depth per sample.

    Copied from timm implementation (Apache-2.0) but simplified to avoid dependency.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition input into non-overlapping windows."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverse window partition back to original layout."""
    b = int(windows.shape[0] // (h * w / (window_size * window_size)))
    windows = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            # mask shape: (num_windows, N, N)
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.num_heads, n, n)
            attn = attn + einops.rearrange(mask, "nw n1 n2 -> 1 nw 1 n1 n2")
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, attn_dropout, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, dropout)

        self._attn_mask: torch.Tensor | None = None
        self._mask_hw: tuple[int, int] | None = None

    def _build_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        if self._attn_mask is not None and self._mask_hw == (h, w):
            return self._attn_mask.to(device)

        img_mask = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for i in range(0, h, self.window_size):
            for j in range(0, w, self.window_size):
                img_mask[:, i : i + self.window_size, j : j + self.window_size, :] = cnt
                cnt += 1

        shifted_mask = torch.roll(img_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        mask_windows = window_partition(shifted_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = einops.rearrange(mask_windows, "nw n -> nw n 1") - einops.rearrange(mask_windows, "nw n -> nw 1 n")
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        self._attn_mask = attn_mask
        self._mask_hw = (h, w)
        return attn_mask

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, num_tokens, c = x.shape
        if num_tokens != h * w:
            raise ValueError("Input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        attn_mask = self._build_mask(h, w, x.device)
        attn_windows = self.attn(x_windows, attn_mask)

        x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(b, num_tokens, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ObsSwinEncoder(nn.Module):
    """Swin-style encoder operating directly on token observations."""

    def __init__(self, config: ObsSwinEncoderConfig, env=None) -> None:
        super().__init__()
        if env is None:
            raise ValueError("Env is required to determine observation dimensions")
        self.config = config
        self.obs_height = env.obs_height
        self.obs_width = env.obs_width

        if self.obs_height % self.config.patch_size != 0 or self.obs_width % self.config.patch_size != 0:
            raise ValueError(
                "Observation dims must be divisible by patch_size. "
                f"Got ({self.obs_height}, {self.obs_width}) with patch_size {self.config.patch_size}."
            )
        if self.config.use_shifted_window and self.config.window_size % 2 != 0:
            raise ValueError("window_size must be even when using shifted windows")
        if self.config.token_feat_dim <= 0:
            raise ValueError("token_feat_dim must be positive")

        self.tokens_key = self.config.tokens_key
        self.mask_key = self.config.mask_key
        self.num_patches_x = self.obs_width // self.config.patch_size
        self.num_patches_y = self.obs_height // self.config.patch_size
        if self.num_patches_x % self.config.window_size != 0 or self.num_patches_y % self.config.window_size != 0:
            raise ValueError(
                "Patch grid must be divisible by window_size. "
                f"Got ({self.num_patches_y}, {self.num_patches_x}) patches with window_size {self.config.window_size}."
            )
        self.num_tokens = self.num_patches_x * self.num_patches_y

        self.token_norm = nn.LayerNorm(self.config.token_feat_dim)
        self.input_proj = nn.Linear(self.config.token_feat_dim, self.config.embed_dim)
        self.pos_drop = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList()
        for layer_idx in range(self.config.depth):
            shift = 0
            if self.config.use_shifted_window and layer_idx % 2 == 1:
                shift = self.config.window_size // 2
            self.blocks.append(
                SwinBlock(
                    dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    window_size=self.config.window_size,
                    shift_size=shift,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                    attn_dropout=self.config.attn_dropout,
                    drop_path=self.config.drop_path,
                )
            )

        self.norm = nn.LayerNorm(self.config.embed_dim)

        if self.config.pool == "flatten":
            out_in_features = self.config.embed_dim * self.num_tokens
        else:
            out_in_features = self.config.embed_dim
        self.output_proj = nn.Linear(out_in_features, self.config.out_dim)

    def forward(self, td: TensorDict) -> TensorDict:
        token_features = td[self.config.in_key]
        tokens = td[self.tokens_key]
        if token_features.shape[:2] != tokens.shape[:2]:
            raise ValueError("Token feature shape must align with raw token shape")
        mask = None
        if self.mask_key is not None and self.mask_key in td.keys():
            mask = td[self.mask_key]

        patch_tokens = self._tokens_to_patches(token_features, tokens, mask)
        h = self.num_patches_y
        w = self.num_patches_x
        if patch_tokens.shape[1] != self.num_tokens:
            raise ValueError("Aggregated patch count mismatch")
        x = self.pos_drop(patch_tokens)

        for block in self.blocks:
            x = block(x, h, w)

        x = self.norm(x)

        if self.config.pool == "mean":
            x = x.mean(dim=1)
        elif self.config.pool == "first":
            x = x[:, 0]
        elif self.config.pool == "flatten":
            x = einops.rearrange(x, "b n d -> b (n d)")
        else:
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")

        x = self.output_proj(x)
        td[self.config.out_key] = x
        return td

    def _tokens_to_patches(
        self,
        token_features: torch.Tensor,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, M, feat_dim = token_features.shape
        if feat_dim != self.config.token_feat_dim:
            raise ValueError(f"token_feat_dim mismatch: expected {self.config.token_feat_dim}, got {feat_dim}.")

        device = token_features.device
        normed_features = self.token_norm(token_features)
        projected = self.input_proj(normed_features)

        coords_byte = tokens[..., 0].to(torch.int64)
        x_coords = (coords_byte >> 4) & 0x0F
        y_coords = coords_byte & 0x0F

        patch_x = torch.div(x_coords, self.config.patch_size, rounding_mode="floor")
        patch_y = torch.div(y_coords, self.config.patch_size, rounding_mode="floor")
        patch_x = patch_x.clamp(max=self.num_patches_x - 1)
        patch_y = patch_y.clamp(max=self.num_patches_y - 1)
        patch_ids = patch_y * self.num_patches_x + patch_x

        if mask is None:
            invalid = coords_byte == 0xFF
        else:
            invalid = mask.bool()
        valid = ~invalid

        num_patches = self.num_tokens
        agg = torch.zeros(B * num_patches, self.config.embed_dim, device=device, dtype=projected.dtype)
        counts = torch.zeros(B * num_patches, 1, device=device, dtype=projected.dtype)

        batch_idx = einops.repeat(torch.arange(B, device=device), "b -> b m", m=M)
        flat_indices = einops.rearrange(batch_idx * num_patches + patch_ids, "b m -> (b m)")
        flat_valid = einops.rearrange(valid, "b m -> (b m)")

        if flat_valid.any():
            patch_indices = flat_indices[flat_valid]
            flat_feats = einops.rearrange(projected, "b m d -> (b m) d")[flat_valid]
            agg.index_add_(0, patch_indices, flat_feats)
            ones = einops.rearrange(torch.ones_like(patch_indices, dtype=projected.dtype, device=device), "n -> n 1")
            counts.index_add_(0, patch_indices, ones)

        counts = counts.clamp_min(1.0)
        agg = agg / counts
        agg = agg.view(B, num_patches, self.config.embed_dim)
        return agg
