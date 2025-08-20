import torch
import torch.nn.functional as F
from torch import nn

# ------------------
# Utility functions
# ------------------


def trunc_normal_init_(tensor, mean=0.0, std=1.0):
    """Truncated normal initialization like in T5/ViT."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


def rms_norm(x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMSNorm (no bias)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + variance_epsilon)


# ------------------
# Core layers
# ------------------


class SwiGLU(nn.Module):
    """SwiGLU MLP block (Shazeer 2020)."""

    def __init__(self, hidden_size, expansion=4):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, expansion * hidden_size)
        self.w2 = nn.Linear(hidden_size, expansion * hidden_size)
        self.proj = nn.Linear(expansion * hidden_size, hidden_size)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """Multi-head self-attention w/ optional rotary embeddings."""

    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads=None, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.causal = causal

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim)
        self.proj = nn.Linear(num_heads * head_dim, hidden_size)

    def forward(self, hidden_states, cos_sin=None):
        B, T, C = hidden_states.size()
        qkv = self.qkv(hidden_states)  # (B, T, 3*H*D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, d)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if passed
        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos_sin)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.full((T, T), float("-inf"), device=attn_scores.device)
            mask = torch.triu(mask, diagonal=1)
            attn_scores = attn_scores + mask

        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ------------------
# Rotary Embeddings
# ------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self):
        return (self.cos, self.sin)


def apply_rotary_pos_emb(q, k, cos_sin):
    cos, sin = cos_sin
    # cos/sin: (T, dim/2) -> expand to (1, 1, T, dim)
    cos = cos[:, None, None, :].to(q.device)
    sin = sin[:, None, None, :].to(q.device)

    def rotary(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotary(q), rotary(k)


# ------------------
# Casted Layers
# ------------------


class CastedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02, cast_to=torch.float32):
        super().__init__(num_embeddings, embedding_dim)
        trunc_normal_init_(self.weight, std=init_std)
        self.cast_to = cast_to

    def forward(self, x):
        return super().forward(x).to(self.cast_to)


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, cast_to=torch.float32):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            nn.init.zeros_(self.bias)
        self.cast_to = cast_to

    def forward(self, x):
        return super().forward(x).to(self.cast_to)


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std=0.02, cast_to=torch.float32):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
        trunc_normal_init_(self.emb.weight, std=init_std)
        self.cast_to = cast_to

    def forward(self, x):
        return self.emb(x).to(self.cast_to)
