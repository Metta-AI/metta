import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torch.nn.attention.flex_attention import flex_attention, BlockMask


def apply_rope(inputs: Tensor, time_steps: Tensor, max_wavelength: int = 10_000) -> Tensor:
    """
    args:
    inputs -- (batch, sequence, head, hidden)
    position -- (batch, sequence) -- the value is the index of the position for the inputs
    max_wavelength -- int

    returns:
    (batch, sequence, head, hidden)
    """
    dtype = inputs.dtype
    device = inputs.device
    head_dim = inputs.shape[-1]

    fraction = 2 * torch.arange(0, head_dim // 2, dtype=dtype, device=device) / head_dim
    timescale = (max_wavelength**fraction).to(dtype=dtype, device=device)
    # timescale could be cached but I'm not sure it matter if torch compile is used

    sinusoid_inp = time_steps[..., None] / timescale[None, None, :]
    sinusoid_inp = sinusoid_inp[..., None, :]

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    first_half, second_half = torch.chunk(inputs, 2, -1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = torch.concatenate([first_part, second_part], -1)

    return out


def flex_attention_wrapper(query: Tensor, key: Tensor, value: Tensor, block_mask: BlockMask | None) -> Tensor:
    """
    Applies flex attention to sequence first data. Optionally takes in a block mask for casual attention,
    this has good performance because it can use sparsity to save compute on masked out parts.
    """

    # batch seq heads dim -> batch heads seq dim
    key = key.transpose(-3, -2).contiguous()
    query = query.transpose(-3, -2).contiguous()
    value = value.transpose(-3, -2).contiguous()

    out = flex_attention(query, key, value, block_mask=block_mask)

    batch, _, seq, _ = out.shape

    # batch heads seq qkv -> batch seq (heads qkv)
    # out = rearrange(out, "... heads seq dim -> ... seq (heads dim)")
    out = out.transpose(-3, -2).contiguous().reshape(batch, seq, -1)
    return out


def position_mask(time_steps: Tensor, max_seq_length: int) -> Tensor:
    """
    Generates a casual mask for einsum attention.
    args
    time_steps -- (batch, sequence), for inference this can be (batch, 1) where the value is the index of the position
    max_seq_length -- this should be the size of the kv cache
    """
    seq_range = torch.arange(max_seq_length, dtype=torch.int64, device=time_steps.device)
    mask = seq_range[None, None, :] <= time_steps[:, None]
    return mask[:, None, :, :]


def einsum_attention(query: Tensor, key: Tensor, value: Tensor, time_steps: Tensor) -> Tensor:
    """
    Computes casual self attention, this isn't as efficient as flex attention but can use different masking with recompiling with end to end torch.compile
    """

    max_seq_length = key.size(-3)
    mask = position_mask(time_steps, max_seq_length)

    depth = float(query.shape[-1])
    query = query / math.sqrt(depth)

    attn_weights = torch.einsum("...qhd,...khd->...hqk", query, key)

    big_neg = torch.finfo(attn_weights.dtype).min
    attn_weights = torch.where(mask, attn_weights, big_neg)

    attn_weights = F.softmax(attn_weights, -1)

    x = torch.einsum("...hqk,...khd->...qhd", attn_weights, value)

    batch, seq, _, _ = x.shape

    x = x.reshape(batch, seq, -1)
    return x


class AttentionBlock(nn.Module):
    def __init__(self, num_heads: int, d_model: int, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dtype = dtype
        self.has_kv_cache = False

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by 'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.d_model // self.num_heads

        self.in_proj = nn.Linear(
            self.d_model,
            self.num_heads * self.head_dim * 3,
            bias=True,
            dtype=self.dtype,
        )

        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.d_model,
            bias=True,
            dtype=self.dtype,
        )

    def ensure_kv_cache(self, batch_size: int, context_size: int, device: torch.device, dtype: torch.dtype):
        if self.has_kv_cache and self.key_cache.shape[0] == batch_size:
            return

        shape = (batch_size, context_size, self.num_heads, self.head_dim)
        key_cache = torch.zeros(shape, device=device, dtype=dtype)
        value_cache = torch.zeros(shape, device=device, dtype=dtype)

        self.register_buffer("key_cache", key_cache, persistent=False)
        self.register_buffer("value_cache", value_cache, persistent=False)
        self.has_kv_cache = True

    def update_kv_cache(self, time_steps: Tensor, key: Tensor, value: Tensor):
        batch_idx = torch.arange(self.key_cache.shape[0], device=time_steps.device, dtype=torch.int64)
        batch_idx = batch_idx[:, None]

        mod_time_steps = time_steps % self.key_cache.shape[1]
        self.key_cache[batch_idx, mod_time_steps] = key
        self.value_cache[batch_idx, mod_time_steps] = value

        return self.key_cache, self.value_cache

    def forward(self, inputs: torch.Tensor, time_steps: Tensor, block_mask: BlockMask | None = None) -> torch.Tensor:
        in_proj: torch.Tensor = self.in_proj(inputs)

        # batch seq (heads qkv) -> batch seq heads qkv
        batch, seq, _ = inputs.shape
        in_proj = in_proj.view(batch, seq, self.num_heads, -1)

        query, key, value = torch.chunk(in_proj, 3, -1)

        query = apply_rope(query, time_steps)
        key = apply_rope(key, time_steps)

        if self.has_kv_cache and not self.training:
            key, value = self.update_kv_cache(time_steps, key, value)

        if block_mask is not None:
            x = flex_attention_wrapper(query, key, value, block_mask=block_mask)
        else:
            x = einsum_attention(query, key, value, time_steps)
        x = self.out_proj(x)

        return x
