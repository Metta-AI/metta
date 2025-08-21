from ast import Tuple
import torch
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



class ReasoningAttnBlock(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()

        self.self_attn = Attention(
            hidden_size=hidden_size, head_dim=hidden_size // 8, num_heads=8, num_key_value_heads=8, causal=False
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=4)
        self.norm_eps = 1e-5

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Post Norm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class ReasoningBlock(nn.Module):
    def __init__(self, layers: List[ReasoningAttnBlock]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


import math


class HRM_ACTV1_Inner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        puzzle_emb_ndim: int,
        batch_size: int,
        pos_encodings: str,
        rope_theta: float,
        seq_len: int,
        num_puzzle_identifiers: int,
        forward_dtype: str,
        H_layers: int,
        L_layers: int,
        H_cycles: int,
        L_cycles: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.batch_size = batch_size
        self.pos_encodings = pos_encodings
        self.rope_theta = rope_theta
        self.seq_len = seq_len
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.num_heads = num_heads
        self.forward_dtype = getattr(torch, forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.vocab_size, self.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size, bias=False)
        self.q_head = CastedLinear(self.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.puzzle_emb_ndim // -self.hidden_size)  # ceil div
        if self.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.num_puzzle_identifiers,
                self.puzzle_emb_ndim,
                batch_size=self.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // self.num_heads,
                max_position_embeddings=self.seq_len + self.puzzle_emb_len,
                base=self.rope_theta,
            )
        elif self.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.seq_len + self.puzzle_emb_len,
                self.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = ReasoningBlock(
            layers=[ReasoningAttnBlock(hidden_size=self.hidden_size) for _i in range(self.H_layers)]
        )
        self.L_level = ReasoningBlock(
            layers=[ReasoningAttnBlock(hidden_size=self.hidden_size) for _i in range(self.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hidden_size, dtype=self.forward_dtype), std=1), persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hidden_size, dtype=self.forward_dtype), std=1), persistent=True
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return (
            z_H=torch.empty(batch_size, self.seq_len + self.puzzle_emb_len, self.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.seq_len + self.puzzle_emb_len, self.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry):
        return (
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self, carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.H_cycles):
                for _L_step in range(self.L_cycles):
                    if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = (z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])




class HRMBackbone(nn.Module):
    """ACT wrapper."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        puzzle_emb_ndim: int,
        batch_size: int,
        pos_encodings: str,
        rope_theta: float,
        seq_len: int,
        num_puzzle_identifiers: int,
        forward_dtype: str,
        H_layers: int,
        L_layers: int,
        H_cycles: int,
        L_cycles: int,
        num_heads: int,
        halt_max_steps: int,
        halt_exploration_prob: float,
    ):
        super().__init__()
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob

        self.inner = HRM_ACTV1_Inner(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            puzzle_emb_ndim=puzzle_emb_ndim,
            batch_size=batch_size,
            pos_encodings=pos_encodings,
            rope_theta=rope_theta,
            seq_len=seq_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            forward_dtype=forward_dtype,
            H_layers=H_layers,
            L_layers=L_layers,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            num_heads=num_heads,
        )

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]

                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return (new_inner_carry, new_steps, halted, new_current_data), outputs
