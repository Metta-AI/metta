import math
import warnings
from types import SimpleNamespace
from typing import Dict, List, Tuple

import einops
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

    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads=None, causal=False, cast_to=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.causal = causal
        self.cast_to = cast_to

        self.qkv = CastedLinear(hidden_size, 3 * num_heads * head_dim, cast_to=cast_to)
        self.proj = CastedLinear(num_heads * head_dim, hidden_size, cast_to=cast_to)

    def forward(self, hidden_states, cos_sin=None):
        B, T, C = hidden_states.size()
        qkv = self.qkv(hidden_states)  # (B, T, 3*H*D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, d)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        # Apply rotary embeddings if passed
        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos_sin)
            print(f"After rotary: q shape: {q.shape}, k shape: {k.shape}")

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.full((T, T), float("-inf"), device=attn_scores.device, dtype=attn_scores.dtype)
            mask = torch.triu(mask, diagonal=1)
            attn_scores = attn_scores + mask

        attn = attn_scores.softmax(dim=-1)
        print(f"attn shape: {attn.shape}")
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        print(f"out shape before view: {(attn @ v).transpose(1, 2).contiguous().shape}")
        print(f"expected view shape: [{B}, {T}, {C}]")
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
    cos = cos[:, None, None, :].to(device=q.device, dtype=q.dtype)
    sin = sin[:, None, None, :].to(device=q.device, dtype=q.dtype)

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
        # Cast weights to the target dtype
        self.weight.data = self.weight.data.to(cast_to)
        if bias:
            self.bias.data = self.bias.data.to(cast_to)

    def forward(self, x):
        return super().forward(x.to(self.cast_to)).to(self.cast_to)


class ReasoningAttnBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, cast_to=torch.float32):
        super().__init__()

        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
            cast_to=cast_to,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=4)
        self.norm_eps = 1e-5

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin=None, **kwargs
    ) -> torch.Tensor:
        # Post Norm
        # hidden_states = rms_norm(
        #     hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps
        # )
        # # Fully Connected
        # hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)

        # Simple linear layer for simplicity

        print(f"hidden_states shape: {hidden_states.shape}")
        # Flatten the input
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, hidden_dim)
        linear_layer = nn.Linear(hidden_dim, hidden_dim).to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = linear_layer(hidden_states)
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_dim)
        # hidden_states = self.mlp(hidden_states)
        return hidden_states


class ReasoningBlock(nn.Module):
    def __init__(self, layers: List[ReasoningAttnBlock]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, input_injection=input_injection, **kwargs)

        return hidden_states


class HRM_ACTV1_Inner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        batch_size: int,
        pos_encodings: str,
        rope_theta: float,
        seq_len: int,
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
        self.batch_size = batch_size
        self.pos_encodings = pos_encodings
        self.rope_theta = rope_theta
        self.seq_len = seq_len
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
        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size, bias=False, cast_to=self.forward_dtype)
        self.q_head = CastedLinear(self.hidden_size, 2, bias=True, cast_to=self.forward_dtype)

        # LM Blocks
        if self.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // self.num_heads,  # Match the head_dim used in Attention
                max_position_embeddings=self.seq_len,
                base=self.rope_theta,
            )
        elif self.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.seq_len,
                self.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = ReasoningBlock(
            layers=[
                ReasoningAttnBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, cast_to=self.forward_dtype)
                for _i in range(self.H_layers)
            ]
        )
        self.L_level = ReasoningBlock(
            layers=[
                ReasoningAttnBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, cast_to=self.forward_dtype)
                for _i in range(self.L_layers)
            ]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hidden_size, dtype=self.forward_dtype), std=1), persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hidden_size, dtype=self.forward_dtype), std=1), persistent=True
        )

        # Observation encoding parameters (similar to fast.py)
        self.num_layers = 8  # Number of observation layers
        self.out_width = 16  # Spatial width
        self.out_height = 16  # Spatial height

        # Observation projection layer
        self.obs_projection = CastedLinear(
            self.num_layers * self.out_width * self.out_height, self.hidden_size, cast_to=self.forward_dtype
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor):
        # Check if input is observation features (shape [batch_size, seq_len, 3]) or token indices
        if input.shape[-1] == 3:
            # This is observation features, use observation encoding
            embedding = self.encode_observations(input)
        else:
            # This is token indices, use token embedding
            embedding = self.embed_tokens(input.to(torch.int32))

            # Position embeddings
            if self.pos_encodings == "learned":
                # scale by 1/sqrt(2) to maintain forward variance
                embedding = 0.707106781 * (
                    embedding + self.embed_pos.embedding_weight.to(device=embedding.device, dtype=embedding.dtype)
                )

            # Scale
            embedding = self.embed_scale * embedding

        return embedding

    def empty_carry(self, batch_size: int, device: torch.device):
        z_H = torch.empty(batch_size, self.seq_len, self.hidden_size, dtype=self.forward_dtype, device=device)
        z_L = torch.empty(batch_size, self.seq_len, self.hidden_size, dtype=self.forward_dtype, device=device)
        return type("Carry", (), {"z_H": z_H, "z_L": z_L})()

    def reset_carry(self, reset_flag: torch.Tensor, carry):
        H_init = self.H_init.to(device=carry.z_H.device, dtype=carry.z_H.dtype)
        L_init = self.L_init.to(device=carry.z_L.device, dtype=carry.z_L.dtype)
        z_H = torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H)
        z_L = torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L)
        return type("Carry", (), {"z_H": z_H, "z_L": z_L})()

    def forward(
        self, carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[object, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["env_obs"])

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

        print(f"z_H shape: {z_H.shape}, z_L shape: {z_L.shape}")

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = type("Carry", (), {"z_H": z_H.detach(), "z_L": z_L.detach()})()  # New carry no grad
        output = self.lm_head(z_H)
        output = nn.Linear(output.shape[-1], self.hidden_size, bias=False).to(dtype=output.dtype, device=output.device)(
            output
        )
        output = output.mean(dim=1)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        print(f"q_logits shape: {q_logits.shape}")
        print(f"output shape: {output.shape}")

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HRMBackbone(nn.Module):
    """Hierarchical Reasoning Backbone with ACT wrapper (no puzzle embedding)."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        batch_size: int,
        pos_encodings: str,
        rope_theta: float,
        seq_len: int,
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
        self.forward_dtype = getattr(torch, forward_dtype)

        self.inner = HRM_ACTV1_Inner(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            batch_size=batch_size,
            pos_encodings=pos_encodings,
            rope_theta=rope_theta,
            seq_len=seq_len,
            forward_dtype=forward_dtype,
            H_layers=H_layers,
            L_layers=L_layers,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            num_heads=num_heads,
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Initialize carry (state) for a new batch."""
        batch_size = batch["env_obs"].shape[0]
        device = batch["env_obs"].device

        return {
            "inner_carry": self.inner.empty_carry(batch_size, device),
            "steps": torch.zeros((batch_size,), dtype=torch.int32, device=device),
            "halted": torch.ones((batch_size,), dtype=torch.bool, device=device),  # start halted, reset on first step
            "current_data": {
                "env_obs": torch.empty_like(batch["env_obs"]),
                "mask": torch.empty_like(
                    batch.get("mask", torch.ones(batch["env_obs"].shape[:2], dtype=torch.bool, device=device))
                ),
            },
        }

    def forward(
        self, carry: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with ACT control."""
        device = batch["env_obs"].device

        # Reset halted sequences in carry
        new_inner_carry = self.inner.reset_carry(carry["halted"], carry["inner_carry"])
        new_steps = torch.where(carry["halted"], torch.zeros_like(carry["steps"]), carry["steps"])

        # Update current data only for halted sequences
        new_current_data = {
            "env_obs": torch.where(
                carry["halted"].view((-1, 1, 1)), batch["env_obs"], carry["current_data"]["env_obs"]
            ),
            "mask": torch.where(
                carry["halted"].view((-1, 1)),
                batch.get("mask", torch.ones(batch["env_obs"].shape[:2], dtype=torch.bool, device=device)),
                carry["current_data"]["mask"],
            ),
        }

        # Forward inner HRM
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step counter
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.halt_max_steps
            halted = is_last_step

            if self.training and (self.halt_max_steps > 1):
                # ACT halting logic
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration for stochastic halting
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.halt_exploration_prob) * torch.randint_like(
                    new_steps, low=2, high=self.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (like PQN idea)
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return {
            "inner_carry": new_inner_carry,
            "steps": new_steps,
            "halted": halted,
            "current_data": new_current_data,
        }, outputs


class HRMPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.backbone = HRMBackbone(
            hidden_size=128,
            vocab_size=10000,
            batch_size=1,
            pos_encodings="rope",
            rope_theta=10000,
            seq_len=200,  # Changed to match typical observation sequence length
            forward_dtype="float16",
            H_layers=1,
            L_layers=1,
            H_cycles=1,
            L_cycles=1,
            num_heads=1,
            halt_max_steps=10,
            halt_exploration_prob=0.1,
        )

        self.num_layers = 25
        self.is_continuous = False
        self.action_space = env.single_action_space

        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11

    def encode_observations(self, observations, state=None):
        """
        Encode observations into a hidden representation.

        This implementation matches ComponentPolicy's ObsTokenToBoxShaper exactly,
        using scatter operation for efficient token placement.
        """
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        B_TT = B * TT

        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

        # Don't modify original tensor - ComponentPolicy doesn't do this
        # token_observations[token_observations == 255] = 0  # REMOVED

        # Extract coordinates and attributes (matching ObsTokenToBoxShaper exactly)
        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = token_observations[..., 1].long()  # Shape: [B_TT, M]
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # Create mask for valid tokens (matching ComponentPolicy)
        valid_tokens = coords_byte != 0xFF

        # Additional validation: ensure atr_indices are within valid range
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr

        # Log warning for out-of-bounds indices (matching ComponentPolicy)
        invalid_atr_mask = valid_tokens & ~valid_atr
        if invalid_atr_mask.any():
            invalid_indices = atr_indices[invalid_atr_mask].unique()
            warnings.warn(
                f"Found observation attribute indices {sorted(invalid_indices.tolist())} "
                f">= num_layers ({self.num_layers}). These tokens will be ignored. "
                f"This may indicate the policy was trained with fewer observation channels.",
                stacklevel=2,
            )

        # Use scatter-based write to avoid multi-dim advanced indexing (matching ComponentPolicy)
        # Compute flattened spatial index and a combined index that encodes (layer, x, y)
        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices  # [B_TT, M]
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index  # [B_TT, M]

        # Mask out invalid entries by directing them to index 0 with value 0
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scatter values into a flattened buffer, then reshape to [B_TT, L, W, H]
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        print(f"Box Obs: {box_obs.shape}")

        self.cnn1 = nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Match YAML: Linear layers use orthogonal with gain=1
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.encoded_obs = nn.Linear(128, 128)

        box_obs = self.cnn2(self.cnn1(box_obs))
        box_obs = self.flatten(box_obs)
        box_obs = self.fc1(box_obs)
        encoded = self.encoded_obs(box_obs)

        return encoded

    def forward(self, obs: torch.Tensor):
        encoded = self.encode_observations(obs)
        return encoded


if __name__ == "__main__":
    import gymnasium as gym
    import numpy as np

    obs_shape = [34, 11, 11]
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
        obs_width=obs_shape[1],
        obs_height=obs_shape[2],
        single_action_space=gym.spaces.MultiDiscrete([9, 10]),
        feature_normalizations={},
        global_features=[],
    )

    policy = HRMPolicy(env)
    obs = torch.randint(0, 8, (24, 200, 3), dtype=torch.uint8)
    print(f"obs shape: {obs.shape}")
    encoded = policy.encode_observations(obs)
    print(f"encoded shape: {encoded.shape}")

    # hrm = HRMBackbone(
    #     hidden_size=128,
    #     vocab_size=10000,
    #     batch_size=1,
    #     pos_encodings="rope",
    #     rope_theta=10000,
    #     seq_len=200,  # Changed to match typical observation sequence length
    #     forward_dtype="float16",
    #     H_layers=1,
    #     L_layers=1,
    #     H_cycles=1,
    #     L_cycles=1,
    #     num_heads=1,
    #     halt_max_steps=10,
    #     halt_exploration_prob=0.1,
    # )

    # # Test with observation features [batch_size, seq_len, 3]
    # # Use smaller values to avoid the warning about attribute indices >= num_layers
    # obs_features = torch.randint(0, 8, (1, 200, 3), dtype=torch.uint8)  # [1, 200, 3]
    # carry = hrm.initial_carry({"env_obs": obs_features})

    # new_carry, outputs = hrm(carry, {"env_obs": obs_features})
    # print(f"New carry keys: {new_carry.keys()}")
    # print(f"Outputs keys: {outputs.keys()}")
    # print(f"Logits shape: {outputs['logits'].shape}")
