import math
import warnings
from types import SimpleNamespace
from typing import Dict, List, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn

from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

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

        self.q_head = CastedLinear(self.hidden_size, 2, bias=True, cast_to=self.forward_dtype)

        # LM Blocks
        if self.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // self.num_heads,  # Match the head_dim used in Attention
                max_position_embeddings=self.seq_len,
                base=int(self.rope_theta),
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
            pass
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

        # New carry no grad and hidden state output
        new_carry = type("Carry", (), {"z_H": z_H.detach(), "z_L": z_L.detach()})()
        hidden_state = z_H.mean(dim=1)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        print(f"q_logits shape: {q_logits.shape}")
        print(f"hidden_state shape: {hidden_state.shape}")

        return new_carry, hidden_state, (q_logits[..., 0], q_logits[..., 1])

    def encode_observations(self, observations):
        """
        Encode observations into a hidden representation.
        Input: observations with shape [batch_size, seq_len, 3]
        Output: encoded features with shape [batch_size, seq_len, hidden_size]
        """
        token_observations = observations
        B = token_observations.shape[0]
        TT = token_observations.shape[1] if token_observations.dim() == 3 else 1
        B_TT = B * TT

        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

        # Extract coordinates and attributes
        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = token_observations[..., 1].long()  # Shape: [B_TT, M]
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # Create mask for valid tokens
        valid_tokens = coords_byte != 0xFF

        # Additional validation: ensure atr_indices are within valid range
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr

        # Log warning for out-of-bounds indices
        invalid_atr_mask = valid_tokens & ~valid_atr
        if invalid_atr_mask.any():
            invalid_indices = atr_indices[invalid_atr_mask].unique()
            warnings.warn(
                f"Found observation attribute indices {sorted(invalid_indices.tolist())} "
                f">= num_layers ({self.num_layers}). These tokens will be ignored.",
                stacklevel=2,
            )

        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices  # [B_TT, M]
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index  # [B_TT, M]

        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scale
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        # Flatten spatial dimensions and project to hidden size
        box_obs_flat = box_obs.view(B_TT, -1)  # [B_TT, num_layers * width * height]

        # Project to hidden size using the pre-initialized linear layer
        encoded = self.obs_projection(box_obs_flat)  # [B_TT, hidden_size]

        # Reshape back to [B, TT, hidden_size]
        if TT > 1:
            encoded = encoded.view(B, TT, self.hidden_size)

        return encoded


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
        new_inner_carry, hidden_state, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "hidden_state": hidden_state,
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


class HRM(PyTorchAgentMixin, nn.Module):
    """
    Hierarchical Reasoning Model with PyTorch mixin integration.

    This class inherits from PyTorchAgentMixin to provide shared functionality
    like weight clipping, TensorDict management, and action conversion utilities.
    """

    def __init__(self, env, **kwargs):
        # Extract mixin parameters
        mixin_params = self.extract_mixin_params(kwargs)

        # Initialize nn.Module
        nn.Module.__init__(self)

        # Initialize the policy
        self.policy = HRMPolicy(env, **kwargs)

        # Initialize mixin
        self.init_mixin(**mixin_params)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def activate_actions(self, action_names, action_max_params, device):
        """
        Initialize the action space, similar to MettaAgent.activate_actions.

        Args:
            action_names: List of action names
            action_max_params: List of maximum parameters for each action head
            device: Device to place tensors on
        """
        assert isinstance(action_max_params, list), "action_max_params must be a list"
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for action index conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long),
            dim=0,
        )

        # Create action_index tensor for conversion
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])
        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)

        # Set these on the policy as well for the mixin methods
        self.policy.action_index_tensor = self.action_index_tensor
        self.policy.cum_action_max_params = self.cum_action_max_params

        # Activate action embeddings on policy
        if hasattr(self.policy, "activate_action_embeddings"):
            self.policy.activate_action_embeddings(action_names, device)

    def forward(self, observations, state=None, action=None):
        """
        Forward pass through the HRM model.

        Args:
            observations: Input observation tensor, shape (B, TT, M, 3) or (B, M, 3)
            state: Dictionary with LSTM state and HRM carry
            action: Optional action tensor for training, shape (B, T, num_action_heads)

        Returns:
            Tuple of (actions, action_log_prob, entropy, value, logits_list)
        """
        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None, "carry": None}

        observations = observations.to(self.device)

        # Initialize carry if not provided
        if state.get("carry") is None:
            batch = {"env_obs": observations}
            state["carry"] = self.policy.backbone.initial_carry(batch)

        # Forward through policy
        new_carry, outputs = self.policy(state["carry"], {"env_obs": observations})

        # Extract logits and value
        logits = outputs["logits"]
        value = outputs["value"]

        # Handle training vs inference
        if action is not None:
            # Training mode - use provided action
            return self._forward_training(observations, action, logits, value)
        else:
            # Inference mode - sample actions
            return self._forward_inference(observations, logits, value)

    def _forward_training(self, observations, action, logits, value):
        """Forward pass for training mode."""
        # Reshape action if needed
        if action.dim() == 3:  # (B, T, A) -> (BT, A)
            batch_size_orig, time_steps, A = action.shape
            action = action.view(batch_size_orig * time_steps, A)

        # Convert action to logit index
        action_logit_index = self._convert_action_to_logit_index(action)

        # Calculate log probs and entropy
        action_log_probs = F.log_softmax(logits, dim=-1)
        action_probs = torch.exp(action_log_probs)

        batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
        selected_log_probs = action_log_probs[batch_indices, action_logit_index]
        entropy = -(action_probs * action_log_probs).sum(dim=-1)

        # Return in expected format
        return (
            action,  # actions
            selected_log_probs,  # action_log_prob
            entropy,  # entropy
            value,  # value
            logits,  # logits_list
        )

    def _forward_inference(self, observations, logits, value):
        """Forward pass for inference mode."""
        # Sample actions
        log_probs = F.log_softmax(logits, dim=-1)
        action_probs = torch.exp(log_probs)

        actions = torch.multinomial(action_probs, num_samples=1).view(-1)
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        selected_log_probs = log_probs[batch_indices, actions]

        # Convert to action format
        action = self._convert_logit_index_to_action(actions)

        # Calculate entropy
        entropy = -(action_probs * log_probs).sum(dim=-1)

        return (
            action,  # actions
            selected_log_probs,  # action_log_prob
            entropy,  # entropy
            value,  # value
            logits,  # logits_list
        )

    def encode_observations(self, observations, state=None):
        """Encode observations using the policy's encoding method."""
        return self.policy.encode_observations(observations, state)


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

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        self.critic_1 = nn.Linear(128, 1024)
        # value_head has no nonlinearity (YAML: nonlinearity: null), so gain=1
        self.value_head = nn.Linear(1024, 1)

        # Actor branch
        # actor_1 uses gain=1 (YAML default for Linear layers with ReLU)
        self.actor_1 = nn.Linear(128, 512)

        # Action embeddings - will be properly initialized via activate_action_embeddings
        self.action_embeddings = nn.Embedding(100, 16)

        self.actor_bias = nn.Parameter(torch.zeros(1, 1))
        self.actor_W = nn.Parameter(torch.randn(1, 1, 16))

        # Store for dynamic action head
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512

        # Track active actions
        self.active_action_names = []
        self.num_active_actions = 100  # Default

        self.effective_rank_enabled = True  # For critic_1 matching YAML

    def forward(self, carry, batch: Dict[str, torch.Tensor]):
        new_carry, outputs = self.backbone(carry, batch)

        logits, value = self.decode_actions(outputs["hidden_state"].to(torch.float32), batch["env_obs"].shape[0])

        print(f"logits shape: {logits.shape}")
        print(f"value shape: {value.shape}")

        return new_carry, outputs

    def decode_actions(self, hidden, batch_size):
        """Decode actions using bilinear interaction to match MettaActorSingleHead."""
        # Critic branch (unchanged)
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)  # [B*TT, 512]
        actor_features = F.relu(actor_features)  # ComponentPolicy has ReLU after actor_1

        # Get action embeddings for all actions
        # Use only the active actions (first num_active_actions embeddings)
        action_embeds = self.action_embeddings.weight[: self.num_active_actions]  # [num_actions, 16]

        # Expand action embeddings for each batch element
        action_embeds = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B*TT, num_actions, 16]

        # Bilinear interaction matching MettaActorSingleHead
        num_actions = action_embeds.shape[1]

        # Reshape for bilinear calculation
        # actor_features: [B*TT, 512] -> [B*TT * num_actions, 512]
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)  # [B*TT, num_actions, 512]
        actor_reshaped = actor_repeated.reshape(-1, self.actor_hidden_dim)  # [B*TT * num_actions, 512]
        action_embeds_reshaped = action_embeds.reshape(-1, self.action_embed_dim)  # [B*TT * num_actions, 16]

        # Perform bilinear operation using einsum (matching MettaActorSingleHead)
        query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)  # [N, 1, 16]
        query = torch.tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # [N, 1]

        biased_scores = scores + self.actor_bias  # [N, 1]

        # Reshape back to [B*TT, num_actions]
        logits = biased_scores.reshape(batch_size, num_actions)

        return logits, value


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
    batch = {"env_obs": obs}
    carry = policy.backbone.initial_carry({"env_obs": obs})
    print(f"obs shape: {obs.shape}")
    new_carry, outputs = policy(carry, batch)
    # print(f"hidden_state shape: {hidden_state.shape}")

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
