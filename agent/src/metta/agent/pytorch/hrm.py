import math
import warnings
from types import SimpleNamespace
from typing import Dict, List, Tuple

import einops
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.policy import Policy

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


# Memory
class HRMMemory:
    def __init__(self):
        self.carry = {}

    def has_memory(self):
        return True

    def set_memory(self, memory):
        self.carry = memory

    def get_memory(self):
        return self.carry

    def reset_memory(self):
        self.carry = {}

    def reset_env_memory(self, env_id):
        if env_id in self.carry:
            del self.carry[env_id]


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
    """Multi-head self-attention."""

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

    def forward(self, hidden_states):
        B, T, C = hidden_states.size()
        qkv = self.qkv(hidden_states)  # (B, T, 3*H*D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, d)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.full((T, T), float("-inf"), device=attn_scores.device, dtype=attn_scores.dtype)
            mask = torch.triu(mask, diagonal=1)
            attn_scores = attn_scores + mask

        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


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

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        # Simple linear layer for simplicity
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, hidden_dim)
        linear_layer = nn.Linear(hidden_dim, hidden_dim).to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = linear_layer(hidden_states)
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_dim)
        return hidden_states


class ReasoningBlock(nn.Module):
    def __init__(self, layers: List[ReasoningAttnBlock]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, input_injection=input_injection)

        return hidden_states


class HRM_ACTV1_Inner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        batch_size: int,
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
        self.seq_len = seq_len
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.num_heads = num_heads
        self.forward_dtype = getattr(torch, forward_dtype)

        # I/O
        self.q_head = CastedLinear(self.hidden_size, 2, bias=True, cast_to=self.forward_dtype)

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

        # Observation encoding parameters
        self.num_layers = 25
        self.out_width = 11
        self.out_height = 11

        # Observation projection layer
        self.obs_projection = CastedLinear(
            self.num_layers * self.out_width * self.out_height, self.hidden_size, cast_to=self.forward_dtype
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

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
        # Input encoding
        input_embeddings = self._input_embeddings(batch["env_obs"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.H_cycles):
                for _L_step in range(self.L_cycles):
                    if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings)

                if not (_H_step == self.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.H_level(z_H, z_L)

        # New carry no grad and hidden state output
        new_carry = type("Carry", (), {"z_H": z_H.detach(), "z_L": z_L.detach()})()
        hidden_state = z_H.mean(dim=1)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
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
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = token_observations[..., 1].long()
        atr_values = token_observations[..., 2].float()

        # Create mask for valid tokens
        valid_tokens = coords_byte != 0xFF

        # Additional validation
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

        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index

        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scale
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        # Flatten spatial dimensions and project to hidden size
        box_obs_flat = box_obs.view(B_TT, -1)

        # Project to hidden size using the pre-initialized linear layer
        encoded = self.obs_projection(box_obs_flat)

        # Reshape back to [B, TT, hidden_size]
        if TT > 1:
            encoded = encoded.view(B, TT, self.hidden_size)

        return encoded


class HRMBackbone(nn.Module):
    """Hierarchical Reasoning Backbone with ACT wrapper."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        batch_size: int,
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
            seq_len=seq_len,
            forward_dtype=forward_dtype,
            H_layers=H_layers,
            L_layers=L_layers,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            num_heads=num_heads,
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Initialize carry (state) for a new batch."""
        batch_size = batch["env_obs"].shape[0]
        device = batch["env_obs"].device

        return {
            "inner_carry": self.inner.empty_carry(batch_size, device),
            "steps": torch.zeros((batch_size,), dtype=torch.int32, device=device),
            "halted": torch.ones((batch_size,), dtype=torch.bool, device=device),
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

                # Compute target Q
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


class HRM(Policy, HRMMemory):
    """Hierarchical Reasoning Model using Metta Policy framework."""

    def __init__(self, env, config=None):
        """Initialize HRM policy following Metta policy pattern."""
        super().__init__()
        HRMMemory.__init__(self)

        self.env = env
        self.config = config
        self.is_continuous = False

        # Store action space info
        self.action_space = getattr(env, "action_space", None)
        self.active_action_names = []
        self.num_active_actions = 100

        # Get environment dimensions
        self.out_width = getattr(env, "obs_width", 11)
        self.out_height = getattr(env, "obs_height", 11)
        self.num_layers = 25  # Default for mettagrid

        # Create the internal policy components
        self.policy = HRMPolicyInner(env)

        # Set up device tracking
        self._device = torch.device("cpu")

    def initialize_to_environment(self, env_metadata, device: torch.device):
        """Initialize policy to environment - required by training framework."""
        # Extract action names from environment metadata
        action_names = getattr(env_metadata, "action_names", ["noop", "move", "attack"])

        # Initialize action embeddings in the policy
        self.policy.initialize_to_environment(action_names, device)

        # Store environment metadata
        self.env_metadata = env_metadata
        self._device = device

        # Move model to device
        self.to(device)

    @property
    def device(self) -> torch.device:
        """Device property required by the Policy interface."""
        return getattr(self, "_device", torch.device("cpu"))

    def reset_memory(self):
        """Reset memory state - required by Policy interface."""
        HRMMemory.reset_memory(self)  # Use HRMMemory's reset_memory method

    def get_agent_experience_spec(self):
        """Get experience spec - required by Policy interface."""
        from torchrl.data import Composite, UnboundedDiscrete

        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @torch._dynamo.disable
    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass following Metta policy pattern."""
        observations = td["env_obs"]

        # Encode observations using HRM backbone
        hidden = self.policy.encode_observations(observations, td, memory_manager=self)

        # Decode actions and values
        logits, value = self.policy.decode_actions(hidden.to(torch.float32))

        # Set outputs in TensorDict
        td["logits"] = logits
        td["values"] = value.flatten()

        return td


class HRMPolicyInner(nn.Module):
    def __init__(self, env, input_size=64, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False

        # Handle both EnvironmentMetaData and legacy env objects
        if hasattr(env, "single_action_space"):
            self.action_space = env.single_action_space
        else:
            # Default action space for EnvironmentMetaData
            from gymnasium.spaces import MultiDiscrete

            self.action_space = MultiDiscrete([9, 10])  # Default arena actions

        self.out_width = getattr(env, "obs_width", 11)
        self.out_height = getattr(env, "obs_height", 11)

        # Default to 25 layers for mettagrid environments
        self.num_layers = 25

        # HRM Backbone
        self.backbone = HRMBackbone(
            hidden_size=hidden_size,
            vocab_size=10,
            batch_size=1,
            seq_len=20,
            forward_dtype="bfloat16",  # More memory efficient than float16
            H_layers=1,
            L_layers=1,
            H_cycles=1,
            L_cycles=1,
            num_heads=1,
            halt_max_steps=3,  # Reduced from 5
            halt_exploration_prob=0.1,
        )

        # Critic branch - reduced sizes
        self.critic_1 = nn.Linear(hidden_size, 256)  # Reduced from 1024
        self.value_head = nn.Linear(256, 1)

        # Actor branch - reduced sizes
        self.actor_1 = nn.Linear(hidden_size, 128)  # Reduced from 512

        # Action embeddings - reduced dimension
        self.action_embeddings = nn.Embedding(100, 8)  # Reduced from 16 to 8
        self._initialize_action_embeddings()

        # Store for dynamic action head
        self.action_embed_dim = 8  # Reduced from 16
        self.actor_hidden_dim = 128  # Reduced from 512

        # Bilinear layer to match MettaActorSingleHead
        self._init_bilinear_actor()

        # Track active actions
        self.active_action_names = []
        self.num_active_actions = 100

        self.effective_rank_enabled = True

    def _initialize_action_embeddings(self):
        """Initialize action embeddings to match YAML ActionEmbedding component."""
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)

    def _init_bilinear_actor(self):
        """Initialize bilinear actor head to match MettaActorSingleHead."""
        self.actor_W = nn.Parameter(
            torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).to(dtype=torch.float32)
        )
        self.actor_bias = nn.Parameter(torch.Tensor(1).to(dtype=torch.float32))

        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings, matching the YAML ActionEmbedding component behavior."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize to environment, setting up action embeddings to match the available actions."""
        self.activate_action_embeddings(full_action_names, device)

    def encode_observations(self, observations, td=None, memory_manager=None):
        """
        Encode observations using the HRM backbone.

        Args:
            observations: Input observation tensor, shape (B, TT, M, 3) or (B, M, 3)
            td: TensorDict containing environment metadata
            memory_manager: Object with get_memory/set_memory methods (usually the HRM instance)

        Returns:
            hidden: Encoded representation, shape (B * TT, hidden_size)
        """

        if len(observations.shape) == 4:
            observations = observations.reshape(-1, 200, 3)

        # Get environment ID for state tracking
        if td is not None:
            training_env_id_start = td.get("training_env_id_start", None)
            if training_env_id_start is not None:
                env_id = training_env_id_start[0].item()
            else:
                env_id = 0
        else:
            env_id = 0

        # Get previous carry state or initialize new one
        if memory_manager is not None:
            prev_memory = memory_manager.get_memory()
        else:
            prev_memory = {}

        if prev_memory is None or f"{env_id}" not in prev_memory:
            prev_carry = self.backbone.initial_carry({"env_obs": observations})
        else:
            prev_carry = prev_memory[f"{env_id}"]

        # Forward through backbone
        new_carry, outputs = self.backbone(prev_carry, {"env_obs": observations})

        # Update memory with new carry state
        if memory_manager is not None:
            current_memory = memory_manager.get_memory() or {}
            current_memory[f"{env_id}"] = new_carry
            memory_manager.set_memory(current_memory)

        # Return hidden state
        return outputs["hidden_state"]

    def decode_actions(self, hidden):
        """Decode actions using bilinear interaction to match MettaActorSingleHead."""
        # Critic branch
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)
        actor_features = F.relu(actor_features)

        # Use the actual batch size from the hidden tensor
        actual_batch_size = hidden.shape[0]

        # Get action embeddings for all actions
        action_embeds = self.action_embeddings.weight[: self.num_active_actions]

        # Expand action embeddings for each batch element
        action_embeds = action_embeds.unsqueeze(0).expand(actual_batch_size, -1, -1)

        # Bilinear interaction matching MettaActorSingleHead
        num_actions = action_embeds.shape[1]

        # Reshape for bilinear calculation
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)
        # Use the actual feature dimension instead of self.actor_hidden_dim
        actor_feature_dim = actor_features.shape[1]
        actor_reshaped = actor_repeated.reshape(-1, actor_feature_dim)
        # Use the actual embedding dimension
        action_embed_dim = action_embeds.shape[2]
        action_embeds_reshaped = action_embeds.reshape(-1, action_embed_dim)

        # Perform bilinear operation using einsum
        # Always ensure actor_W has the correct dimensions for this forward pass
        if (
            not hasattr(self, "actor_W")
            or self.actor_W.shape[1] != actor_feature_dim
            or self.actor_W.shape[2] != action_embed_dim
        ):
            # Initialize or reinitialize actor_W with correct dimensions
            self.actor_hidden_dim = actor_feature_dim
            self.action_embed_dim = action_embed_dim
            self._init_bilinear_actor()

        query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)
        query = torch.tanh(query)
        # Reshape query to match action_embeds_reshaped for the dot product
        query_reshaped = query.reshape(-1, action_embed_dim)
        scores = torch.einsum("n e, n e -> n", query_reshaped, action_embeds_reshaped)
        # Reshape scores back to [batch_size * num_actions, 1] then [batch_size, num_actions]
        scores = scores.reshape(actual_batch_size, num_actions)

        biased_scores = scores + self.actor_bias

        # Reshape back to [B*TT, num_actions]
        logits = biased_scores.reshape(actual_batch_size, num_actions)

        return logits, value


if __name__ == "__main__":
    import gymnasium as gym
    import numpy as np

    from metta.agent.metta_agent import MettaAgent
    from metta.rl.system_config import SystemConfig

    obs_shape = [34, 11, 11]
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
        obs_width=obs_shape[1],
        obs_height=obs_shape[2],
        single_action_space=gym.spaces.MultiDiscrete([9, 10]),
        feature_normalizations={},
        global_features=[],
    )

    # Test the HRM model
    hrm = HRM(env)

    # Test observations - for inference mode (3D: batch, tokens, features)
    obs = torch.randint(0, 8, (2, 200, 3), dtype=torch.uint8)  # batch=2, tokens=200
    td = TensorDict({"env_obs": obs}, batch_size=[obs.shape[0]])

    from metta.agent.agent_config import AgentConfig

    agent = MettaAgent(
        env,
        system_cfg=SystemConfig(device="cpu"),
        policy_architecture_cfg=AgentConfig(name="pytorch/hrm", clip_range=0.0),
        policy=HRM(env),
    )

    agent.initialize_to_environment(
        features={},
        action_names=["move", "attack", "heal"],
        action_max_params=[10, 10, 10],
        device=torch.device("cpu"),
        is_training=True,
    )

    output_td = agent(td)
