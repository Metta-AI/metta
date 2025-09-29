import math
from types import SimpleNamespace
from typing import Dict, List, Tuple

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
        self.hidden_size = hidden_size
        self.cast_to = cast_to

        # Simplified reasoning block - just MLP with residual
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(), nn.Linear(hidden_size * 2, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        # Simple residual MLP block
        residual = hidden_states

        # Normalize input
        normalized = self.norm(hidden_states)

        # Apply MLP
        mlp_out = self.mlp(normalized)

        # Residual connection
        output = residual + mlp_out

        return output.to(self.cast_to)


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

        # Observation encoder will be created dynamically in encode_observations

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
        return {"z_H": z_H, "z_L": z_L}

    def reset_carry(self, reset_flag: torch.Tensor, carry):
        H_init = self.H_init.to(device=carry["z_H"].device, dtype=carry["z_H"].dtype)
        L_init = self.L_init.to(device=carry["z_L"].device, dtype=carry["z_L"].dtype)
        z_H = torch.where(reset_flag.view(-1, 1, 1), H_init, carry["z_H"])
        z_L = torch.where(reset_flag.view(-1, 1, 1), L_init, carry["z_L"])
        return {"z_H": z_H, "z_L": z_L}

    def forward(
        self, carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[object, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Input encoding
        input_embeddings = self._input_embeddings(batch["env_obs"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry["z_H"], carry["z_L"]

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
        new_carry = {"z_H": z_H.detach(), "z_L": z_L.detach()}
        hidden_state = z_H.mean(dim=1)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, hidden_state, (q_logits[..., 0], q_logits[..., 1])

    def encode_observations(self, observations):
        """
        Simplified observation encoding to avoid numerical issues.
        Input: observations with shape [batch_size, seq_len, 3]
        Output: encoded features with shape [batch_size, seq_len, hidden_size]
        """
        # Ensure observations are in the right shape and dtype
        if len(observations.shape) == 4:
            B, TT, M, C = observations.shape
            observations = observations.reshape(B * TT, M, C)
        else:
            B, M, C = observations.shape
            TT = 1

        # Simple approach: treat as sequence of tokens and encode directly
        # observations shape: [B*TT, seq_len, 3] where 3 = (x, y, attr)

        # Flatten the last dimension and use linear projection
        obs_flat = observations.float()  # Ensure float32

        # Simple linear encoding of each token
        batch_size, seq_len, features = obs_flat.shape
        obs_reshaped = obs_flat.view(batch_size, seq_len * features)

        # Project to hidden size
        if not hasattr(self, "_obs_encoder"):
            self._obs_encoder = nn.Linear(seq_len * features, self.hidden_size).to(
                device=observations.device, dtype=torch.float32
            )
            nn.init.xavier_uniform_(self._obs_encoder.weight)
            nn.init.zeros_(self._obs_encoder.bias)

        encoded = self._obs_encoder(obs_reshaped)

        # Add sequence dimension back for compatibility
        encoded = encoded.unsqueeze(1)  # [batch_size, 1, hidden_size]

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

        # Sample actions from logits
        import torch.distributions as dist

        action_dist = dist.Categorical(logits=logits)
        actions_flat = action_dist.sample()
        act_log_prob = action_dist.log_prob(actions_flat.float())  # log_prob needs float input
        entropy = action_dist.entropy()  # Calculate entropy for PPO loss

        # For MultiDiscrete environments, we need to reshape actions
        # Assuming 2 action components (typical for arena environments)
        batch_size = actions_flat.shape[0]
        actions = actions_flat.view(batch_size, 1).expand(batch_size, 2)

        # Set outputs in TensorDict
        td["logits"] = logits
        td["values"] = value.flatten()
        td["actions"] = actions.to(dtype=torch.int32)  # Match expected dtype in actor.py
        td["act_log_prob"] = act_log_prob
        td["entropy"] = entropy  # Required by PPO loss

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

        # Simplified HRM Core instead of complex backbone
        self.core = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
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
        self.actor_W = nn.Parameter(torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim))
        self.actor_bias = nn.Parameter(torch.Tensor(1))

        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0.1
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
        Encode observations using simplified HRM core.

        Args:
            observations: Input observation tensor, shape (B, TT, M, 3) or (B, M, 3)
            td: TensorDict containing environment metadata
            memory_manager: Object with get_memory/set_memory methods (usually the HRM instance)

        Returns:
            hidden: Encoded representation, shape (B * TT, hidden_size)
        """

        if len(observations.shape) == 4:
            observations = observations.reshape(-1, 200, 3)

        # Simple observation encoding
        obs_flat = observations.float()  # Ensure float32
        batch_size, seq_len, features = obs_flat.shape
        obs_reshaped = obs_flat.view(batch_size, seq_len * features)

        # Project to hidden size
        if not hasattr(self, '_obs_encoder'):
            self._obs_encoder = nn.Linear(seq_len * features, self.hidden_size).to(
                device=observations.device, dtype=torch.float32
            )
            nn.init.xavier_uniform_(self._obs_encoder.weight)
            nn.init.zeros_(self._obs_encoder.bias)

        encoded = self._obs_encoder(obs_reshaped)

        # Apply simplified HRM core
        hidden = self.core(encoded)

        return hidden

    def decode_actions(self, hidden):
        """Decode actions using simple linear projection."""
        # Ensure hidden is float32 to prevent numerical issues
        hidden = hidden.float()

        # Critic branch
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        # Actor branch - simplified approach
        actor_features = F.relu(self.actor_1(hidden))

        # Simple linear projection to action logits
        # Use a direct linear layer instead of complex bilinear operations
        if not hasattr(self, 'action_head'):
            self.action_head = nn.Linear(actor_features.shape[1], self.num_active_actions)
            nn.init.xavier_uniform_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
            # Move to same device as actor_features
            self.action_head = self.action_head.to(device=actor_features.device, dtype=actor_features.dtype)

        logits = self.action_head(actor_features)

        # Ensure no NaNs or infinities
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        return logits, value

