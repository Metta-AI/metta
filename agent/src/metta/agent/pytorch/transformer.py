"""Transformer agent for Metta."""

import logging
import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from pufferlib.pytorch import layer_init as init_layer
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.transformer_module import TransformerModule
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    def __init__(
        self,
        env,
        input_size: int = 128,
        hidden_size: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11
        self.num_layers = max(env.feature_normalizations.keys()) + 1 if hasattr(env, "feature_normalizations") else 25

        # Enhanced CNN backbone - borrow wider cnn2 from TransformerImproved
        self.cnn1 = init_layer(nn.Conv2d(self.num_layers, 64, 5, 3), std=1.0)
        self.cnn2 = init_layer(nn.Conv2d(64, 128, 3, 1), std=1.0)  # Improved: 128 channels

        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(torch.zeros(1, self.num_layers, self.out_width, self.out_height)))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()
        # Enhanced feature processing - borrow larger fc1 from TransformerImproved
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 256), std=1.0)  # Improved: 256 dims
        self.encoded_obs = init_layer(nn.Linear(256, input_size), std=1.0)

        # Enhanced transformer with minimal memory - reduce memory_len for RL efficiency
        self._transformer = TransformerModule(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            memory_len=16,  # Add minimal memory (vs 64 in TransformerImproved)
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
        )

        # Hybrid approach: Keep bilinear actor but add simplified critic option
        self.use_simple_heads = True  # Flag to choose between simple vs bilinear heads

        if self.use_simple_heads:
            # Simplified heads inspired by TransformerImproved
            self.critic = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 100),  # Max action space size
            )
            # Initialize simplified heads with proper scaling from TransformerImproved
            for head in [self.critic, self.actor]:
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        init_layer(layer, std=1.0 if layer != head[-1] else 0.1)
        else:
            # Original bilinear approach (fallback)
            self.critic_1 = init_layer(nn.Linear(hidden_size, 1024), std=1.0)
            self.value_head = init_layer(nn.Linear(1024, 1), std=0.1)
            self.actor_1 = init_layer(nn.Linear(hidden_size, 512), std=0.5)
            self.action_embeddings = nn.Embedding(100, 16)
            self._initialize_action_embeddings()
            self.action_embed_dim = 16
            self.actor_hidden_dim = 512
            self._init_bilinear_actor()

        max_values = [1.0] * self.num_layers
        if hasattr(env, "feature_normalizations"):
            for fid, norm in env.feature_normalizations.items():
                if fid < self.num_layers:
                    max_values[fid] = norm if norm > 0 else 1.0
        self.register_buffer("max_vec", torch.tensor(max_values, dtype=torch.float32)[None, :, None, None])
        self.active_action_names = []
        self.num_active_actions = 100

    def _initialize_action_embeddings(self):
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            self.action_embeddings.weight.mul_(0.1 / torch.max(torch.abs(self.action_embeddings.weight)))

    def _init_bilinear_actor(self):
        self.actor_W = nn.Parameter(torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).float())
        self.actor_bias = nn.Parameter(torch.Tensor(1).float())
        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)

    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize to environment, setting up action embeddings to match the available actions."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def network_forward(self, x):
        x = x / self.max_vec
        x = F.relu(self.cnn2(F.relu(self.cnn1(x))))
        x = F.relu(self.encoded_obs(F.relu(self.fc1(self.flatten(x)))))
        return x

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Clean observation encoding with simplified tensor handling."""
        # Handle batching: flatten to (batch_size, num_tokens, 3) if needed
        if observations.dim() == 4:  # (B, T, M, 3) -> (B*T, M, 3)
            batch_size = observations.shape[0] * observations.shape[1]
            observations = observations.view(batch_size, *observations.shape[2:])
        else:  # (B, M, 3)
            batch_size = observations.shape[0]

        assert observations.shape[-1] == 3, f"Expected 3 channels per token, got {observations.shape}"

        # Extract coordinate and attribute information
        coords_byte = observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        # Create validity masks
        valid_tokens = coords_byte != 0xFF
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr

        # Warn about invalid indices but continue
        if (valid_tokens & ~valid_atr).any():
            warnings.warn(f"Found obs attribute indices >= {self.num_layers}, ignoring", stacklevel=2)

        # Compute flattened indices for scatter operation
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + x_coord_indices * self.out_height + y_coord_indices
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scatter to grid representation
        box_flat = torch.zeros(
            (batch_size, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)

        # Reshape and pass through CNN
        box_obs = box_flat.view(batch_size, self.num_layers, self.out_width, self.out_height)
        return self.network_forward(box_obs)

    def decode_actions(self, hidden: torch.Tensor, batch_size: int) -> tuple:
        """Enhanced decode_actions with hybrid approach."""
        if self.use_simple_heads:
            # Simplified approach from TransformerImproved
            values = self.critic(hidden).squeeze(-1)
            full_logits = self.actor(hidden)  # (B, max_action_space)
            logits = full_logits[:, : self.num_active_actions]  # Slice to active actions
            return logits, values
        else:
            # Original bilinear approach
            value = self.value_head(torch.tanh(self.critic_1(hidden)))
            actor_features = F.relu(self.actor_1(hidden))
            action_embeds = (
                self.action_embeddings.weight[: self.num_active_actions].unsqueeze(0).expand(batch_size, -1, -1)
            )
            num_actions = action_embeds.shape[1]
            actor_reshaped = actor_features.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, self.actor_hidden_dim)
            action_embeds_reshaped = action_embeds.reshape(-1, self.action_embed_dim)
            query = torch.tanh(torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W))
            logits = (torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped) + self.actor_bias).reshape(
                batch_size, num_actions
            )
            return logits, value

    def transformer(self, hidden: torch.Tensor, terminations: torch.Tensor = None, memory: dict = None):
        """Enhanced transformer with proper memory handling."""
        output, new_memory = self._transformer(hidden, memory)
        return output, new_memory

    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize transformer memory."""
        return self._transformer.initialize_memory(batch_size)


class Transformer(PyTorchAgentMixin, TransformerWrapper):
    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        input_size: int = 128,
        hidden_size: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
        **kwargs,
    ):
        mixin_params = self.extract_mixin_params(kwargs)
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_causal_mask=use_causal_mask,
                use_gating=use_gating,
            )
        super().__init__(env, policy, hidden_size)
        self.init_mixin(**mixin_params)

    def forward(self, td: TensorDict, state=None, action=None):
        """Cleaner forward pass with simplified tensor handling."""
        observations = td["env_obs"]

        # Initialize state if needed
        if state is None:
            state = {"transformer_memory": None, "hidden": None}

        # Handle different input shapes more cleanly
        is_sequential = observations.dim() == 4  # (B, T, M, 3)
        if is_sequential:
            batch_size, seq_len = observations.shape[:2]
            flat_batch_size = batch_size * seq_len
            # Reshape TensorDict for processing
            if td.batch_dims > 1:
                td = td.reshape(flat_batch_size)
        else:
            batch_size, seq_len = observations.shape[0], 1
            flat_batch_size = batch_size

        self.set_tensordict_fields(td, observations)

        # Encode observations - this handles batching internally
        hidden = self.policy.encode_observations(observations, state)  # -> (flat_batch_size, hidden_size)

        # Reshape for transformer: (seq_len, batch_size, hidden_size)
        if is_sequential:
            hidden = hidden.view(batch_size, seq_len, -1).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)  # Add sequence dimension

        # Pass through transformer with memory
        hidden, new_memory = self.policy.transformer(hidden, None, state.get("transformer_memory"))

        # Update memory state
        if new_memory is not None:
            state["transformer_memory"] = new_memory

        # Reshape back to flat for action decoding: (flat_batch_size, hidden_size)
        if is_sequential:
            hidden = hidden.transpose(0, 1).contiguous().view(flat_batch_size, -1)
        else:
            hidden = hidden.squeeze(0)

        # Decode actions and values
        logits, values = self.policy.decode_actions(hidden, flat_batch_size)

        # Ensure proper value shape
        if values.dim() > 1:
            values = values.squeeze(-1)

        # Forward through mixin
        if action is None:
            td = self.forward_inference(td, logits, values)
        else:
            td = self.forward_training(td, action, logits, values)

        return td
