"""
Improved transformer agent building on the stable baseline.

This implementation makes the transformer BIGGER and more powerful:
1. Keeps all critical components from the working version (scatter encoding, bilinear decoding)
2. Doubles most dimensions (hidden_size, n_heads, d_ff) for more capacity
3. Increases number of layers for deeper processing
4. Uses the stable base TransformerModule without modifications
5. Larger action embeddings for richer action representations
"""

import logging
import math
import warnings
from typing import Optional

import einops
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.transformer_module import TransformerModule
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin
from pufferlib.pytorch import layer_init as init_layer

logger = logging.getLogger(__name__)


class ImprovedPolicy(nn.Module):
    """Improved policy network building on the stable transformer baseline."""

    def __init__(
        self,
        env,
        input_size: int = 256,  # BIGGER: Doubled from 128
        hidden_size: int = 256,  # BIGGER: Doubled from 128
        n_heads: int = 16,  # BIGGER: Doubled from 8
        n_layers: int = 8,  # BIGGER: Increased from 6
        d_ff: int = 1024,  # BIGGER: Doubled from 512
        max_seq_len: int = 512,  # BIGGER: Doubled from 256
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
        # Improvements
        use_layer_norm: bool = True,  # Add layer norm before output
        init_scale: float = 1.0,  # Keep standard initialization
    ):
        """Initialize the improved policy network.

        Args:
            env: Environment
            input_size: Input size for transformer
            hidden_size: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating
            use_layer_norm: Whether to add layer normalization
            positional_scale: Scale factor for positional encoding
            init_scale: Scale factor for weight initialization
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.use_layer_norm = use_layer_norm
        self.init_scale = init_scale

        # Observation parameters (matching Fast/Working Transformer)
        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11

        # Dynamically determine num_layers from environment
        if hasattr(env, "feature_normalizations"):
            self.num_layers = max(env.feature_normalizations.keys()) + 1
        else:
            self.num_layers = 25

        # CNN architecture matching Fast/Working Transformer exactly
        self.cnn1 = init_layer(
            nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3),
            std=1.0,  # Keep original initialization
        )
        self.cnn2 = init_layer(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), std=1.0)

        # Calculate flattened size
        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Linear layers matching the bigger input_size
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 256), std=1.0)  # BIGGER: Match increased size
        self.encoded_obs = init_layer(nn.Linear(256, input_size), std=self.init_scale)

        # IMPROVEMENT: Add optional layer norm after encoding
        if self.use_layer_norm:
            self.encoding_norm = nn.LayerNorm(input_size)

        # Use the base TransformerModule directly - it's already well-optimized
        logger.info(
            f"Creating TransformerModule with hidden_size={hidden_size}, n_heads={n_heads}, n_layers={n_layers}"
        )
        self._transformer = TransformerModule(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
        )
        logger.info("TransformerModule created successfully")

        # BIGGER critic branch with more capacity
        self.critic_1 = init_layer(nn.Linear(hidden_size, 2048), std=self.init_scale)  # BIGGER: Doubled
        self.value_head = init_layer(nn.Linear(2048, 1), std=0.1)  # Keep small for value head

        # BIGGER actor branch with more capacity
        self.actor_1 = init_layer(nn.Linear(hidden_size, 1024), std=0.5 * self.init_scale)  # BIGGER: Doubled

        # Action embeddings - critical for good performance!
        self.action_embeddings = nn.Embedding(100, 32)  # BIGGER: Doubled embedding dimension
        self._initialize_action_embeddings()

        # Store for dynamic action head
        self.action_embed_dim = 32  # BIGGER: Doubled embedding dimension
        self.actor_hidden_dim = 1024  # BIGGER: Doubled to match actor_1

        # Bilinear layer matching working transformer
        self._init_bilinear_actor()

        # Build normalization vector from environment
        if hasattr(env, "feature_normalizations"):
            max_values = [1.0] * self.num_layers
            for feature_id, norm_value in env.feature_normalizations.items():
                if feature_id < self.num_layers:
                    max_values[feature_id] = norm_value if norm_value > 0 else 1.0
            max_vec = torch.tensor(max_values, dtype=torch.float32)[None, :, None, None]
        else:
            max_vec = torch.ones(1, self.num_layers, 1, 1, dtype=torch.float32)
        self.register_buffer("max_vec", max_vec)

        # Track active actions
        self.active_action_names = []
        self.num_active_actions = 100

    def _initialize_action_embeddings(self):
        """Initialize action embeddings matching working transformer."""
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)

    def _init_bilinear_actor(self):
        """Initialize bilinear actor head matching working transformer."""
        self.actor_W = nn.Parameter(
            torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).to(dtype=torch.float32)
        )
        self.actor_bias = nn.Parameter(torch.Tensor(1).to(dtype=torch.float32))

        # Kaiming initialization
        bound = (1 / math.sqrt(self.actor_hidden_dim)) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def network_forward(self, x):
        """CNN forward pass matching working transformer."""
        x = x / self.max_vec
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.encoded_obs(x)
        x = F.relu(x)

        # IMPROVEMENT: Optional layer norm for stability
        if self.use_layer_norm:
            x = self.encoding_norm(x)

        return x

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Encode observations using scatter-based token placement.

        This is CRITICAL for performance - exact copy from working transformer.
        """
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        B_TT = B * TT

        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

        # Extract coordinates and attributes (matching Fast/Working transformer exactly)
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

        # Use scatter-based write (critical for performance!)
        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index

        # Mask out invalid entries
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scatter values into a flattened buffer, then reshape
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        # Process through CNN
        return self.network_forward(box_obs)

    def decode_actions(self, hidden: torch.Tensor, batch_size: int) -> tuple:
        """Decode actions using bilinear interaction.

        Exact copy from working transformer - this is critical!
        """
        # Critic branch
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)
        actor_features = F.relu(actor_features)

        # Get action embeddings for active actions
        action_embeds = self.action_embeddings.weight[: self.num_active_actions]

        # Expand action embeddings for each batch element
        action_embeds = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Bilinear interaction
        num_actions = action_embeds.shape[1]

        # Reshape for bilinear calculation
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)
        actor_reshaped = actor_repeated.reshape(-1, self.actor_hidden_dim)
        action_embeds_reshaped = action_embeds.reshape(-1, self.action_embed_dim)

        # Perform bilinear operation
        query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)
        query = torch.tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)

        biased_scores = scores + self.actor_bias

        # Reshape back
        logits = biased_scores.reshape(batch_size, num_actions)

        return logits, value

    def transformer(self, hidden: torch.Tensor, terminations: torch.Tensor = None, memory: dict = None):
        """Forward pass through transformer."""
        # Full-context transformer doesn't need memory
        output = self._transformer(hidden)
        return output, None

    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory for the transformer."""
        return {}


# Removed ImprovedTransformerModule class - we'll use the base TransformerModule directly
# The base module is already well-optimized and stable
# Making the model "beefier" by increasing dimensions is a better approach


class TransformerImproved(PyTorchAgentMixin, TransformerWrapper):
    """Improved transformer agent building on the stable baseline.

    This implementation carefully adds improvements without breaking training:
    - Keeps all critical components from working transformer
    - Adds layer normalization for stability
    - Uses slightly better weight initialization
    - Maintains compatibility with PyTorchAgentMixin
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        input_size: int = 256,  # BIGGER: Doubled from 128
        hidden_size: int = 256,  # BIGGER: Doubled from 128
        n_heads: int = 16,  # BIGGER: Doubled from 8
        n_layers: int = 8,  # BIGGER: Increased from 6
        d_ff: int = 1024,  # BIGGER: Doubled from 512
        max_seq_len: int = 512,  # BIGGER: Doubled from 256
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
        # Improvements
        use_layer_norm: bool = True,
        init_scale: float = 1.0,
        **kwargs,
    ):
        """Initialize the improved transformer agent."""
        # Extract mixin parameters
        mixin_params = self.extract_mixin_params(kwargs)

        logger.info("Initializing TransformerImproved agent...")

        if policy is None:
            logger.info("Creating ImprovedPolicy network...")
            policy = ImprovedPolicy(
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
                use_layer_norm=use_layer_norm,
                init_scale=init_scale,
            )
            logger.info("ImprovedPolicy network created successfully")

        # Initialize transformer wrapper
        logger.info("Initializing TransformerWrapper...")
        super().__init__(env, policy, hidden_size)
        logger.info("TransformerWrapper initialized successfully")

        # Initialize mixin
        logger.info("Initializing PyTorchAgentMixin...")
        self.init_mixin(**mixin_params)
        logger.info("TransformerImproved agent initialization complete")

    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass through the agent.

        Exact copy from working transformer to ensure compatibility.
        """
        observations = td["env_obs"]

        if state is None:
            state = {"transformer_memory": None, "hidden": None}

        # Determine dimensions
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Set TensorDict fields
        self.set_tensordict_fields(td, observations)

        # Encode observations with scatter-based token placement
        hidden = self.policy.encode_observations(observations, state)

        # Reshape for transformer
        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (T, B, hidden)
        else:
            hidden = hidden.unsqueeze(0)  # (1, B, hidden)

        # Forward through transformer
        hidden, _ = self.policy.transformer(hidden, None, state.get("transformer_memory"))

        # Reshape back
        if TT > 1:
            hidden = hidden.transpose(0, 1).reshape(B * TT, -1)
        else:
            hidden = hidden.squeeze(0)

        # Decode actions with bilinear interaction
        logits, values = self.policy.decode_actions(hidden, B * TT)

        # Process based on mode
        if action is None:
            td = self.forward_inference(td, logits, values)
        else:
            # Flatten values for training
            if values.dim() == 2:
                values = values.reshape(-1)
            td = self.forward_training(td, action, logits, values)

        return td
