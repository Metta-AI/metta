import logging
import math
import warnings
from typing import Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.pytorch.base import LSTMWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin
from pufferlib.pytorch import layer_init as init_layer

from metta.agent.modules.agalite_batched import BatchedAGaLiTe

logger = logging.getLogger(__name__)


class AgaliteHybrid(PyTorchAgentMixin, LSTMWrapper):
    """Hybrid AGaLiTe-LSTM architecture for efficient RL training.

    This uses AGaLiTe's sophisticated attention-based observation encoding
    combined with LSTM for temporal processing. This hybrid approach provides:
    - AGaLiTe's powerful observation processing with linear attention
    - LSTM's efficient O(1) temporal state updates
    - Full compatibility with Metta's training infrastructure via mixin
    - Automatic gradient detachment and proper state management

    Note: This is not the pure AGaLiTe transformer from the paper, but a
    practical hybrid that achieves better training efficiency."""

    def __init__(self, env, policy=None, input_size=256, hidden_size=256, num_layers=2, **kwargs):
        """Initialize with mixin support for configuration parameters.
        
        Args:
            env: Environment
            policy: Optional inner policy 
            input_size: LSTM input size
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            **kwargs: Configuration parameters (clip_range, analyze_weights_interval, etc.)
        """
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)
        
        if policy is None:
            policy = AgalitePolicy(env, input_size=input_size, hidden_size=hidden_size)

        # Use enhanced LSTMWrapper with proper state management
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass with enhanced state management."""
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Determine dimensions from observations
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
            # Reshape TD for training if needed
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Set critical TensorDict fields using mixin
        self.set_tensordict_fields(td, observations)

        # Encode observations through policy
        hidden = self.policy.encode_observations(observations, state)

        # Use enhanced LSTMWrapper's state management (includes automatic detachment!)
        lstm_h, lstm_c, env_id = self._manage_lstm_state(td, B, TT, observations.device)
        lstm_state = (lstm_h, lstm_c)

        # LSTM forward pass
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, hidden)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)

        # Store state with automatic detachment to prevent gradient accumulation
        self._store_lstm_state(new_lstm_h, new_lstm_c, env_id)

        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and value
        logits, value = self.policy.decode_actions(flat_hidden)

        # Use mixin for mode-specific processing
        if action is None:
            # Mixin handles inference mode properly
            td = self.forward_inference(td, logits, value)
        else:
            # Mixin handles training mode with proper reshaping
            td = self.forward_training(td, action, logits, value)

        return td


class AgalitePolicy(nn.Module):
    """Inner policy using AGaLiTe architecture."""

    def __init__(self, env, input_size=256, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = False  # Required by PufferLib
        self.action_space = env.single_action_space
        # Observation parameters
        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Create observation encoders
        self.cnn_encoder = nn.Sequential(
            init_layer(nn.Conv2d(self.num_layers, 128, 5, stride=3)),
            nn.ReLU(),
            init_layer(nn.Conv2d(128, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(128, input_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            init_layer(nn.Linear(self.num_layers, input_size // 2)),
            nn.ReLU(),
        )

        # AGaLiTe core (without LSTM since PufferLib wrapper handles that)
        self.agalite_core = BatchedAGaLiTe(
            n_layers=2,  # Reduced for efficiency
            d_model=input_size,
            d_head=64,
            d_ffc=hidden_size * 2,
            n_heads=4,
            eta=4,
            r=4,
            reset_on_terminate=True,
            dropout=0.0,  # No dropout for inference speed
        )

        # Initialize AGaLiTe memory
        self.agalite_memory = None

        # Critic branch - Two layers matching Fast
        # critic_1 uses gain=sqrt(2) because it's followed by tanh
        self.critic_1 = init_layer(nn.Linear(input_size, 1024), std=np.sqrt(2))
        # value_head has no nonlinearity, so gain=1
        self.value_head = init_layer(nn.Linear(1024, 1), std=1.0)
        
        # Actor branch
        # actor_1 uses gain=1 (for ReLU activation)
        self.actor_1 = init_layer(nn.Linear(input_size, 512), std=1.0)
        
        # Action embeddings - will be properly initialized
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()
        
        # Store for dynamic action head
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512
        
        # Bilinear layer to match MettaActorSingleHead
        self._init_bilinear_actor()

        # Build normalization vector dynamically from environment
        if hasattr(env, "feature_normalizations"):
            # Create max_vec from feature_normalizations
            max_values = [1.0] * self.num_layers  # Default to 1.0
            for feature_id, norm_value in env.feature_normalizations.items():
                if feature_id < self.num_layers:
                    max_values[feature_id] = norm_value if norm_value > 0 else 1.0
            max_vec = torch.tensor(max_values, dtype=torch.float32)[None, :, None, None]
        else:
            # Fallback normalization vector
            max_vec = torch.ones(1, self.num_layers, 1, 1, dtype=torch.float32)
        self.register_buffer("max_vec", max_vec)
        
        # Track active actions
        self.active_action_names = []
        self.num_active_actions = 100  # Default

    def _initialize_action_embeddings(self):
        """Initialize action embeddings to match YAML ActionEmbedding component."""
        # Match the YAML component's initialization (orthogonal then scaled to max 0.1)
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)

    def _init_bilinear_actor(self):
        """Initialize bilinear actor head to match MettaActorSingleHead."""
        # Bilinear parameters matching MettaActorSingleHead
        self.actor_W = nn.Parameter(
            torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).to(dtype=torch.float32)
        )
        self.actor_bias = nn.Parameter(torch.Tensor(1).to(dtype=torch.float32))

        # Kaiming (He) initialization
        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings, matching the YAML ActionEmbedding component behavior."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Encode observations into features."""
        B = observations.shape[0]

        # Process token observations
        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
            B = observations.shape[0]

        # Don't modify original tensor - ComponentPolicy doesn't do this (PR #2126)
        # observations[observations == 255] = 0  # REMOVED
        
        # Extract coordinates and attributes (matching ObsTokenToBoxShaper exactly)
        coords_byte = observations[..., 0].to(torch.uint8)
        x_coords = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B, M]
        y_coords = (coords_byte & 0x0F).long()  # Shape: [B, M]
        atr_indices = observations[..., 1].long()  # Shape: [B, M]
        atr_values = observations[..., 2].float()  # Shape: [B, M]
        
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
        flat_spatial_index = x_coords * self.out_height + y_coords  # [B, M]
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index  # [B, M]
        
        # Mask out invalid entries by directing them to index 0 with value 0
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))
        
        # Scatter values into a flattened buffer, then reshape to [B, L, W, H]
        box_flat = torch.zeros(
            (B, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B, self.num_layers, self.out_width, self.out_height)

        # Normalize and encode
        features = box_obs / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.cnn_encoder(features)

        encoded = torch.cat([self_features, cnn_features], dim=1)

        # Pass through AGaLiTe core (without time dimension - LSTM wrapper handles that)
        # Initialize memory if needed
        if self.agalite_memory is None:
            self.agalite_memory = BatchedAGaLiTe.initialize_memory(
                batch_size=B,
                n_layers=2,
                n_heads=4,
                d_head=64,
                eta=4,
                r=4,
                device=encoded.device,
            )

        # Check if batch size changed
        memory_batch_size = next(iter(next(iter(self.agalite_memory.values())))).shape[0]
        if memory_batch_size != B:
            # Reinitialize memory for new batch size
            self.agalite_memory = BatchedAGaLiTe.initialize_memory(
                batch_size=B,
                n_layers=2,
                n_heads=4,
                d_head=64,
                eta=4,
                r=4,
                device=encoded.device,
            )

        # Add time dimension for AGaLiTe
        encoded = encoded.unsqueeze(0)  # (1, B, features)
        terminations = torch.zeros(1, B, device=encoded.device)

        # Forward through AGaLiTe
        agalite_out, new_memory = self.agalite_core(encoded, terminations, self.agalite_memory)

        # Detach memory to prevent gradient accumulation
        self.agalite_memory = {
            key: tuple(tensor.detach() for tensor in layer_memory) for key, layer_memory in new_memory.items()
        }

        # Remove time dimension
        return agalite_out.squeeze(0)

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode actions using two-layer critic and bilinear actor."""
        # Critic branch - two layers matching Fast
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)
        
        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)
        actor_features = F.relu(actor_features)  # ReLU after actor_1
        
        # Get action embeddings for all actions
        action_embeds = self.action_embeddings.weight[:self.num_active_actions]  # [num_actions, embed_dim]
        
        # Bilinear interaction for action selection
        batch_size = actor_features.shape[0]
        num_actions = action_embeds.shape[0]
        
        # Expand for batched computation
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)
        actor_reshaped = actor_repeated.reshape(-1, self.actor_hidden_dim)
        action_embeds_expanded = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        action_embeds_reshaped = action_embeds_expanded.reshape(-1, self.action_embed_dim)
        
        # Bilinear operation
        query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)
        query = torch.tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)
        biased_scores = scores + self.actor_bias
        
        # Reshape to [batch_size, num_actions]
        logits = biased_scores.reshape(batch_size, num_actions)
        
        return logits, value

    def reset_memory(self):
        """Reset AGaLiTe memory."""
        self.agalite_memory = None
