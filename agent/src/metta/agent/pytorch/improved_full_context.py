"""
Improved full-context transformer agent with proper Metta integration.

This implementation ensures:
1. Full PyTorchAgentMixin integration for action conversion and TD management
2. Scatter-based observation encoding (CRITICAL for performance)
3. Proper weight initialization for transformers
4. Better positional encoding handling
5. Bilinear action decoding with embeddings
"""

import logging
import math
import warnings
from typing import Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pufferlib.pytorch import layer_init as init_layer
from tensordict import TensorDict

from metta.agent.modules.improved_transformer import ImprovedFullContextTransformer
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class ImprovedPolicy(nn.Module):
    """Improved policy network with better transformer integration."""
    
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
        use_causal_mask: bool = False,  # Usually not needed for RL
        use_gating: bool = True,
        learnable_pos: bool = True,  # Use learnable positional embeddings
        pos_scale: float = 0.1,
    ):
        """Initialize improved policy with proper Metta integration."""
        super().__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        
        # CRITICAL: Observation parameters for scatter-based encoding
        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11
        
        # Dynamically determine num_layers from environment
        if hasattr(env, "feature_normalizations"):
            self.num_layers = max(env.feature_normalizations.keys()) + 1
        else:
            self.num_layers = 25
        
        # CRITICAL: CNN architecture for spatial feature extraction
        # Use conservative initialization for CNN layers
        self.cnn1 = init_layer(
            nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3),
            std=1.0  # Standard initialization
        )
        self.cnn2 = init_layer(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            std=1.0
        )
        
        # Calculate flattened size
        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]
        
        self.flatten = nn.Flatten()
        
        # Linear layers with proper initialization
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 128), std=1.0)
        self.encoded_obs = init_layer(nn.Linear(128, input_size), std=1.0)
        
        # IMPROVED: Full-context transformer with better initialization
        logger.info(
            f"Creating ImprovedFullContextTransformer with hidden_size={hidden_size}, "
            f"n_heads={n_heads}, n_layers={n_layers}"
        )
        self._transformer = ImprovedFullContextTransformer(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
            learnable_pos=learnable_pos,
            pos_scale=pos_scale,
            activation="gelu"  # Use GELU for better performance
        )
        logger.info("ImprovedFullContextTransformer created successfully")
        
        # CRITICAL: Critic branch with proper initialization
        self.critic_1 = init_layer(nn.Linear(hidden_size, 1024), std=np.sqrt(2))
        self.value_head = init_layer(nn.Linear(1024, 1), std=1.0)
        
        # CRITICAL: Actor branch
        self.actor_1 = init_layer(nn.Linear(hidden_size, 512), std=1.0)
        
        # CRITICAL: Action embeddings for bilinear decoding
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()
        
        # Store for dynamic action head
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512
        
        # CRITICAL: Bilinear layer for action decoding
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
        
        # IMPORTANT: Store for PyTorchAgentMixin compatibility
        # These will be set by MettaAgent.activate_actions()
        self.action_index_tensor = None
        self.cum_action_max_params = None
    
    def _initialize_action_embeddings(self):
        """Initialize action embeddings with small values."""
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)
    
    def _init_bilinear_actor(self):
        """Initialize bilinear actor head for action decoding."""
        self.actor_W = nn.Parameter(
            torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).to(dtype=torch.float32)
        )
        self.actor_bias = nn.Parameter(torch.Tensor(1).to(dtype=torch.float32))
        
        # Conservative initialization for bilinear layer
        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)
    
    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings - called by MettaAgent."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)
        logger.debug(f"Activated {self.num_active_actions} actions")
    
    def network_forward(self, x):
        """CNN forward pass for spatial feature extraction."""
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
        return x
    
    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """
        CRITICAL: Scatter-based token observation encoding.
        This is THE most important part for performance!
        """
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        B_TT = B * TT
        
        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")
        
        assert token_observations.shape[-1] == 3, f"Expected 3 channels, got {token_observations.shape}"
        
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
                f"Found observation indices {sorted(invalid_indices.tolist())} "
                f">= num_layers ({self.num_layers}). These will be ignored.",
                stacklevel=2,
            )
        
        # CRITICAL: Scatter-based write for performance
        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index
        
        # Mask out invalid entries
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))
        
        # Scatter values into flattened buffer, then reshape
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), 
            dtype=atr_values.dtype, 
            device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)
        
        # Process through CNN
        return self.network_forward(box_obs)
    
    def decode_actions(self, hidden: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CRITICAL: Bilinear action decoding with embeddings.
        This interaction pattern is much more powerful than simple linear heads.
        """
        # Critic branch
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)
        
        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)
        actor_features = F.relu(actor_features)
        
        # Get action embeddings for active actions
        action_embeds = self.action_embeddings.weight[:self.num_active_actions]
        
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
        """Forward pass through transformer - expected by TransformerWrapper."""
        # Full-context transformer doesn't need memory
        output = self._transformer(hidden)
        return output, None
    
    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory for the transformer."""
        return {}


class ImprovedFullContext(PyTorchAgentMixin, TransformerWrapper):
    """
    Improved full-context transformer agent with proper Metta integration.
    
    Key improvements:
    1. Better weight initialization for transformers
    2. Learnable positional embeddings
    3. Full PyTorchAgentMixin integration
    4. Proper scatter-based observation encoding
    5. Bilinear action decoding with embeddings
    """
    
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
        use_causal_mask: bool = False,
        use_gating: bool = True,
        learnable_pos: bool = True,
        pos_scale: float = 0.1,
        **kwargs,
    ):
        """Initialize improved full-context transformer agent."""
        # Extract mixin parameters FIRST
        mixin_params = self.extract_mixin_params(kwargs)
        
        logger.info("Initializing ImprovedFullContext transformer agent...")
        
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
                learnable_pos=learnable_pos,
                pos_scale=pos_scale,
            )
            logger.info("ImprovedPolicy network created successfully")
        
        # Initialize transformer wrapper
        logger.info("Initializing TransformerWrapper...")
        super().__init__(env, policy, hidden_size)
        logger.info("TransformerWrapper initialized successfully")
        
        # Initialize mixin - this provides critical functionality
        logger.info("Initializing PyTorchAgentMixin...")
        self.init_mixin(**mixin_params)
        logger.info("ImprovedFullContext agent initialization complete")
        
        # Log parameter count
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {param_count:,}")
    
    def forward(self, td: TensorDict, state=None, action=None):
        """
        Forward pass through the agent with proper PyTorchAgentMixin integration.
        """
        observations = td["env_obs"]
        
        if state is None:
            state = {"transformer_memory": None, "hidden": None}
        
        # Determine dimensions and reshape TD if needed
        if observations.dim() == 4:  # Training: [B, T, obs_tokens, 3]
            B = observations.shape[0]
            TT = observations.shape[1]
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference: [B, obs_tokens, 3]
            B = observations.shape[0]
            TT = 1
        
        # CRITICAL: Set TensorDict fields for BPTT management
        # This is essential for proper integration with ComponentPolicy
        self.set_tensordict_fields(td, observations)
        
        # CRITICAL: Encode observations with scatter-based token placement
        hidden = self.policy.encode_observations(observations, state)
        
        # Reshape for transformer: (T, B, hidden)
        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)
        
        # Forward through transformer
        hidden, _ = self.policy.transformer(hidden, None, state.get("transformer_memory"))
        
        # Reshape back
        if TT > 1:
            hidden = hidden.transpose(0, 1).reshape(B * TT, -1)
        else:
            hidden = hidden.squeeze(0)
        
        # CRITICAL: Decode actions with bilinear interaction
        logits, values = self.policy.decode_actions(hidden, B * TT)
        
        # CRITICAL: Use PyTorchAgentMixin methods for proper mode handling
        if action is None:
            # Inference mode - use mixin's forward_inference
            td = self.forward_inference(td, logits, values)
        else:
            # Training mode - use mixin's forward_training
            if values.dim() == 2:
                values = values.reshape(-1)
            td = self.forward_training(td, action, logits, values)
        
        return td
    
    def clip_weights(self):
        """Override to use PyTorchAgentMixin's weight clipping."""
        # This will be called by the trainer when clip_range > 0
        super().clip_weights()
    
    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss for regularization."""
        return super().l2_init_loss()
    
    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings - passes through to policy."""
        self.policy.activate_action_embeddings(full_action_names, device)