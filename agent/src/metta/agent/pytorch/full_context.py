"""
Full-context transformer agent for Metta with proper observation encoding and action decoding.

This version correctly implements:
1. Scatter-based token observation encoding (matching Fast/AGaLiTe)
2. CNN-based spatial feature extraction
3. Bilinear action decoding with action embeddings
4. Proper normalization and initialization
"""

import logging
import math
import warnings
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
from pufferlib.pytorch import layer_init as init_layer
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.full_context_transformer import FullContextTransformer
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    """Policy network using full-context transformer with proper Metta integration."""
    
    def __init__(
        self,
        env,
        input_size: int = 128,  # Match Fast's default
        hidden_size: int = 128,  # Match Fast's default
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,  # Smaller default for efficiency
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        """Initialize the policy network with proper CNN and action embeddings.
        
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
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        
        # Observation parameters (matching Fast)
        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11
        
        # Dynamically determine num_layers from environment
        if hasattr(env, "feature_normalizations"):
            self.num_layers = max(env.feature_normalizations.keys()) + 1
        else:
            self.num_layers = 25  # Default value
        
        # CNN architecture matching Fast exactly
        self.cnn1 = init_layer(
            nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3),
            std=1.0  # Match Fast's initialization
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
        
        # Linear layers to match Fast
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 128), std=1.0)
        self.encoded_obs = init_layer(nn.Linear(128, input_size), std=1.0)
        
        # Full-context transformer (replacing LSTM)
        logger.info(f"Creating FullContextTransformer with hidden_size={hidden_size}, "
                   f"n_heads={n_heads}, n_layers={n_layers}")
        self._transformer = FullContextTransformer(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
        )
        logger.info("FullContextTransformer created successfully")
        
        # Critic branch (matching Fast)
        self.critic_1 = init_layer(nn.Linear(hidden_size, 1024), std=np.sqrt(2))
        self.value_head = init_layer(nn.Linear(1024, 1), std=1.0)
        
        # Actor branch (matching Fast)
        self.actor_1 = init_layer(nn.Linear(hidden_size, 512), std=1.0)
        
        # Action embeddings - critical for good performance!
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()
        
        # Store for dynamic action head
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512
        
        # Bilinear layer to match MettaActorSingleHead
        self._init_bilinear_actor()
        
        # Build normalization vector dynamically from environment
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
        self.num_active_actions = 100  # Default
        
    def _initialize_action_embeddings(self):
        """Initialize action embeddings to match Fast."""
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
        
        # Kaiming (He) initialization
        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)
    
    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)
    
    def network_forward(self, x):
        """CNN forward pass matching Fast exactly."""
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
        """Encode observations using scatter-based token placement (matching Fast).
        
        This is CRITICAL for performance - the environment provides token observations
        that need to be scattered into a spatial grid before CNN processing.
        """
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        B_TT = B * TT
        
        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")
        
        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"
        
        # Extract coordinates and attributes (matching Fast exactly)
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
        """Decode actions using bilinear interaction (matching Fast).
        
        This bilinear interaction with action embeddings is much more powerful
        than simple linear heads and is critical for good performance.
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
        """Forward pass through transformer.
        
        This method is expected by TransformerWrapper.
        """
        # Full-context transformer doesn't need memory
        output = self._transformer(hidden)
        return output, None
    
    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory for the transformer."""
        return {}


class FullContext(PyTorchAgentMixin, TransformerWrapper):
    """Full-context transformer agent with proper observation encoding and action decoding.
    
    This implementation includes:
    - Scatter-based token observation encoding (critical for performance!)
    - CNN spatial feature extraction matching Fast agent
    - Bilinear action decoding with embeddings
    - GTrXL-style stabilization with GRU gating
    - All optimizations from Fast/AGaLiTe agents
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
        use_causal_mask: bool = True,
        use_gating: bool = True,
        **kwargs,
    ):
        """Initialize the fixed full-context transformer agent."""
        # Extract mixin parameters
        mixin_params = self.extract_mixin_params(kwargs)
        
        logger.info("Initializing FullContext transformer agent...")
        
        if policy is None:
            logger.info("Creating Policy network...")
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
            logger.info("Policy network created successfully")
        
        # Initialize transformer wrapper
        logger.info("Initializing TransformerWrapper...")
        super().__init__(env, policy, hidden_size)
        logger.info("TransformerWrapper initialized successfully")
        
        # Initialize mixin
        logger.info("Initializing PyTorchAgentMixin...")
        self.init_mixin(**mixin_params)
        logger.info("FullContext agent initialization complete")
    
    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass through the agent."""
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