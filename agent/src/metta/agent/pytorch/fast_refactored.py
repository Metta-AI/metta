"""
Example of Fast agent refactored to use PyTorchAgentMixin.

This shows how to simplify PyTorch agents by using the shared mixin
for common functionality while keeping agent-specific logic separate.
"""

import logging
import math
import warnings

import einops
import numpy as np
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

# Import base classes and mixin
from metta.agent.pytorch.base import LSTMWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class FastRefactored(PyTorchAgentMixin, LSTMWrapper):
    """Fast CNN-based policy using PyTorchAgentMixin for shared functionality."""

    def __init__(self, env, policy=None, cnn_channels=128, input_size=128, hidden_size=128, num_layers=2, **kwargs):
        """
        Initialize Fast policy with mixin support.

        The mixin handles:
        - Configuration parameters (clip_range, analyze_weights_interval)
        - Weight clipping
        - TensorDict field management
        - Action conversion
        - Training/inference mode handling
        """
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)

        # Create policy if not provided
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )

        # Initialize LSTMWrapper
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

        logger.info(f"[DEBUG] FastRefactored initialized with {sum(p.numel() for p in self.parameters())} parameters")
        logger.info(f"[DEBUG] LSTM: {self.lstm.num_layers} layers, hidden_size={self.lstm.hidden_size}")
        logger.info(f"[DEBUG] clip_range={self.clip_range}, analyze_weights_interval={self.analyze_weights_interval}")

    # clip_weights() is provided by the mixin

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None):
        """
        Forward pass simplified using mixin utilities.

        The mixin handles:
        - Setting td["bptt"] and td["batch"] fields
        - Training vs inference mode processing
        - Action conversion
        """
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Use mixin to set critical TensorDict fields
        B, TT = self.set_tensordict_fields(td, observations)

        # Encode observations (agent-specific)
        hidden = self.policy.encode_observations(observations, state)

        # Use base class for LSTM state management
        lstm_h, lstm_c, env_id = self._manage_lstm_state(td, B, TT, observations.device)
        lstm_state = (lstm_h, lstm_c)

        # LSTM forward pass
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)

        # Store state with automatic detachment
        self._store_lstm_state(new_lstm_h, new_lstm_c, env_id)

        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and value (agent-specific)
        logits_list, value = self.policy.decode_actions(flat_hidden, B * TT)

        # Use mixin for mode-specific processing
        if action is None:
            # Mixin handles inference mode
            td = self.handle_inference_mode(td, logits_list, value)
        else:
            # Mixin handles training mode with proper reshaping
            td = self.handle_training_mode(td, action, logits_list, value)

        return td

    # _convert_logit_index_to_action and _convert_action_to_logit_index
    # are provided by the mixin

    # activate_action_embeddings is provided by the mixin


class Policy(nn.Module):
    """Inner policy network - unchanged from original Fast implementation."""

    def __init__(self, env, input_size=128, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space

        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11

        # Dynamically determine num_layers from environment features
        if hasattr(env, "feature_normalizations"):
            self.num_layers = max(env.feature_normalizations.keys()) + 1
        else:
            self.num_layers = 25  # Default value

        # CNN layers with proper initialization
        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3),
            std=1.0,  # Match YAML orthogonal gain=1
        )
        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            std=1.0,  # Match YAML orthogonal gain=1
        )

        # Calculate flattened size
        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Linear layers
        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(self.flattened_size, 128), std=1.0)
        self.encoded_obs = pufferlib.pytorch.layer_init(nn.Linear(128, 128), std=1.0)

        # Critic branch
        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024), std=np.sqrt(2))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)

        # Actor branch
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 512), std=1.0)

        # Action embeddings
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()

        self.action_embed_dim = 16
        self.actor_hidden_dim = 512
        self._init_bilinear_actor()

        # Build normalization vector
        if hasattr(env, "feature_normalizations"):
            max_values = [1.0] * self.num_layers
            for feature_id, norm_value in env.feature_normalizations.items():
                if feature_id < self.num_layers:
                    max_values[feature_id] = norm_value if norm_value > 0 else 1.0
            max_vec = torch.tensor(max_values, dtype=torch.float32)[None, :, None, None]
        else:
            max_vec = torch.ones(1, self.num_layers, 1, 1, dtype=torch.float32)
        self.register_buffer("max_vec", max_vec)

        self.active_action_names = []
        self.num_active_actions = 100

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
        """Activate action embeddings."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def network_forward(self, x):
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

    def encode_observations(self, observations, state=None):
        """Encode observations with scatter operation."""
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
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

        # Use scatter-based write
        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index

        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        return self.network_forward(box_obs)

    def decode_actions(self, hidden, batch_size):
        """Decode actions using bilinear interaction."""
        # Critic branch
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)
        actor_features = F.relu(actor_features)

        action_embeds = self.action_embeddings.weight[: self.num_active_actions]
        action_embeds = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        num_actions = action_embeds.shape[1]
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)
        actor_reshaped = actor_repeated.reshape(-1, self.actor_hidden_dim)
        action_embeds_reshaped = action_embeds.reshape(-1, self.action_embed_dim)

        query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)
        query = torch.tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)

        biased_scores = scores + self.actor_bias
        logits = biased_scores.reshape(batch_size, num_actions)

        return logits, value
