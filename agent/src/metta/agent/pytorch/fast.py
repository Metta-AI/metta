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

# Import the improved LSTMWrapper from base.py
from metta.agent.pytorch.base import LSTMWrapper

logger = logging.getLogger(__name__)


class Fast(LSTMWrapper):
    """Fast CNN-based policy with LSTM that matches the YAML fast.yaml implementation."""

    def __init__(self, env, policy=None, cnn_channels=128, input_size=128, hidden_size=128, num_layers=2, clip_range=0):
        logger.info(
            f"[DEBUG] Fast.__init__ called with input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}"
        )
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        # Pass num_layers=2 to match YAML configuration
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        logger.info(f"[DEBUG] Fast initialized with {sum(p.numel() for p in self.parameters())} parameters")
        logger.info(f"[DEBUG] LSTM: {self.lstm.num_layers} layers, hidden_size={self.lstm.hidden_size}")
        logger.info("[DEBUG] LSTM bias initialized to 1, weights orthogonal")

        # Initialize parity features to match ComponentPolicy
        self.clip_range = clip_range  # Match YAML's clip_range: 0
        self.analyze_weights_interval = 300  # Match YAML config

    # Memory management methods are inherited from LSTMWrapper base class

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # CRITICAL FIX: Set bptt and batch fields to match ComponentPolicy behavior
        # ComponentPolicy sets these in every forward pass, and the LSTM component depends on them
        if observations.dim() == 4:  # Training: [B, T, obs_tokens, 3]
            B = observations.shape[0]
            TT = observations.shape[1]
            # Flatten batch dimension and set fields exactly like ComponentPolicy
            total_batch = B * TT
            td.set("bptt", torch.full((total_batch,), TT, device=observations.device, dtype=torch.long))
            td.set("batch", torch.full((total_batch,), B, device=observations.device, dtype=torch.long))
        else:  # Inference: [B, obs_tokens, 3]
            B = observations.shape[0]
            TT = 1
            # Set fields for inference mode
            td.set("bptt", torch.full((B,), 1, device=observations.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=observations.device, dtype=torch.long))

        # Encode obs
        hidden = self.policy.encode_observations(observations, state)

        # Use base class method for LSTM state management
        lstm_h, lstm_c, env_id = self._manage_lstm_state(td, B, TT, observations.device)
        lstm_state = (lstm_h, lstm_c)

        # Forward LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)

        # Use base class method to store state with automatic detachment
        self._store_lstm_state(new_lstm_h, new_lstm_c, env_id)

        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode
        logits_list, value = self.policy.decode_actions(flat_hidden, B * TT)

        if action is None:
            # ---------- Inference Mode ----------
            log_probs = F.log_softmax(logits_list, dim=-1)  # [batch_size, num_actions]
            action_probs = torch.exp(log_probs)  # [batch_size, num_actions]

            actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

            batch_indices = torch.arange(actions.shape[0], device=actions.device)
            selected_log_probs = log_probs[batch_indices, actions]  # [batch_size]

            action = self._convert_logit_index_to_action(actions)

            td["actions"] = action.to(dtype=torch.int32)
            td["act_log_prob"] = selected_log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = log_probs

        else:
            # ---------- Training Mode ----------
            # CRITICAL: ComponentPolicy expects the action to be flattened already during training
            # The TD should be reshaped to match the flattened batch dimension
            if action.dim() == 3:  # (B, T, 2) → flatten to (BT, 2)
                batch_size_orig, time_steps, A = action.shape
                action = action.view(batch_size_orig * time_steps, A)
                # Also flatten the TD to match
                if td.batch_dims > 1:
                    td = td.reshape(td.batch_size.numel())

            action_log_probs = F.log_softmax(logits_list, dim=-1)

            action_probs = torch.exp(action_log_probs)
            action_logit_index = self._convert_action_to_logit_index(action)  # shape [BT]

            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            selected_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)  # [batch_size]

            # Store in flattened TD (will be reshaped by caller if needed)
            td["act_log_prob"] = selected_log_probs
            td["entropy"] = entropy
            td["full_log_probs"] = action_log_probs
            td["value"] = value

            # ComponentPolicy reshapes the TD after training forward based on td["batch"] and td["bptt"]
            # The reshaping happens in ComponentPolicy.forward() after forward_training()
            if "batch" in td.keys() and "bptt" in td.keys():
                batch_size = td["batch"][0].item()
                bptt_size = td["bptt"][0].item()
                td = td.reshape(batch_size, bptt_size)

        return td

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings to match ComponentPolicy interface.

        This is called by MettaAgent.activate_actions() but was missing in the
        original Fast implementation, potentially causing the performance difference.
        """
        logger.info(f"[DEBUG] Fast.activate_action_embeddings called with {len(full_action_names)} actions")
        # Pass through to the policy
        if hasattr(self.policy, "activate_action_embeddings"):
            self.policy.activate_action_embeddings(full_action_names, device)

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        # Match ComponentPolicy's formula that compensates for wrong cumsum
        return action_type_numbers + cumulative_sum + action_params

    # Note: Weight clipping, L2-init loss, and weight metrics are now handled
    # by MettaAgent's default implementations which work for any PyTorch policy.
    # Fast doesn't need to override these unless it wants custom behavior.


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128):
        super().__init__()
        logger.info(f"[DEBUG] Fast.Policy.__init__ called with input_size={input_size}, hidden_size={hidden_size}")
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space

        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11

        # Dynamically determine num_layers from environment features
        # This matches what ComponentPolicy does via ObsTokenToBoxShaper
        if hasattr(env, "feature_normalizations"):
            self.num_layers = max(env.feature_normalizations.keys()) + 1
        else:
            # Fallback for environments without feature_normalizations
            self.num_layers = 25  # Default value

        # Match YAML component initialization more closely
        # Use dynamically determined num_layers as input channels
        # Note: YAML uses orthogonal with gain=1, not sqrt(2) like pufferlib default
        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3),
            std=1.0,  # Match YAML orthogonal gain=1
        )
        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            std=1.0,  # Match YAML orthogonal gain=1
        )

        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Match YAML: Linear layers use orthogonal with gain=1
        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(self.flattened_size, 128), std=1.0)
        self.encoded_obs = pufferlib.pytorch.layer_init(nn.Linear(128, 128), std=1.0)

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024), std=np.sqrt(2))
        # value_head has no nonlinearity (YAML: nonlinearity: null), so gain=1
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)

        # Actor branch
        # actor_1 uses gain=1 (YAML default for Linear layers with ReLU)
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 512), std=1.0)

        # Action embeddings - will be properly initialized via activate_action_embeddings
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()

        # Store for dynamic action head
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512

        # Bilinear layer to match MettaActorSingleHead
        self._init_bilinear_actor()

        # Build normalization vector dynamically from environment
        # This matches what ObservationNormalizer does in ComponentPolicy
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

        # Note: Weight tracking, clipping, and L2-init are now handled by MettaAgent's
        # default implementations. We only need to track effective_rank for critic_1
        # to match the YAML configuration's specific requirement.
        self.effective_rank_enabled = True  # For critic_1 matching YAML

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
        logger.info(f"[DEBUG] Policy.activate_action_embeddings called with {len(full_action_names)} actions")
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

        # Could implement proper action name to index mapping here if needed
        # For now, we'll use the first N embeddings

    def network_forward(self, x):
        x = x / self.max_vec
        x = self.cnn1(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after cnn1
        x = self.cnn2(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after cnn2
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after fc1
        x = self.encoded_obs(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after encoded_obs
        return x

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

        return self.network_forward(box_obs)

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

    # Note: The following methods are now handled by MettaAgent's default implementations:
    # - _store_initial_weights()
    # - clip_weights()
    # - l2_init_loss()
    # - update_l2_init_weight_copy()
    # - compute_weight_metrics()
    #
    # MettaAgent provides general implementations that work for any PyTorch policy.
    # We could override them here if we needed custom behavior, but the defaults
    # work perfectly for Fast.
