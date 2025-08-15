import logging
import math

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.util.weights_analysis import analyze_weights

logger = logging.getLogger(__name__)


class LSTMWrapper(nn.Module):
    """Enhanced LSTM wrapper that supports multi-layer LSTMs.

    Based on pufferlib.models.LSTMWrapper but with num_layers support
    to match the YAML fast.yaml implementation which uses 2 layers.
    """

    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=2):
        """Initialize LSTM wrapper with configurable number of layers.

        Args:
            env: Environment
            policy: The policy to wrap (must have encode_observations and decode_actions)
            input_size: Input size to LSTM
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers (default 2 to match YAML)
        """
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_continuous = self.policy.is_continuous

        # Create multi-layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        # Initialize parameters after LSTM creation
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Match YAML agent initialization
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # Orthogonal initialization

        # Note: We don't create LSTMCell for multi-layer LSTMs
        # as it would only work for single layer

        # Store action conversion tensors (will be set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None


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

    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Encode obs
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare LSTM state
        lstm_h, lstm_c = state.get("lstm_h"), state.get("lstm_c")
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h[: self.lstm.num_layers]
            lstm_c = lstm_c[: self.lstm.num_layers]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # Forward LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
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
            action = action
            if action.dim() == 3:  # (B, T, 2) → flatten to (BT, 2)
                B, T, A = action.shape
                action = action.view(B * T, A)

            action_log_probs = F.log_softmax(logits_list, dim=-1)

            action_probs = torch.exp(action_log_probs)
            action_logit_index = self._convert_action_to_logit_index(action)  # shape [BT]

            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            selected_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)  # [batch_size]

            td["act_log_prob"] = selected_log_probs.view(B, TT)
            td["entropy"] = entropy.view(B, TT)
            td["full_log_probs"] = action_log_probs.view(B, TT, -1)
            td["value"] = value.view(B, TT, -1)

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

    def clip_weights(self):
        """Clip weights to match ComponentPolicy's weight clipping."""
        if hasattr(self, "clip_range") and self.clip_range > 0:
            # Delegate to policy for weight clipping
            if hasattr(self.policy, "clip_weights"):
                self.policy.clip_weights(self.clip_range)

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2-init regularization loss to match ComponentPolicy."""
        if hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss()
        return torch.tensor(0.0, dtype=torch.float32)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies to match ComponentPolicy."""
        if hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics to match ComponentPolicy."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []


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
        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=self.num_layers, out_channels=64, kernel_size=5, stride=3)
        )
        self.cnn2 = pufferlib.pytorch.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1))

        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(self.flattened_size, 128))
        self.encoded_obs = pufferlib.pytorch.layer_init(nn.Linear(128, 128))

        # Critic branch
        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)

        # Actor branch
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 512))

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

        # Initialize weight tracking for parity features
        self._store_initial_weights()
        self.clip_scale = 1  # Match ParamLayer default
        self.l2_init_scale = 1  # Match ParamLayer default
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
        """

        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"
        token_observations[token_observations == 255] = 0

        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = token_observations[..., 1].long()
        atr_values = token_observations[..., 2].float()

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)

        valid_tokens = coords_byte != 0xFF
        valid_tokens = valid_tokens & (x_coord_indices < self.out_width) & (y_coord_indices < self.out_height)
        valid_tokens = valid_tokens & (atr_indices < self.num_layers)

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

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

    def _store_initial_weights(self):
        """Store initial weights for L2-init regularization."""
        self.initial_weights = {}
        # Store initial weights for all linear layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if hasattr(module, "weight"):
                    self.initial_weights[name] = module.weight.data.clone()

    def clip_weights(self, clip_range: float):
        """Clip weights to prevent exploding gradients, matching ComponentPolicy."""
        if clip_range > 0:
            with torch.no_grad():
                for name, module in self.named_modules():
                    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        if hasattr(module, "weight"):
                            # Calculate clip value based on largest initial weight
                            if name in self.initial_weights:
                                largest_weight = self.initial_weights[name].abs().max().item()
                                clip_value = clip_range * largest_weight * self.clip_scale
                                module.weight.data = module.weight.data.clamp(-clip_value, clip_value)

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2-init regularization loss, matching ComponentPolicy."""
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=next(self.parameters()).device)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if hasattr(module, "weight") and name in self.initial_weights:
                    weight_diff = module.weight - self.initial_weights[name].to(module.weight.device)
                    total_loss += torch.sum(weight_diff**2) * self.l2_init_scale

        return total_loss

    def update_l2_init_weight_copy(self):
        """Update the stored initial weights to current weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if hasattr(module, "weight"):
                    self.initial_weights[name] = module.weight.data.clone()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis, matching ComponentPolicy."""
        metrics_list = []

        # Compute metrics for 2D weight matrices
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.data.dim() == 2:
                    metrics = analyze_weights(module.weight.data, delta)
                    metrics["name"] = name

                    # Add effective rank tracking for critic_1 to match YAML
                    if "critic_1" in name and self.effective_rank_enabled:
                        metrics["effective_rank_enabled"] = True

                    metrics_list.append(metrics)

        return metrics_list
