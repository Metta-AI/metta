import logging

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.pytorch.base import (
    LSTMWrapper,
    bilinear_actor_forward,
    init_bilinear_actor,
    initialize_action_embeddings,
)

logger = logging.getLogger(__name__)


class Fast(LSTMWrapper):
    """Fast CNN-based policy with LSTM that matches the YAML fast.yaml implementation."""

    def __init__(self, env, policy=None, cnn_channels=128, input_size=128, hidden_size=128, num_layers=2):
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        # Pass num_layers=2 to match YAML configuration
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

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
        # Pass through to the policy
        if hasattr(self.policy, "activate_action_embeddings"):
            self.policy.activate_action_embeddings(full_action_names, device)

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        from metta.agent.pytorch.base import convert_action_to_logit_index

        return convert_action_to_logit_index(flattened_action, self.cum_action_max_params)


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128):
        super().__init__()
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
        initialize_action_embeddings(self.action_embeddings)

        # Store for dynamic action head
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512

        # Bilinear layer to match MettaActorSingleHead
        self.actor_W, self.actor_bias = init_bilinear_actor(self.actor_hidden_dim, self.action_embed_dim)

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

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings, matching the YAML ActionEmbedding component behavior."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

        # Could implement proper action name to index mapping here if needed
        # For now, we'll use the first N embeddings

    def network_forward(self, x):
        x = x / self.max_vec
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.encoded_obs(x)
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

        # Get action embeddings for all actions
        # Use only the active actions (first num_active_actions embeddings)
        action_embeds = self.action_embeddings.weight[: self.num_active_actions]  # [num_actions, 16]

        # Expand action embeddings for each batch element
        action_embeds = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B*TT, num_actions, 16]

        # Bilinear interaction matching MettaActorSingleHead
        logits = bilinear_actor_forward(
            actor_features,
            action_embeds,
            self.actor_W,
            self.actor_bias,
            self.actor_hidden_dim,
            self.action_embed_dim,
        )

        return logits, value
