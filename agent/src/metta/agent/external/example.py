from __future__ import annotations

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
    ):
        """Pass initialization to wrapped policy."""
        if hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(features, action_names, action_max_params, device)


class Policy(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Define the standard feature order and their empirically determined normalizations
        # This acts like original_feature_mapping in MettaAgent
        self.feature_normalizations = {
            "type_id": 9.0,
            "agent:group": 1.0,
            "hp": 1.0,
            "agent:frozen": 10.0,
            "agent:orientation": 3.0,
            "agent:color": 254.0,
            "converting": 1.0,
            "swappable": 1.0,
            "episode_completion_pct": 235.0,
            "last_action": 8.0,
            "last_action_arg": 9.0,
            "last_reward": 250.0,
            "agent:glyph": 29.0,
            "resource_rewards": 1.0,
            # Inventory features (positions 14-21)
            "inv:0": 1.0,
            "inv:1": 8.0,
            "inv:2": 1.0,
            "inv:3": 1.0,
            "inv:4": 6.0,
            "inv:5": 3.0,
            "inv:6": 1.0,
            "inv:7": 2.0,
        }

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, hidden_size // 2)),
            nn.ReLU(),
        )

        # Initialize max_vec with ones - will be properly set during initialize_to_environment
        # This ensures the model works even if initialize_to_environment isn't called
        max_vec = torch.ones(self.num_layers, dtype=torch.float32)[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        action_nvec = env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec]
        )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1
        if token_observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = token_observations.shape[1]
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"
        token_observations[token_observations == 255] = 0

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = token_observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = token_observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # In ObservationShaper we permute. Here, we create the observations pre-permuted.
        # We'd like to pre-create this as part of initialization, but we don't know the batch size or time steps at
        # that point.
        box_obs = torch.zeros(
            (B * TT, 22, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)

        # Add bounds checking to prevent out-of-bounds access
        valid_tokens = coords_byte != 0xFF
        valid_tokens = valid_tokens & (x_coord_indices < self.out_width) & (y_coord_indices < self.out_height)
        valid_tokens = valid_tokens & (atr_indices < 22)  # Also check attribute indices

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        observations = box_obs

        features = observations / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        # hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
    ):
        """Initialize policy by mapping our feature normalizations to current environment IDs.

        This works like MettaAgent's feature remapping: we have a fixed set of known
        features with empirically determined normalizations, and we map them to whatever
        IDs the current environment uses.
        """
        # Create max_vec based on current environment's feature IDs
        max_values = [1.0] * self.num_layers  # Default normalization

        # Map our known features to the environment's feature IDs
        for feature_name, feature_props in features.items():
            if "id" in feature_props and 0 <= feature_props["id"] < self.num_layers:
                feature_id = feature_props["id"]

                # Check if this is a feature we know about
                if feature_name in self.feature_normalizations:
                    # Use our empirically determined normalization
                    max_values[feature_id] = self.feature_normalizations[feature_name]
                elif feature_name.startswith("inv:") and "inv:0" in self.feature_normalizations:
                    # For unknown inventory items, use a default inventory normalization
                    max_values[feature_id] = 100.0  # DEFAULT_INVENTORY_NORMALIZATION
                elif "normalization" in feature_props:
                    # Use environment's normalization for unknown features
                    max_values[feature_id] = feature_props["normalization"]

        # Update max_vec with the mapped values
        new_max_vec = torch.tensor(max_values, dtype=torch.float32, device=device)[None, :, None, None]
        self.max_vec.data = new_max_vec
