from typing import Optional

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.actor import ActionProbs, ActionProbsConfig
from metta.agent.policy import Policy, PolicyArchitecture


class BasicConfig (PolicyArchitecture):
    """Simple PufferLib policy configuration."""

    class_path: str = "metta.agent.policies.puffer_simple.PufferSimplePolicy"

    # Only ActionProbs needed for compatibility
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class BasicPolicy(Policy):
    """Direct PufferLib policy with minimal Metta integration."""

    def __init__(self, env_metadata, config: Optional[PufferSimpleConfig] = None):
        super().__init__()
        self.config = config or PufferSimpleConfig()
        self.env_metadata = env_metadata

        # Create the core PufferLib policy directly
        self.policy = PufferLibPolicy(env_metadata)

        # Only ActionProbs component needed for Metta compatibility
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        """Forward pass - minimal TensorDict wrapper around PufferLib."""
        observations = td["env_obs"]

        # Call PufferLib policy directly
        logits, value = self.policy(observations)

        # Convert back to TensorDict format expected by Metta
        td["logits"] = torch.cat(logits, dim=-1)  # Flatten for ActionProbs
        td["values"] = value.flatten()

        # Use ActionProbs for action sampling/evaluation
        self.action_probs(td, action)

        return td

    def initialize_to_environment(self, env_metadata, device: torch.device):
        """Initialize to environment."""
        self.to(device)
        self.action_probs.initialize_to_environment(env_metadata, device)

    def reset_memory(self):
        """Reset LSTM memory."""
        if hasattr(self.policy, 'reset_memory'):
            self.policy.reset_memory()

    def get_agent_experience_spec(self) -> Composite:
        """Get experience spec."""
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def action_names(self):
        return self.env_metadata.action_names

    @property
    def observation_space(self):
        return self.env_metadata.observation_space


class PufferLibPolicy(nn.Module):

    def __init__(self, env_metadata, cnn_channels=128, hidden_size=512, input_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False

        self.out_width = env_metadata.obs_width
        self.out_height = env_metadata.obs_height

        # Use dynamic layer calculation
        self.num_layers = max(env_metadata.feature_normalizations.keys()) + 1

        # Define CNN layers separately to calculate output size
        self.conv1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3), std=1.0
        )
        self.conv2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1), std=1.0
        )

        # Calculate actual CNN output size dynamically
        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.conv2(torch.relu(self.conv1(test_input)))
            self.cnn_flattened_size = test_output.numel() // test_output.shape[0]

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(
                nn.Linear(self.cnn_flattened_size, hidden_size // 2), std=1.0
            ),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.num_layers, hidden_size // 2), std=1.0
            ),
            nn.ReLU(),
        )

        # Build normalization vector from environment
        max_values = [1.0] * self.num_layers
        for feature_id, norm_value in env_metadata.feature_normalizations.items():
            if feature_id < self.num_layers:
                max_values[feature_id] = norm_value if norm_value > 0 else 1.0

        max_vec = torch.tensor(max_values, dtype=torch.float32)
        max_vec = torch.maximum(max_vec, torch.ones_like(max_vec))
        max_vec = max_vec[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        # Action heads - use max_action_args instead of single_action_space.nvec
        action_nvec = [max_args + 1 for max_args in env_metadata.max_action_args]
        self.actor = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
            for n in action_nvec
        ])

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """PufferLib observation encoding."""
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        x_coords = ((coords_byte >> 4) & 0x0F).long()
        y_coords = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=observations.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )

        batch_idx = (
            torch.arange(B * TT, device=observations.device)
            .unsqueeze(-1)
            .expand_as(atr_values)
        )
        box_obs[
            batch_idx[valid_tokens],
            atr_indices[valid_tokens],
            x_coords[valid_tokens],
            y_coords[valid_tokens],
        ] = atr_values[valid_tokens]

        # Normalize features
        features = box_obs / (self.max_vec + 1e-8)

        # PufferLib uses center pixel for self features
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        result = torch.cat([self_features, cnn_features], dim=1)
        return result

    def decode_actions(self, hidden):
        """PufferLib action decoding."""
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value

    def forward(self, observations: torch.Tensor, state=None):
        """PufferLib forward pass - returns logits and values directly."""
        encoded = self.encode_observations(observations)
        return self.decode_actions(encoded)
