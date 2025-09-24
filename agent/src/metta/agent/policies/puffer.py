from typing import List, Optional

import einops
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.actor import ActionProbs, ActionProbsConfig
from metta.agent.components.lstm import LSTM, LSTMConfig
from metta.agent.policy import Policy, PolicyArchitecture


class PufferPolicyConfig(PolicyArchitecture):
    """
    Policy configuration that exactly matches PufferLib architecture for checkpoint loading.

    Based on analysis of PufferLib checkpoint:
    - CNN: 128 channels, 24 input channels, 5x5 and 3x3 kernels
    - LSTM: 512 hidden size
    - Actor: 5 actions + 9 action args
    - Critic: Single value output
    """

    class_path: str = "metta.agent.policies.puffer.PufferPolicy"

    lstm_config: LSTMConfig = LSTMConfig(
        in_key="encoded_obs",
        out_key="core",
        latent_size=512,  # Match PufferLib: 256 (self) + 256 (cnn) = 512 input to LSTM
        hidden_size=512,  # Match PufferLib LSTM: 512 not 128
        num_layers=1,
    )

    # Minimal action_probs_config to satisfy base class requirement
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class PufferPolicy(Policy):
    """Policy that exactly matches PufferLib architecture for seamless checkpoint loading."""

    def __init__(self, env_metadata, config: Optional[PufferPolicyConfig] = None):
        super().__init__()
        self.config = config or PufferPolicyConfig()
        self.env_metadata = env_metadata
        self.is_continuous = False
        self.action_space = env_metadata.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.out_width = env_metadata.obs_width
        self.out_height = env_metadata.obs_height

        self.num_layers = max(env_metadata.feature_normalizations.keys()) + 1

        self.conv1 = nn.Conv2d(self.num_layers, 128, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(self.num_layers, 256),
            nn.ReLU(),
        )

        # Initialize max_vec based on actual number of features
        max_feature_id = max(env_metadata.feature_normalizations.keys()) + 1
        self.max_vec = [1.0] * max_feature_id
        for feature_id, norm_value in env_metadata.feature_normalizations.items():
            print(f"feature_id: {feature_id}, norm_value: {norm_value}")
            if feature_id < max_feature_id:
                self.max_vec[feature_id] = norm_value if norm_value > 0 else 1.0
        self.max_vec = torch.tensor(self.max_vec, dtype=torch.float32)
        self.max_vec = torch.maximum(self.max_vec, torch.ones_like(self.max_vec))
        self.max_vec = self.max_vec[None, :, None, None]

        # Create actor heads for each action: one for action type, and one for each action's max args + 1
        action_dims = [len(env_metadata.action_names)] + [max_args + 1 for max_args in env_metadata.max_action_args]
        self.actor = nn.ModuleList([nn.Linear(512, n) for n in action_dims])
        self.value = nn.Linear(512, 1)

        # Use LSTM from config
        self.lstm = LSTM(config=self.config.lstm_config)

        # Value head as TensorDictModule
        self.value_head = TDM(self.value, in_keys=["core"], out_keys=["values"])

        # Action probabilities component
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Converts raw observation tokens into a concatenated self + CNN feature vector."""
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coords = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coords = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

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

        batch_idx = torch.arange(B * TT, device=observations.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[
            batch_idx[valid_tokens],
            atr_indices[valid_tokens],
            x_coords[valid_tokens],
            y_coords[valid_tokens],
        ] = atr_values[valid_tokens]

        # Normalize features with epsilon for numerical stability
        max_vec_device = self.max_vec.to(box_obs.device)
        features = box_obs / (max_vec_device + 1e-8)

        # Self encoder processes aggregated per-channel features (sum across spatial dimensions)
        # Shape: [B, num_layers=24] -> [B, 256]
        self_input = features.sum(dim=(-2, -1))  # Sum across height and width dimensions
        self_features = self.self_encoder(self_input)

        # CNN processes spatial features normally
        # Shape: [B, 24, H, W] -> [B, 256]
        cnn_features = self.network(features)

        # Concatenate self and CNN features: [B, 256] + [B, 256] = [B, 512]
        result = torch.cat([self_features, cnn_features], dim=1)
        return result

    def decode_actions(self, hidden):
        """Decode hidden state into action logits."""
        logits = [actor_head(hidden) for actor_head in self.actor]
        return logits

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        """Forward pass through the policy."""
        observations = td["env_obs"]

        # Encode observations: [B, obs] -> [B, 512]
        encoded_obs = self.encode_observations(observations)
        td["encoded_obs"] = encoded_obs

        # Pass through LSTM: [B, 512] -> [B, 512]
        self.lstm(td)

        # Decode actions
        logits = self.decode_actions(td["core"])

        # Concatenate logits for action_probs component (they have different sizes)
        td["logits"] = torch.cat(logits, dim=-1)

        # Get value
        self.value_head(td)

        # Process action probabilities using Metta's ActionProbs component
        self.action_probs(td, action)

        # Flatten values as expected by training
        td["values"] = td["values"].flatten()

        return td

    def initialize_to_environment(self, env_metadata, device: torch.device):
        """Initialize policy components to environment."""
        self.to(device)
        self.action_probs.initialize_to_environment(env_metadata, device)

    def reset_memory(self):
        """Reset LSTM memory."""
        self.lstm.reset_memory()

    def get_agent_experience_spec(self) -> Composite:
        """Get the experience specification for this agent."""
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        """Get the device this policy is on."""
        return next(self.parameters()).device

    @property
    def action_names(self) -> list[str]:
        """Return list of action names."""
        return self.env_metadata.action_names

    @property
    def observation_space(self):
        """Return observation space."""
        return self.env_metadata.observation_space
