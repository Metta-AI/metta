from typing import Optional

import einops
import pufferlib.pytorch
import torch
import torch.nn as nn
from tensordict import TensorDict
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
    
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class PufferPolicy(Policy):
    """Policy that exactly matches PufferLib architecture for seamless checkpoint loading."""

    def __init__(self, env_metadata, config: Optional[PufferPolicyConfig] = None):
        super().__init__()

        # Create a submodule called 'policy' to match checkpoint structure
        self.policy = torch.nn.Module()
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

        self.num_layers = 24
        hidden_size = 512
        cnn_channels = 128

        # Define CNN layers separately to calculate output size (matching PufferLib)
        self.policy.conv1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3), std=1.0
        )
        self.policy.conv2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1), std=1.0
        )

        # Calculate actual CNN output size dynamically (matching PufferLib)
        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.policy.conv2(torch.relu(self.policy.conv1(test_input)))
            self.cnn_flattened_size = test_output.numel() // test_output.shape[0]

        self.policy.network = nn.Sequential(
            self.policy.conv1,
            nn.ReLU(),
            self.policy.conv2,
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(
                nn.Linear(self.cnn_flattened_size, hidden_size // 2), std=1.0
            ),
            nn.ReLU(),
        )

        self.policy.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.num_layers, hidden_size // 2), std=1.0
            ),
            nn.ReLU(),
        )

        # Build normalization vector dynamically from environment feature_normalizations (matching PufferLib)
        max_values = [1.0] * self.num_layers  # Default to 1.0
        for feature_id, norm_value in env_metadata.feature_normalizations.items():
            if feature_id < self.num_layers:
                max_values[feature_id] = norm_value if norm_value > 0 else 1.0

        max_vec = torch.tensor(max_values, dtype=torch.float32)
        # Clamp minimum value to 1.0 to avoid near-zero divisions
        max_vec = torch.maximum(max_vec, torch.ones_like(max_vec))
        max_vec = max_vec[None, :, None, None]
        self.policy.register_buffer("max_vec", max_vec)

        action_nvec = [5, 9]  # Match checkpoint exactly: actor.0 (5 actions), actor.1 (9 actions)
        self.policy.actor = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
            for n in action_nvec
        ])
        self.policy.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1)

        # Initialize LSTM weights to match PufferLib initialization
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)

        # LSTM state storage for persistence between forward passes
        self._hidden_state = None
        self._cell_state = None

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
        max_vec_device = self.policy.max_vec.to(box_obs.device)
        features = box_obs / (max_vec_device + 1e-8)

        # Self encoder processes center pixel features (matching PufferLib exactly)
        # Shape: [B, num_layers] -> [B, 256]
        self_features = self.policy.self_encoder(features[:, :, 5, 5])

        # CNN processes spatial features normally
        # Shape: [B, 24, H, W] -> [B, 256]
        cnn_features = self.policy.network(features)

        # Concatenate self and CNN features: [B, 256] + [B, 256] = [B, 512]
        result = torch.cat([self_features, cnn_features], dim=1)
        return result

    def decode_actions(self, hidden):
        """Decode hidden state into action logits (matching PufferLib exactly)."""
        logits = [dec(hidden) for dec in self.policy.actor]
        value = self.policy.value(hidden)
        return logits, value

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        """Forward pass through the policy."""
        observations = td["env_obs"]

        print(self._hidden_state, self._cell_state)

        # [B, obs] -> [B, 512]
        encoded_obs = self.encode_observations(observations)
        td["encoded_obs"] = encoded_obs

        # Pass through LSTM: [1, B, 512] -> [1, B, 512]
        lstm_input = encoded_obs.unsqueeze(0)
        batch_size = encoded_obs.shape[0]

        # Initialize state if None
        if (self._hidden_state is None or self._cell_state is None or
            self._hidden_state.shape[1] != batch_size):
            device = encoded_obs.device
            self._hidden_state = torch.zeros(1, batch_size, 512, device=device)
            self._cell_state = torch.zeros(1, batch_size, 512, device=device)

        lstm_output, (self._hidden_state, self._cell_state) = self.lstm(
            lstm_input, (self._hidden_state, self._cell_state)
        )
        
        # [1, B, 512] -> [B, 512]
        core_features = lstm_output.squeeze(0)
        # returns separate logits per action type
        logits, value = self.decode_actions(core_features)

        # For ActionProbs compatibility, we need to flatten logits into single tensor
        # This matches how ActionEmbedding creates action names like "move_0", "attack_0", "attack_1", etc.
        td["logits"] = torch.cat(logits, dim=-1)
        td["values"] = value.flatten()
        self.action_probs(td, action)

        return td

    def initialize_to_environment(self, env_metadata, device: torch.device):
        """Initialize policy components to environment."""
        self.to(device)
        self.action_probs.initialize_to_environment(env_metadata, device)

    def reset_memory(self):
        """Reset LSTM memory by clearing stored hidden and cell states."""
        self._hidden_state = None
        self._cell_state = None

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
