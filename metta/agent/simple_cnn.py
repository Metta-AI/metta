"""Simple CNN-based agent implementation."""

from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn

from metta.agent.components.action import ActionEmbedding
from metta.agent.components.actor import MettaActorSingleHead
from metta.agent.components.lstm import LSTM
from metta.agent.components.observation_normalizer import ObservationNormalizer
from metta.agent.policy_state import PolicyState

from .base_agent import BaseAgent


class SimpleCNNAgent(BaseAgent):
    """Simple CNN-based agent with 2 convolutional layers."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        obs_width: int,
        obs_height: int,
        feature_normalizations: dict,
        device: str = "cuda",
        hidden_size: int = 128,
        lstm_layers: int = 2,
    ):
        super().__init__(obs_space, action_space, hidden_size, lstm_layers, device)

        self.obs_width = obs_width
        self.obs_height = obs_height
        self.feature_normalizations = feature_normalizations

        # Get observation shape
        if hasattr(obs_space, "spaces") and "grid_obs" in obs_space.spaces:
            obs_shape = obs_space.spaces["grid_obs"].shape
        else:
            obs_shape = obs_space.shape

        # Observation normalizer
        self.obs_normalizer = ObservationNormalizer(
            input_shape=obs_shape, feature_normalizations=feature_normalizations
        )

        # CNN layers
        self.cnn1 = nn.Conv2d(obs_shape[-1], 64, kernel_size=5, stride=3)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()

        # Calculate flattened size after convolutions
        dummy_input = torch.zeros(1, obs_shape[-1], obs_height, obs_width)
        dummy_output = self.cnn2(self.cnn1(dummy_input))
        flattened_size = dummy_output.numel() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()

        self.encoded_obs = nn.Linear(128, hidden_size)
        self.relu4 = nn.ReLU()

        # LSTM core
        self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers)

        self.core_relu = nn.ReLU()

        # Critic head
        self.critic_1 = nn.Linear(hidden_size, 1024)
        self.critic_activation = nn.Tanh()
        self.value_head = nn.Linear(1024, 1)

        # Actor head
        self.actor_1 = nn.Linear(hidden_size, 512)
        self.actor_relu = nn.ReLU()

        # Action embedding and actor head will be initialized when actions are activated
        self.action_embedding = None
        self.actor_head = None

    def _activate_actions_hook(self, action_names: list[str], action_max_params: list[int]):
        """Initialize action-related components when actions are activated."""
        # Create full action names for embedding
        full_action_names = []
        for action_name, max_param in zip(action_names, action_max_params, strict=False):
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        # Initialize action embedding
        self.action_embedding = ActionEmbedding(num_embeddings=len(full_action_names), embedding_dim=16)
        self.action_embedding.activate_actions(full_action_names, self.device)

        # Initialize actor head
        self.actor_head = MettaActorSingleHead(
            input_size=512, action_embedding_size=16, num_actions=len(full_action_names)
        )

    def compute_outputs(
        self, x: torch.Tensor, state: PolicyState
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute value and action logits from observation."""
        # Handle different input shapes
        if x.dim() == 5:  # (B, T, H, W, C) - training
            B, T, H, W, C = x.shape
            x = x.view(B * T, H, W, C)
            need_reshape = True
        else:  # (BT, H, W, C) - inference
            need_reshape = False

        # Reshape to (BT, C, H, W) for CNN
        x = x.permute(0, 3, 1, 2).float()

        # Normalize observations
        x = self.obs_normalizer(x)

        # CNN layers
        x = self.relu1(self.cnn1(x))
        x = self.relu2(self.cnn2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.encoded_obs(x))

        # LSTM
        if need_reshape:
            x = x.view(B, T, -1)

        # Prepare LSTM state
        lstm_state = None
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_state = (state.lstm_h, state.lstm_c)

        x, (new_h, new_c) = self.lstm(x, lstm_state)

        # Flatten back if needed
        if need_reshape:
            x = x.view(B * T, -1)

        x = self.core_relu(x)

        # Value head
        value = self.value_head(self.critic_activation(self.critic_1(x)))

        # Actor head
        actor_features = self.actor_relu(self.actor_1(x))
        action_embeds = self.action_embedding.weight
        logits = self.actor_head(actor_features, action_embeds)

        return value, logits, (new_h, new_c)
