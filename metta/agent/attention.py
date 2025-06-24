"""Self-attention based agent implementation."""

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


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for spatial feature processing."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)


class AttentionAgent(BaseAgent):
    """Agent with self-attention layers for spatial reasoning."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        obs_width: int,
        obs_height: int,
        feature_normalizations: dict,
        device: str = "cuda",
        hidden_size: int = 256,
        lstm_layers: int = 2,
        num_attention_heads: int = 8,
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

        # CNN backbone
        self.cnn1 = nn.Conv2d(obs_shape[-1], 64, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()

        # Calculate spatial dimensions after convolutions
        dummy_input = torch.zeros(1, obs_shape[-1], obs_height, obs_width)
        dummy_output = self.cnn2(self.cnn1(dummy_input))
        _, C, H, W = dummy_output.shape
        self.spatial_size = H * W
        self.feature_dim = C

        # Self-attention layers
        self.spatial_attention = SelfAttentionLayer(embed_dim=self.feature_dim, num_heads=num_attention_heads)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM core
        self.lstm = LSTM(input_size=self.feature_dim, hidden_size=hidden_size, num_layers=lstm_layers)

        # Critic head
        self.critic_1 = nn.Linear(hidden_size, 1024)
        self.critic_relu = nn.ReLU()
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
        self.action_embedding = ActionEmbedding(num_embeddings=len(full_action_names), embedding_dim=32)
        self.action_embedding.activate_actions(full_action_names, self.device)

        # Initialize actor head
        self.actor_head = MettaActorSingleHead(
            input_size=512, action_embedding_size=32, num_actions=len(full_action_names)
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
            B_T = x.shape[0]

        # Reshape to (BT, C, H, W) for CNN
        x = x.permute(0, 3, 1, 2).float()

        # Normalize observations
        x = self.obs_normalizer(x)

        # CNN backbone
        x = self.relu1(self.cnn1(x))
        x = self.relu2(self.cnn2(x))

        # Reshape for self-attention: (BT, C, H, W) -> (BT, H*W, C)
        x = x.view(x.size(0), self.feature_dim, -1).transpose(1, 2)

        # Apply self-attention
        x = self.spatial_attention(x)

        # Global pooling: (BT, H*W, C) -> (BT, C)
        x = x.transpose(1, 2)  # (BT, C, H*W)
        x = self.global_pool(x).squeeze(-1)  # (BT, C)

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

        # Value head
        value = self.value_head(self.critic_relu(self.critic_1(x)))

        # Actor head
        actor_features = self.actor_relu(self.actor_1(x))
        action_embeds = self.action_embedding.weight
        logits = self.actor_head(actor_features, action_embeds)

        return value, logits, (new_h, new_c)
