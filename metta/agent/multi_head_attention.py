"""Multi-head attention agent with cross-attention between observations and actions."""

from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn

from metta.agent.components.action import ActionEmbedding
from metta.agent.components.lstm import LSTM
from metta.agent.components.observation_normalizer import ObservationNormalizer
from metta.agent.policy_state import PolicyState

from .base_agent import BaseAgent


class CrossAttentionActorHead(nn.Module):
    """Actor head using cross-attention between features and action embeddings."""

    def __init__(self, feature_dim: int, action_embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Project features and action embeddings to same dimension
        self.feature_proj = nn.Linear(feature_dim, 256)
        self.action_proj = nn.Linear(action_embedding_dim, 256)

        # Cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, action_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)
            action_embeds: (num_actions, embedding_dim)

        Returns:
            logits: (batch_size, num_actions)
        """
        batch_size = features.shape[0]
        num_actions = action_embeds.shape[0]

        # Project features and expand for each action
        feat_proj = self.feature_proj(features).unsqueeze(1)  # (B, 1, 256)
        feat_proj = feat_proj.expand(-1, num_actions, -1)  # (B, num_actions, 256)

        # Project action embeddings and expand for batch
        act_proj = self.action_proj(action_embeds).unsqueeze(0)  # (1, num_actions, 256)
        act_proj = act_proj.expand(batch_size, -1, -1)  # (B, num_actions, 256)

        # Cross attention: features attend to actions
        attended, _ = self.cross_attention(feat_proj, act_proj, act_proj)  # (B, num_actions, 256)

        # Compute logits
        logits = self.output_proj(attended).squeeze(-1)  # (B, num_actions)

        return logits


class MultiHeadAttentionAgent(BaseAgent):
    """Agent using multi-head attention for both spatial processing and action selection."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        obs_width: int,
        obs_height: int,
        feature_normalizations: dict,
        device: str = "cuda",
        hidden_size: int = 512,
        lstm_layers: int = 2,
        num_attention_heads: int = 8,
    ):
        super().__init__(obs_space, action_space, hidden_size, lstm_layers, device)

        self.obs_width = obs_width
        self.obs_height = obs_height
        self.feature_normalizations = feature_normalizations
        self.num_attention_heads = num_attention_heads

        # Get observation shape
        if hasattr(obs_space, "spaces") and "grid_obs" in obs_space.spaces:
            obs_shape = obs_space.spaces["grid_obs"].shape
        else:
            obs_shape = obs_space.shape

        # Observation normalizer
        self.obs_normalizer = ObservationNormalizer(
            input_shape=obs_shape, feature_normalizations=feature_normalizations
        )

        # CNN backbone with residual connections
        self.cnn1 = nn.Conv2d(obs_shape[-1], 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.cnn3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        # Calculate spatial dimensions after convolutions
        dummy_input = torch.zeros(1, obs_shape[-1], obs_height, obs_width)
        dummy_output = self.cnn3(self.cnn2(self.cnn1(dummy_input)))
        _, C, H, W = dummy_output.shape
        self.spatial_size = H * W
        self.feature_dim = C

        # Spatial self-attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=num_attention_heads, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(self.feature_dim)

        # Feature aggregation
        self.feature_mlp = nn.Sequential(nn.Linear(self.feature_dim, hidden_size), nn.ReLU(), nn.Dropout(0.1))

        # LSTM core
        self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers)

        # Value head with deeper network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        # Action embedding and cross-attention actor head
        self.action_embedding = None
        self.actor_head = None

    def _activate_actions_hook(self, action_names: list[str], action_max_params: list[int]):
        """Initialize action-related components when actions are activated."""
        # Create full action names for embedding
        full_action_names = []
        for action_name, max_param in zip(action_names, action_max_params, strict=False):
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        # Initialize action embedding with larger dimension
        self.action_embedding = ActionEmbedding(num_embeddings=len(full_action_names), embedding_dim=64)
        self.action_embedding.activate_actions(full_action_names, self.device)

        # Initialize cross-attention actor head
        self.actor_head = CrossAttentionActorHead(
            feature_dim=self.hidden_size, action_embedding_dim=64, num_heads=self.num_attention_heads
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

        # CNN backbone with batch norm
        x = self.relu1(self.bn1(self.cnn1(x)))
        x = self.relu2(self.bn2(self.cnn2(x)))
        x = self.relu3(self.bn3(self.cnn3(x)))

        # Reshape for self-attention: (BT, C, H, W) -> (BT, H*W, C)
        batch_size = x.size(0)
        x = x.view(batch_size, self.feature_dim, -1).transpose(1, 2)

        # Spatial self-attention with residual
        attended, _ = self.spatial_attention(x, x, x)
        x = self.spatial_norm(x + attended)

        # Global max pooling
        x = x.max(dim=1)[0]  # (BT, C)

        # Feature MLP
        x = self.feature_mlp(x)

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
        value = self.value_net(x)

        # Actor head with cross-attention
        action_embeds = self.action_embedding.weight
        logits = self.actor_head(x, action_embeds)

        return value, logits, (new_h, new_c)
