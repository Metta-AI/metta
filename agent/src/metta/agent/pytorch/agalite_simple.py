"""
Simplified AGaLiTe implementation that works with Metta's batching.
This version doesn't try to maintain memory across different batch sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pufferlib.pytorch
from typing import Optional, Tuple
from tensordict import TensorDict


class AGaLiTeSimple(nn.Module):
    """Simplified AGaLiTe that works with Metta's training infrastructure."""

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get environment info
        self.action_space = env.single_action_space
        obs_shape = env.single_observation_space.shape
        print(f"DEBUG: obs_shape from env = {obs_shape}")

        # Observation shape - handle token observations which might be (max_tokens, attributes)
        if len(obs_shape) == 2:
            # Token observations - (max_tokens, attributes)
            # We'll treat this differently - just use a simple embedding
            self.is_token_obs = True
            self.max_tokens, self.num_attributes = obs_shape
            self.num_layers = self.num_attributes  # For compatibility
            self.out_height = self.out_width = 11  # Dummy values
        elif len(obs_shape) == 3:
            # Grid observations - (height, width, channels)
            self.is_token_obs = False
            self.out_height, self.out_width, self.num_layers = obs_shape
        else:
            # Default fallback
            self.is_token_obs = False
            self.out_height = self.out_width = 11
            self.num_layers = obs_shape[-1] if len(obs_shape) > 0 else 48

        # Model parameters
        self.d_model = d_model
        self.is_continuous = False

        # Create appropriate encoder based on observation type
        if self.is_token_obs:
            # For token observations, use an embedding + linear layers
            self.token_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(self.max_tokens * self.num_attributes, d_model)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(d_model, d_model)),
                nn.ReLU(),
            )
            # No CNN or self encoder for token obs
            self.cnn_encoder = None
            self.self_encoder = None
        else:
            # For grid observations, use CNN
            self.token_encoder = None
            self.cnn_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, 128, 3, padding=1)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, padding=1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                pufferlib.pytorch.layer_init(nn.Linear(128, d_model // 2)),
                nn.ReLU(),
            )
            # Self encoder
            self.self_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, d_model // 2)),
                nn.ReLU(),
            )

        # Simple transformer layers (without persistent memory)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Output heads
        if hasattr(self.action_space, "nvec"):
            num_actions = sum(self.action_space.nvec)
            self.is_multidiscrete = True
            self.action_nvec = tuple(self.action_space.nvec)
        else:
            num_actions = self.action_space.n
            self.is_multidiscrete = False

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(d_model, num_actions), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(nn.Linear(d_model, 1), std=1)

        # Normalization
        self.register_buffer("max_vec", torch.ones((1, self.num_layers, 1, 1)) * 255.0)

        # Action conversion (set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.to(self.device)

    def reset_memory(self):
        """No persistent memory in this simple version."""
        pass

    def get_memory(self):
        """No persistent memory in this simple version."""
        return {}

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations to hidden representations."""
        observations = observations.to(self.device).float()  # Ensure float dtype
        B = observations.shape[0]
        
        if self.is_token_obs:
            # Token observations - just flatten and encode
            hidden = observations.view(B, -1)  # Flatten to (B, max_tokens * attributes)
            hidden = self.token_encoder(hidden)
        else:
            # Grid observations - use CNN
            # Handle different input formats - observations come as (B, H, W, C)
            if observations.dim() == 4:
                # Always permute from (B, H, W, C) to (B, C, H, W)
                if observations.shape[-1] > observations.shape[1]:
                    # Channels are last dimension
                    observations = observations.permute(0, 3, 1, 2)
            elif observations.dim() == 3:
                # Single observation, add batch dim
                observations = observations.unsqueeze(0)
                if observations.shape[-1] > observations.shape[1]:
                    observations = observations.permute(0, 3, 1, 2)

            # Normalize (skip if shapes don't match)
            if observations.shape[1] == self.max_vec.shape[1]:
                observations = observations / self.max_vec.to(observations.device)
            else:
                # Just use a simple normalization
                observations = observations / 255.0

            # Encode with CNN
            cnn_features = self.cnn_encoder(observations)

            # Self features
            center_h, center_w = self.out_height // 2, self.out_width // 2
            self_input = observations[:, :, center_h, center_w]
            self_features = self.self_encoder(self_input)

            # Combine
            hidden = torch.cat([cnn_features, self_features], dim=-1)
        
        return hidden

    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass compatible with MettaAgent."""
        observations = td["env_obs"].to(self.device)

        # Encode observations
        hidden = self.encode_observations(observations)

        # Pass through transformer layers (no memory, just self-attention)
        for layer in self.transformer_layers:
            # Add batch/sequence dimension if needed
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1)  # (B, 1, d_model)
            hidden = layer(hidden)
            if hidden.dim() == 3 and hidden.shape[1] == 1:
                hidden = hidden.squeeze(1)  # Back to (B, d_model)

        # Get logits and values
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)

        # Handle multi-discrete
        if self.is_multidiscrete:
            logits = logits.split(self.action_nvec, dim=-1)

        if action is None:
            # Inference mode - sample actions
            if self.is_multidiscrete:
                actions = []
                log_probs = []
                for logit in logits:
                    probs = F.softmax(logit, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                    actions.append(a)
                    log_probs.append(lp)
                actions = torch.stack(actions, dim=-1)
                log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            td["actions"] = actions.to(dtype=torch.int32)
            td["act_log_prob"] = log_probs
            td["values"] = values
        else:
            # Training mode - compute log probs for given actions
            action = action.to(self.device)

            if self.is_multidiscrete:
                log_probs = []
                entropy = []
                for i, logit in enumerate(logits):
                    probs = F.softmax(logit, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    lp = dist.log_prob(action[..., i])
                    ent = dist.entropy()
                    log_probs.append(lp)
                    entropy.append(ent)
                log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # Handle action shape
                if action.dim() > 1:
                    action_flat = action.view(-1)
                else:
                    action_flat = action

                log_probs = dist.log_prob(action_flat)
                entropy = dist.entropy()

            # Handle batch shapes for training
            obs_dim = 3  # Assuming 3D observations (H, W, C)
            if observations.dim() > obs_dim + 1:
                B = observations.shape[0]
                T = observations.shape[1]
                if log_probs.numel() == B * T:
                    log_probs = log_probs.view(B, T)
                    entropy = entropy.view(B, T)
                    values = values.view(B, T)

            td["act_log_prob"] = log_probs
            td["values"] = values
            td["entropy"] = entropy

        return td
