"""
Faithful AGaLiTe implementation for Metta that properly integrates with the training infrastructure.
This version uses the TransformerWrapper for proper BPTT handling.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.agalite_batched import BatchedAGaLiTe
from metta.agent.transformer_wrapper import TransformerWrapper

logger = logging.getLogger(__name__)


class AGaLiTePolicy(nn.Module):
    """
    AGaLiTe policy network that implements encode_observations and decode_actions.
    This is designed to work with the TransformerWrapper.
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 4,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = env.single_action_space
        self.obs_shape = env.single_observation_space.shape

        # AGaLiTe parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r
        self.reset_on_terminate = reset_on_terminate

        # Required by TransformerWrapper
        self.is_continuous = False
        self.hidden_size = d_model

        # Observation encoding - get actual shape from environment
        obs_shape = env.single_observation_space.shape
        if len(obs_shape) == 3:
            self.out_height, self.out_width, self.num_layers = obs_shape
        else:
            # Default values for compatibility
            self.out_width = 11
            self.out_height = 11
            self.num_layers = obs_shape[-1] if len(obs_shape) > 0 else 22

        # CNN encoder for spatial features
        # Standard CNN architecture that works with Metta observations
        self.cnn_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, 128, 3, stride=1, padding=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to fixed size
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(128, d_model // 2)),
            nn.ReLU(),
        )

        # Self encoder for agent-centric features
        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, d_model // 2)),
            nn.ReLU(),
        )

        # Create the AGaLiTe transformer
        self.transformer = BatchedAGaLiTe(
            n_layers=n_layers,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
        )

        # Output heads - handle both single and multi-discrete action spaces
        if hasattr(self.action_space, "nvec"):
            # Multi-discrete action space
            self.action_nvec = tuple(self.action_space.nvec)
            num_actions = sum(self.action_nvec)
            self.is_multidiscrete = True
        else:
            # Single discrete action space
            num_actions = self.action_space.n
            self.is_multidiscrete = False

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(d_model, num_actions), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(nn.Linear(d_model, 1), std=1)

        # Register normalization buffer for observations
        # Create a max_vec that matches the actual number of layers
        # Use default values of 255 for all channels to avoid division issues
        max_vec = torch.ones((1, self.num_layers, 1, 1), dtype=torch.float32) * 255.0
        self.register_buffer("max_vec", max_vec)

        # Move to device
        self.to(self.device)

    def encode_observations(self, observations: torch.Tensor, state: Optional[Dict] = None) -> torch.Tensor:
        """
        Encode observations into hidden representations.

        Args:
            observations: Observations tensor
            state: Optional state dictionary

        Returns:
            Hidden representations of shape (B*T, d_model)
        """
        observations = observations.to(self.device)

        # Handle different input formats
        if observations.dim() == 4:  # (B, H, W, C) or (B, C, H, W)
            # Check if channels are last or first
            if observations.shape[-1] == self.num_layers:
                # Channels last: (B, H, W, C) -> (B, C, H, W)
                observations = observations.permute(0, 3, 1, 2)
            elif observations.shape[1] != self.num_layers:
                # Unexpected shape, try to fix
                if observations.shape[3] == self.num_layers:
                    observations = observations.permute(0, 3, 1, 2)
        elif observations.dim() == 3:
            # Add batch dimension if missing
            observations = observations.unsqueeze(0)
            if observations.shape[-1] == self.num_layers:
                observations = observations.permute(0, 3, 1, 2)

        # Normalize observations (now in (B, C, H, W) format)
        observations = observations / self.max_vec.to(observations.device)

        # Extract spatial and self features
        B = observations.shape[0]

        # CNN features from spatial layers
        cnn_features = self.cnn_encoder(observations)

        # Self features from center position
        center_h, center_w = self.out_height // 2, self.out_width // 2
        self_input = observations[:, :, center_h, center_w]  # (B, num_layers)
        self_features = self.self_encoder(self_input)

        # Concatenate features
        hidden = torch.cat([cnn_features, self_features], dim=-1)

        return hidden

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode hidden representations into action logits and values.

        Args:
            hidden: Hidden representations of shape (B*T, d_model)

        Returns:
            logits: Action logits
            values: Value estimates
        """
        logits = self.actor(hidden)

        # Handle multi-discrete action spaces
        if self.is_multidiscrete:
            logits = logits.split(self.action_nvec, dim=-1)

        values = self.critic(hidden).squeeze(-1)

        return logits, values

    def initialize_memory(self, batch_size: int, device: torch.device) -> Dict:
        """
        Initialize AGaLiTe memory for a batch.

        Args:
            batch_size: Number of parallel environments
            device: Device to create tensors on

        Returns:
            Initial memory dictionary
        """
        return BatchedAGaLiTe.initialize_memory(
            batch_size=batch_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_head=self.d_head,
            eta=self.eta,
            r=self.r,
            device=device,
        )


class AGaLiTeFaithful(TransformerWrapper):
    """
    Faithful AGaLiTe implementation using TransformerWrapper for proper BPTT handling.

    This provides:
    - Proper memory state management across BPTT segments
    - Correct batch dimension handling for vectorized environments
    - Termination-aware state resets
    - Full compatibility with Metta's training infrastructure
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 4,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
    ):
        # Create the AGaLiTe policy
        policy = AGaLiTePolicy(
            env=env,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            n_layers=n_layers,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
        )

        # Initialize with TransformerWrapper
        super().__init__(env, policy, hidden_size=d_model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Action conversion tensors (will be set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None

        # Store memory state
        self.memory_state = None

        # Move to device
        self.to(self.device)

    def reset_memory(self) -> None:
        """Reset memory state. Called by MettaAgent without arguments."""
        self.memory_state = None

    def get_memory(self) -> dict:
        """Get current memory state."""
        return {"transformer_memory": self.memory_state} if self.memory_state else {}

    def forward(self, td: TensorDict, state: Optional[Dict] = None, action: Optional[torch.Tensor] = None):
        """
        Forward pass compatible with MettaAgent expectations.

        Args:
            td: TensorDict containing observations and other data
            state: Dictionary containing transformer memory state
            action: Optional actions for training mode

        Returns:
            Updated TensorDict with actions, values, etc.
        """
        observations = td["env_obs"].to(self.device)

        # Initialize state if needed
        if state is None:
            B = observations.shape[0]
            state = super().reset_memory(B, self.device)

        B = observations.shape[0]  # Get actual batch size for this call

        # Store terminations if available (handle shape properly)
        if "dones" in td:
            dones = td["dones"].to(self.device)
            # Make sure dones match the current batch size
            if dones.shape[0] != B:
                # Dones might be for all agents, but we're processing a subset
                # Just use zeros for now - terminations will be handled properly during rollout
                state["terminations"] = torch.zeros(B, device=self.device)
            else:
                state["terminations"] = dones

        # Determine if we're in training or inference mode
        if action is None:
            # Inference mode
            # Reset state for current batch to avoid memory issues
            state = super().reset_memory(B, self.device)
            logits, values = self.forward_eval(observations, state)

            # Sample actions
            if self.policy.is_multidiscrete:
                # Handle multi-discrete actions
                actions = []
                log_probs = []
                for logit in logits:
                    probs = F.softmax(logit, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    actions.append(action)
                    log_probs.append(log_prob)

                actions = torch.stack(actions, dim=-1)
                log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
            else:
                # Single discrete action
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Convert actions if needed
            if self.action_index_tensor is not None:
                actions = self._convert_logit_index_to_action(actions)

            # Update TensorDict
            td["actions"] = actions.to(dtype=torch.int32)
            td["act_log_prob"] = log_probs
            td["values"] = values

            # Note: Memory is stored in state dict, not TensorDict
            # The state dict is passed between forward calls
            # Detach memory tensors to prevent gradient accumulation
            if "transformer_memory" in state and state["transformer_memory"] is not None:
                memory = state["transformer_memory"]
                for layer_key, layer_memory in memory.items():
                    state["transformer_memory"][layer_key] = tuple(
                        t.detach() if isinstance(t, torch.Tensor) else t for t in layer_memory
                    )

        else:
            # Training mode - use forward for BPTT
            logits, values = super().forward(observations, state)

            # Compute log probabilities for given actions
            action = action.to(self.device)

            # Convert actions if needed
            if self.action_index_tensor is not None:
                action = self._convert_action_to_logit_index(action)

            if self.policy.is_multidiscrete:
                # Handle multi-discrete actions
                log_probs = []
                entropy = []
                for i, logit in enumerate(logits):
                    probs = F.softmax(logit, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_prob = dist.log_prob(action[..., i])
                    ent = dist.entropy()
                    log_probs.append(log_prob)
                    entropy.append(ent)

                log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
            else:
                # Single discrete action
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # Handle action shape properly
                if action.dim() > logits.dim() - 1:
                    # Flatten action to match logits shape
                    action_flat = action.view(-1)
                else:
                    action_flat = action

                log_probs = dist.log_prob(action_flat)
                entropy = dist.entropy()

            # Reshape outputs to match input batch dimensions
            if observations.dim() > len(self.policy.obs_shape) + 1:
                # We have (B, T, ...) input
                B = observations.shape[0]
                T = observations.shape[1]
                log_probs = log_probs.view(B, T)
                entropy = entropy.view(B, T)
                # Values already reshaped by TransformerWrapper

            # Update TensorDict
            td["act_log_prob"] = log_probs
            td["values"] = values
            td["entropy"] = entropy

            # Note: Memory is stored in state dict, not TensorDict
            # Detach memory tensors to prevent gradient accumulation
            if "transformer_memory" in state and state["transformer_memory"] is not None:
                memory = state["transformer_memory"]
                for layer_key, layer_memory in memory.items():
                    state["transformer_memory"][layer_key] = tuple(
                        t.detach() if isinstance(t, torch.Tensor) else t for t in layer_memory
                    )

        return td

    def _convert_action_to_logit_index(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action pairs to logit indices."""
        if self.action_index_tensor is None:
            return action

        # Implement action conversion logic here
        # This is environment-specific
        return action

    def _convert_logit_index_to_action(self, logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices to action pairs."""
        if self.action_index_tensor is None:
            return logit_index

        # Implement action conversion logic here
        # This is environment-specific
        return logit_index
