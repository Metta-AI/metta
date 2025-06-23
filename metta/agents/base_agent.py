"""Base agent class that all Metta agents inherit from."""

from typing import Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions


class BaseAgent(nn.Module):
    """Base class for all Metta agents.

    This provides a standard interface and common functionality for agents
    while allowing them to be implemented as regular torch.nn.Module classes.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        device: str = "cuda",
    ):
        super().__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.device = device

        # These will be set when activate_actions is called
        self.action_names = None
        self.action_max_params = None
        self.cum_action_max_params = None
        self.action_index_tensor = None

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device):
        """Activate specific actions for this agent."""
        self.action_names = action_names
        self.action_max_params = action_max_params
        self.device = device

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(torch.tensor([0] + action_max_params, device=device), dim=0)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=device, dtype=torch.int32)

        # Let subclasses do any additional setup
        self._activate_actions_hook(action_names, action_max_params)

    def _activate_actions_hook(self, action_names: list[str], action_max_params: list[int]):
        """Hook for subclasses to do additional setup when actions are activated."""
        pass

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the agent.

        Args:
            x: Input observation tensor
            state: Policy state containing LSTM hidden and cell states
            action: Optional action tensor for training mode

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
        """
        # Get value and logits from the agent's network
        value, logits, new_state = self.compute_outputs(x, state)

        # Update LSTM state
        state.lstm_h = new_state[0]
        state.lstm_c = new_state[1]

        if action is None:
            # Inference mode - sample new actions
            return self._forward_inference(value, logits)
        else:
            # Training mode - evaluate given actions
            return self._forward_training(value, logits, action)

    def compute_outputs(
        self, x: torch.Tensor, state: PolicyState
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute value and action logits from observation.

        This method should be implemented by subclasses.

        Args:
            x: Input observation tensor
            state: Policy state containing LSTM hidden and cell states

        Returns:
            Tuple of (value, logits, (new_lstm_h, new_lstm_c))
        """
        raise NotImplementedError("Subclasses must implement compute_outputs")

    def _forward_inference(
        self, value: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference mode."""
        # Sample actions
        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)

        # Convert logit index to action
        action = self._convert_logit_index_to_action(action_logit_index)

        return action, action_log_prob, entropy, value, log_probs

    def _forward_training(
        self, value: torch.Tensor, logits: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training mode."""
        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)

        action_log_prob, entropy, log_probs = evaluate_actions(logits, action_logit_index)

        return action, action_log_prob, entropy, value, log_probs

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete action indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()

        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        action_logit_indices = action_type_numbers + cumulative_sum + action_params

        return action_logit_indices

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    @property
    def total_params(self):
        """Total number of parameters in the agent."""
        return sum(p.numel() for p in self.parameters())

    @property
    def lstm(self):
        """Access the LSTM module if it exists."""
        return getattr(self, "lstm", None)

    def l2_reg_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss.

        Can be overridden by subclasses to provide custom regularization.

        Returns:
            torch.Tensor: L2 regularization loss (default: 0.0)
        """
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def l2_init_loss(self) -> torch.Tensor:
        """Compute L2 initialization loss.

        Can be overridden by subclasses to provide custom regularization.

        Returns:
            torch.Tensor: L2 initialization loss (default: 0.0)
        """
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def update_l2_init_weight_copy(self):
        """Update the L2 initialization weight copy.

        Can be overridden by subclasses if they maintain initial weight copies.
        """
        pass

    def clip_weights(self):
        """Clip weights to prevent exploding gradients.

        Can be overridden by subclasses to implement weight clipping.
        """
        pass

    def compute_weight_metrics(self, delta: float = 0.01):
        """Compute metrics about the weights.

        Can be overridden by subclasses to provide weight analysis.

        Args:
            delta: Small constant used in metric calculations

        Returns:
            List of weight metric dictionaries (default: empty list)
        """
        return []
