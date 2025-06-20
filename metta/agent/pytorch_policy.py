import logging
from typing import Optional

import torch
from torch import nn

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

logger = logging.getLogger("pytorch_policy")


class PytorchPolicy(nn.Module):
    """Base class for external PyTorch policies to integrate with MettaAgent.

    This class provides a standard interface that external policies can implement
    to work seamlessly with the Metta training infrastructure. It handles the
    translation between different naming conventions and provides default
    implementations for required methods.

    External policies should either:
    1. Inherit from this class and override the necessary methods
    2. Be wrapped by this class if they follow a different interface
    """

    def __init__(self, policy: Optional[nn.Module] = None):
        super().__init__()
        if policy is not None:
            # Wrapping an existing policy
            self.policy = policy
            self._wrapped = True
        else:
            # Subclass usage
            self._wrapped = False

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device):
        """Activate actions for the policy.

        This method is called at the beginning of training to set up the action space.
        External policies can override this to perform any necessary initialization.
        """
        self.device = device
        self.action_names = action_names
        self.action_max_params = action_max_params

        # For wrapped policies, check if they have this method
        if self._wrapped and hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)

        logger.info(f"PytorchPolicy actions activated with: {list(zip(action_names, action_max_params, strict=False))}")

    @property
    def lstm(self):
        """Return the LSTM module if the policy has one."""
        if self._wrapped:
            return getattr(self.policy, "lstm", None)
        return getattr(self, "_lstm", None)

    @property
    def total_params(self):
        """Return the total number of parameters in the policy."""
        return sum(p.numel() for p in self.parameters())

    @property
    def hidden_size(self):
        """Return the hidden size of the policy."""
        if self._wrapped:
            return getattr(self.policy, "hidden_size", 256)
        return getattr(self, "_hidden_size", 256)

    @property
    def core_num_layers(self):
        """Return the number of LSTM layers."""
        if self._wrapped:
            # Try to get from policy's lstm module
            if hasattr(self.policy, "lstm") and hasattr(self.policy.lstm, "num_layers"):
                return self.policy.lstm.num_layers
            return getattr(self.policy, "core_num_layers", 2)
        return getattr(self, "_core_num_layers", 2)

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the policy.

        Args:
            x: Input observation tensor
            state: Policy state containing LSTM hidden and cell states
            action: Optional action tensor for training mode

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
        """
        if self._wrapped:
            # Handle wrapped policies with different interfaces
            if hasattr(self.policy, "forward"):
                # Check if it's a policy that returns (action, logprob, entropy, value, logits)
                # This handles the old PytorchAgent interface
                result = self.policy(x, state, action)
                if len(result) == 5:
                    return result

                # Handle policies that return (logits, value)
                if len(result) == 2:
                    logits, value = result
                    if action is None:
                        # Inference mode
                        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)
                        # Assume action is already in the correct format
                        action = action_logit_index.unsqueeze(-1).repeat(1, 2)  # Dummy expansion
                    else:
                        # Training mode - need to evaluate given actions
                        # This is a simplified version - real implementation would need proper action conversion
                        action_log_prob, entropy, log_probs = evaluate_actions(logits, action[:, 0])

                    return action, action_log_prob, entropy, value, log_probs

        # Default implementation for subclasses
        raise NotImplementedError("Subclasses must implement the forward method")

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss.

        External policies can override this to add custom regularization.
        """
        if self._wrapped and hasattr(self.policy, "l2_reg_loss"):
            return self.policy.l2_reg_loss()
        return torch.tensor(0.0, device=self.device if hasattr(self, "device") else "cpu")

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss.

        External policies can override this to add custom initialization loss.
        """
        if self._wrapped and hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss()
        return torch.tensor(0.0, device=self.device if hasattr(self, "device") else "cpu")

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copy.

        External policies can override this if they use L2 initialization loss.
        """
        if self._wrapped and hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def clip_weights(self):
        """Clip weights.

        External policies can override this to implement custom weight clipping.
        """
        if self._wrapped and hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis.

        External policies can override this to provide custom weight metrics.
        """
        if self._wrapped and hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []


class SimplePytorchPolicy(PytorchPolicy):
    """Example implementation of a simple PyTorch policy.

    This shows how external policies can inherit from PytorchPolicy
    and implement the required methods.
    """

    def __init__(self, obs_shape: tuple, num_actions: int, hidden_size: int = 256):
        super().__init__()
        self._hidden_size = hidden_size
        self._core_num_layers = 2

        # Example architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_shape[0] * obs_shape[1] * obs_shape[2], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self._lstm = nn.LSTM(hidden_size, hidden_size, self._core_num_layers)

        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flatten observation
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # Extract features
        features = self.feature_extractor(x_flat)

        # LSTM forward
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_out, (h, c) = self._lstm(features.unsqueeze(0), (state.lstm_h, state.lstm_c))
            state.lstm_h = h
            state.lstm_c = c
        else:
            lstm_out, (h, c) = self._lstm(features.unsqueeze(0))
            state.lstm_h = h
            state.lstm_c = c

        lstm_out = lstm_out.squeeze(0)

        # Actor-critic heads
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)

        # Sample or evaluate actions
        if action is None:
            # Inference mode
            action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)
            # Convert to (action_type, action_param) format
            # This is a simplified version - real implementation would need proper conversion
            action = torch.stack([action_logit_index, torch.zeros_like(action_logit_index)], dim=-1)
        else:
            # Training mode
            # Convert action to logit index (simplified)
            action_logit_index = action[:, :, 0].flatten()
            action_log_prob, entropy, log_probs = evaluate_actions(logits, action_logit_index)

        return action, action_log_prob, entropy, value, log_probs
