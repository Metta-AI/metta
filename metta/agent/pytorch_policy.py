import logging
from typing import Optional

import torch
from pufferlib.pytorch import sample_logits
from torch import nn

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

logger = logging.getLogger("pytorch_policy")


class PytorchPolicy(nn.Module):
    """Adapter class for external PyTorch policies to integrate with MettaAgent.

    This class wraps external policies that follow different interfaces and
    translates them to the expected MettaAgent interface. It's particularly
    useful for legacy policies that return (hidden, critic) tuples.

    The wrapped policy should be passed to the constructor and will be
    accessible via self.policy.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        # For legacy policies, ensure we expose their hidden_size
        self._policy_hidden_size = getattr(policy, "hidden_size", 256)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device):
        """Activate actions for the policy.

        This method is called at the beginning of training to set up the action space.
        """
        self.device = device
        self.action_names = action_names
        self.action_max_params = action_max_params

        # Forward to wrapped policy if it has this method
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)

        logger.info(f"PytorchPolicy actions activated with: {list(zip(action_names, action_max_params, strict=False))}")

    @property
    def lstm(self):
        """Return the LSTM module if the policy has one."""
        return getattr(self.policy, "lstm", None)

    @property
    def total_params(self):
        """Return the total number of parameters in the policy."""
        return sum(p.numel() for p in self.parameters())

    @property
    def hidden_size(self):
        """Return the hidden size of the policy."""
        return self._policy_hidden_size

    @hidden_size.setter
    def hidden_size(self, value):
        """Set the hidden size (for backward compatibility)."""
        self._policy_hidden_size = value

    @property
    def core_num_layers(self):
        """Return the number of LSTM layers."""
        # Try to get from policy's lstm module
        if hasattr(self.policy, "lstm") and hasattr(self.policy.lstm, "num_layers"):
            return self.policy.lstm.num_layers
        return getattr(self.policy, "core_num_layers", 2)

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
        # Get the raw output from the wrapped policy
        result = self.policy(x, state, action)

        # Check if it's a legacy policy that returns (hidden, critic)
        if len(result) == 2 and result[0].dim() >= 2:
            # Legacy policy interface: (hidden, critic) -> (action, logprob, entropy, value, logits)
            hidden, critic = result

            # Use pufferlib's sample_logits to handle action sampling
            action_result, logprob, logits_entropy = sample_logits(hidden, action)

            # Return in MettaAgent format
            # hidden -> log_probs (the raw logits)
            # critic -> value
            return action_result, logprob, logits_entropy, critic, hidden

        # Check if it's a policy that returns the full 5-tuple
        elif len(result) == 5:
            return result

        # Handle other policies that might return (logits, value)
        elif len(result) == 2:
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

        else:
            raise ValueError(f"Wrapped policy returned unexpected number of values: {len(result)}")

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss."""
        if hasattr(self.policy, "l2_reg_loss"):
            return self.policy.l2_reg_loss()
        return torch.tensor(0.0, device=self.device if hasattr(self, "device") else "cpu")

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss."""
        if hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss()
        return torch.tensor(0.0, device=self.device if hasattr(self, "device") else "cpu")

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copy."""
        if hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def clip_weights(self):
        """Clip weights."""
        if hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []
