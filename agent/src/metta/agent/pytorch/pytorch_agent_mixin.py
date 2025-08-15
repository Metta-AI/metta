"""
PyTorchAgentMixin: Shared functionality for all PyTorch-based agents.

This mixin provides all the common functionality that PyTorch agents need
to work properly with MettaAgent and the training pipeline. It ensures
consistency across all PyTorch implementations and makes it easy to
create new PyTorch agents that are fully compatible with the system.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.util.weights_analysis import analyze_weights

logger = logging.getLogger(__name__)


class PyTorchAgentMixin:
    """
    Mixin class providing shared functionality for PyTorch agents.

    This mixin should be used with classes that also inherit from LSTMWrapper
    or nn.Module and provides:
    - Configuration parameter handling
    - Weight clipping functionality
    - TensorDict field management
    - Action conversion utilities
    - Training/inference mode handling

    Usage:
        class MyAgent(PyTorchAgentMixin, LSTMWrapper):
            def __init__(self, env, **kwargs):
                # Extract mixin parameters
                mixin_params = self.extract_mixin_params(kwargs)
                super().__init__(env, ...)
                # Initialize mixin
                self.init_mixin(**mixin_params)
    """

    def extract_mixin_params(self, kwargs: dict) -> dict:
        """
        Extract parameters needed by the mixin from kwargs.

        Args:
            kwargs: Keyword arguments passed to the agent

        Returns:
            Dictionary with mixin-specific parameters
        """
        return {
            "clip_range": kwargs.pop("clip_range", 0),
            "analyze_weights_interval": kwargs.pop("analyze_weights_interval", 300),
            "extra_kwargs": kwargs,  # Remaining kwargs for logging
        }

    def init_mixin(
        self, clip_range: float = 0, analyze_weights_interval: int = 300, extra_kwargs: Optional[dict] = None
    ):
        """
        Initialize mixin parameters.

        Args:
            clip_range: Weight clipping range (0 = disabled)
            analyze_weights_interval: Interval for weight analysis
            extra_kwargs: Additional configuration parameters for logging
        """
        self.clip_range = clip_range
        self.analyze_weights_interval = analyze_weights_interval
        self.l2_init_scale = 1  # Default L2-init scale
        self.clip_scale = 1  # Default clip scale

        # Store initial weights for L2-init regularization
        self._store_initial_weights()

        # Note: action_index_tensor and cum_action_max_params are set by
        # MettaAgent.activate_actions() directly on the policy object

    def clip_weights(self):
        """
        Clip weights to prevent large updates during training.

        This matches ComponentPolicy's weight clipping behavior and is called
        by the trainer after each optimizer step when clip_range > 0.
        """
        if self.clip_range > 0:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, "weight") and module.weight is not None:
                        module.weight.data.clamp_(-self.clip_range, self.clip_range)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data.clamp_(-self.clip_range, self.clip_range)

    def set_tensordict_fields(self, td: TensorDict, observations: torch.Tensor):
        """
        Set critical TensorDict fields required by ComponentPolicy.

        These fields (bptt and batch) are essential for:
        - LSTM state management
        - Experience buffer integration
        - Proper tensor reshaping during training

        NOTE: The caller must reshape the TD BEFORE calling this if needed.
        The fields will be set to match the flattened batch dimension.

        Args:
            td: TensorDict to update (should be reshaped if needed)
            observations: Observation tensor to determine dimensions

        Returns:
            Tuple of (B, TT) dimensions
        """
        if observations.dim() == 4:  # Training: [B, T, obs_tokens, 3]
            B = observations.shape[0]
            TT = observations.shape[1]
            # Fields should match the flattened batch size
            total_batch = B * TT
            td.set("bptt", torch.full((total_batch,), TT, device=observations.device, dtype=torch.long))
            td.set("batch", torch.full((total_batch,), B, device=observations.device, dtype=torch.long))
        else:  # Inference: [B, obs_tokens, 3]
            B = observations.shape[0]
            TT = 1
            # Set fields for inference mode
            td.set("bptt", torch.full((B,), 1, device=observations.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=observations.device, dtype=torch.long))

        return B, TT

    def handle_training_mode(
        self, td: TensorDict, action: torch.Tensor, logits_list: torch.Tensor, value: torch.Tensor
    ) -> TensorDict:
        """
        Handle training mode processing with proper TD reshaping.

        Args:
            td: TensorDict to update
            action: Action tensor from training data
            logits_list: Logits from policy
            value: Value estimates from critic

        Returns:
            Updated TensorDict with proper shape
        """
        # CRITICAL: ComponentPolicy expects the action to be flattened already during training
        # The TD should be reshaped to match the flattened batch dimension
        if action.dim() == 3:  # (B, T, A) -> (BT, A)
            batch_size_orig, time_steps, A = action.shape
            action = action.view(batch_size_orig * time_steps, A)
            # Also flatten the TD to match
            if td.batch_dims > 1:
                td = td.reshape(td.batch_size.numel())

        action_log_probs = F.log_softmax(logits_list, dim=-1)
        action_probs = torch.exp(action_log_probs)

        action_logit_index = self._convert_action_to_logit_index(action)
        batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
        selected_log_probs = action_log_probs[batch_indices, action_logit_index]

        entropy = -(action_probs * action_log_probs).sum(dim=-1)

        # Store in flattened TD (will be reshaped by caller if needed)
        td["act_log_prob"] = selected_log_probs
        td["entropy"] = entropy
        td["full_log_probs"] = action_log_probs
        td["value"] = value

        # ComponentPolicy reshapes the TD after training forward based on td["batch"] and td["bptt"]
        # The reshaping happens in ComponentPolicy.forward() after forward_training()
        if "batch" in td.keys() and "bptt" in td.keys():
            batch_size = td["batch"][0].item()
            bptt_size = td["bptt"][0].item()
            td = td.reshape(batch_size, bptt_size)

        return td

    def handle_inference_mode(self, td: TensorDict, logits_list: torch.Tensor, value: torch.Tensor) -> TensorDict:
        """
        Handle inference mode processing with action sampling.

        Args:
            td: TensorDict to update
            logits_list: Logits from policy
            value: Value estimates from critic

        Returns:
            Updated TensorDict with sampled actions
        """
        log_probs = F.log_softmax(logits_list, dim=-1)
        action_probs = torch.exp(log_probs)

        actions = torch.multinomial(action_probs, num_samples=1).view(-1)
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        selected_log_probs = log_probs[batch_indices, actions]

        action = self._convert_logit_index_to_action(actions)

        td["actions"] = action.to(dtype=torch.int32)
        td["act_log_prob"] = selected_log_probs
        td["values"] = value.flatten()
        td["full_log_probs"] = log_probs

        return td

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs.

        NOTE: This overrides MettaAgent's implementation to use the policy's
        action_index_tensor which is set by MettaAgent.activate_actions().

        Args:
            action_logit_index: Indices into flattened action space

        Returns:
            Action tensor with (action_type, action_param) pairs
        """
        # Use the action_index_tensor that MettaAgent sets on the policy
        if not hasattr(self, "action_index_tensor") or self.action_index_tensor is None:
            raise RuntimeError("action_index_tensor not set. MettaAgent.activate_actions() must be called first.")
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete indices.

        For MultiDiscrete action spaces, actions are represented as pairs:
        - action_type: which action type (e.g., move, attack, etc.)
        - action_param: parameter for that action type

        Args:
            flattened_action: Actions as (action_type, action_param) pairs

        Returns:
            Indices into flattened action space
        """
        # Use the cum_action_max_params that MettaAgent sets on the policy
        if not hasattr(self, "cum_action_max_params") or self.cum_action_max_params is None:
            raise RuntimeError("cum_action_max_params not set. MettaAgent.activate_actions() must be called first.")

        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]

        # Formula for MultiDiscrete action space conversion
        return action_type_numbers + cumulative_sum + action_params

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """
        Activate action embeddings to match ComponentPolicy interface.

        This is called by MettaAgent.activate_actions() and should be
        overridden by agents that have action embeddings.

        Args:
            full_action_names: List of action names
            device: Device for tensors
        """
        # Pass through to the policy if it exists
        if hasattr(self, "policy") and hasattr(self.policy, "activate_action_embeddings"):
            self.policy.activate_action_embeddings(full_action_names, device)

    def _store_initial_weights(self):
        """Store initial weights for L2-init regularization."""
        self.initial_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
                if hasattr(module, "weight"):
                    # Store with full path for uniqueness
                    self.initial_weights[name if name else "root"] = module.weight.data.clone()

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss for regularization."""
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        if hasattr(self, "initial_weights"):
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
                    if hasattr(module, "weight"):
                        weight_name = name if name else "root"
                        if weight_name in self.initial_weights:
                            weight_diff = module.weight - self.initial_weights[weight_name].to(module.weight.device)
                            total_loss = total_loss.to(module.weight.device)
                            total_loss += torch.sum(weight_diff**2) * self.l2_init_scale
        return total_loss

    def update_l2_init_weight_copy(self):
        """Update the stored initial weights for L2 regularization."""
        self._store_initial_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for monitoring."""
        metrics_list = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.data.dim() == 2:  # Only analyze 2D matrices
                    metrics = analyze_weights(module.weight.data, delta)
                    metrics["name"] = name if name else "root"
                    metrics_list.append(metrics)
        return metrics_list
