import logging
from typing import Optional

import pufferlib.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.util.weights_analysis import analyze_weights

logger = logging.getLogger(__name__)


class PyTorchAgentMixin:
    """Mixin class providing shared functionality for PyTorch agents.

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
                self.init_mixin(**mixin_params)"""

    @staticmethod
    def _is_regularizable_layer(module):
        """Check if a module is a layer type that should have weight regularization."""
        return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d))

    def extract_mixin_params(self, kwargs: dict) -> dict:
        """Extract parameters needed by the mixin from kwargs."""
        return {
            "clip_range": kwargs.pop("clip_range", 0),
            "analyze_weights_interval": kwargs.pop("analyze_weights_interval", 300),
            "extra_kwargs": kwargs,  # Remaining kwargs for logging
        }

    def init_mixin(
        self, clip_range: float = 0, analyze_weights_interval: int = 300, extra_kwargs: Optional[dict] = None
    ):
        """Initialize mixin parameters."""
        self.clip_range = clip_range
        self.analyze_weights_interval = analyze_weights_interval
        self.l2_init_scale = 1  # Default L2-init scale
        self.clip_scale = 1  # Default clip scale

        # Store initial weights for L2-init regularization
        self._store_initial_weights()

        # Note: action_index_tensor and cum_action_max_params are set by
        # MettaAgent.initialize_to_environment() directly on the policy object

    def create_action_heads(self, env, input_size: int = 512 + 16):
        """Create action heads based on the environment's action configuration.

        For multi-discrete action spaces, we create a single head with the total
        flattened action space (sum of all action parameter ranges)."""
        # Calculate total flattened action space from environment
        total_actions = sum(max_arg + 1 for max_arg in env.max_action_args)

        # Create a single head for the flattened action space
        return nn.ModuleList([pufferlib.pytorch.layer_init(nn.Linear(input_size, total_actions), std=0.01)])

    def clip_weights(self):
        """Clip weights to prevent large updates during training.

        This matches ComponentPolicy's weight clipping behavior and is called
        by the trainer after each optimizer step when clip_range > 0."""
        if self.clip_range > 0:
            for module in self.modules():
                # Note: ConvTranspose2d is not included here to match original clipping behavior
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    module.weight.data.clamp_(-self.clip_range, self.clip_range)
                    if module.bias is not None:
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

    def forward_training(
        self, td: TensorDict, action: torch.Tensor, logits_list: torch.Tensor, value: torch.Tensor
    ) -> TensorDict:
        """Forward pass for training mode with proper TD reshaping."""
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

    def forward_inference(self, td: TensorDict, logits_list: torch.Tensor, value: torch.Tensor) -> TensorDict:
        """Forward pass for inference mode with action sampling."""
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
        action_index_tensor which is set by MettaAgent.initialize_to_environment().
        """
        # Use the action_index_tensor that MettaAgent sets on the policy
        if not hasattr(self, "action_index_tensor") or self.action_index_tensor is None:
            raise RuntimeError(
                "action_index_tensor not set. MettaAgent.initialize_to_environment() must be called first."
            )
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete indices.

        For MultiDiscrete action spaces, actions are represented as pairs:
        - action_type: which action type (e.g., move, attack, etc.)
        - action_param: parameter for that action type
        """
        # Use the cum_action_max_params that MettaAgent sets on the policy
        if not hasattr(self, "cum_action_max_params") or self.cum_action_max_params is None:
            raise RuntimeError(
                "cum_action_max_params not set. MettaAgent.initialize_to_environment() must be called first."
            )

        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]

        # Formula for MultiDiscrete action space conversion
        return action_type_numbers + cumulative_sum + action_params

    def initialize_to_environment(self, full_action_names: list[str], device):
        """
        Initialize to environment to match ComponentPolicy interface.

        This is called by MettaAgent.initialize_to_environment() and provides
        a safe no-op implementation for vanilla PyTorch agents. Agents with
        components that need environment initialization should override this method.
        """
        # Pass through to nested policy if it exists and has the method
        if hasattr(self, "policy") and hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(full_action_names, device)
        # Otherwise this is a no-op, which is safe for vanilla PyTorch agents

    def _store_initial_weights(self):
        """Store initial weights for L2-init regularization."""
        self.initial_weights = {}
        for name, module in self.named_modules():
            if self._is_regularizable_layer(module):
                if hasattr(module, "weight"):
                    # Store with full path for uniqueness
                    self.initial_weights[name if name else "root"] = module.weight.data.clone()

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss for regularization."""
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        if hasattr(self, "initial_weights"):
            for name, module in self.named_modules():
                if self._is_regularizable_layer(module):
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

    def _apply_feature_remapping(self, remap_tensor: torch.Tensor):
        """Apply feature remapping for compatibility with ComponentPolicy interface.

        PyTorch vanilla models don't need feature remapping since they work with
        fixed feature spaces, so this is a no-op.
        """
        pass

    def update_normalization_factors(self, features: dict[str, dict], original_feature_mapping: dict[str, int] | None):
        """Update normalization factors for compatibility with ComponentPolicy interface.

        PyTorch vanilla models don't dynamically update normalizations, so this is a no-op.
        """
        pass
