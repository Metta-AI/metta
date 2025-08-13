"""
Adapter for backwards compatibility with old MettaAgent checkpoints.

Old checkpoints have MettaAgent with self.components directly, while new architecture
has MettaAgent.policy (ComponentPolicy) with policy.components. This adapter wraps
old MettaAgent instances to make them compatible with the new interface.
"""

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

logger = logging.getLogger("legacy_adapter")


class LegacyMettaAgentAdapter(nn.Module):
    """Adapter that wraps old MettaAgent checkpoints to work with new architecture.

    This adapter solves the backwards compatibility problem without circular references.
    Old checkpoints where MettaAgent directly contained components are wrapped by this
    adapter, which provides the same interface as ComponentPolicy.
    """

    def __init__(self, legacy_agent):
        """Initialize the adapter with a legacy MettaAgent instance.

        Args:
            legacy_agent: An old MettaAgent instance with self.components
        """
        super().__init__()
        self.legacy_agent = legacy_agent

        # Copy over attributes that the new system expects
        if hasattr(legacy_agent, "components"):
            self.components = legacy_agent.components
        if hasattr(legacy_agent, "components_with_memory"):
            self.components_with_memory = legacy_agent.components_with_memory
        if hasattr(legacy_agent, "clip_range"):
            self.clip_range = legacy_agent.clip_range

        # Copy action conversion attributes if they exist
        for attr in ["cum_action_max_params", "action_index_tensor"]:
            if hasattr(legacy_agent, attr):
                setattr(self, attr, getattr(legacy_agent, attr))

        logger.info("Created LegacyMettaAgentAdapter for backwards compatibility")

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass - delegate to legacy agent's forward methods."""
        # Set up batch/bptt info like old MettaAgent did
        B = td.batch_size[0] if td.batch_dims > 1 else td.batch_size.numel()
        if td.batch_dims > 1:
            TT = td.batch_size[1]
            td = td.reshape(B * TT)
            td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
        else:
            td.set("bptt", torch.ones(B, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))

        # Run value/action components if they exist
        if hasattr(self, "components"):
            if "_value_" in self.components:
                self.components["_value_"](td)
            if "_action_" in self.components:
                self.components["_action_"](td)

        # Delegate to appropriate legacy method
        if action is None and hasattr(self.legacy_agent, "forward_inference"):
            return self.legacy_agent.forward_inference(td)
        elif action is not None and hasattr(self.legacy_agent, "forward_training"):
            output_td = self.legacy_agent.forward_training(td, action)
            # Reshape back for training
            batch_size = td["batch"][0].item()
            bptt_size = td["bptt"][0].item()
            return output_td.reshape(batch_size, bptt_size)
        else:
            # Fallback - return td with minimal required fields
            td.setdefault("actions", torch.zeros((B, 2), dtype=torch.long, device=td.device))
            td.setdefault("values", torch.zeros(B, device=td.device))
            return td

    def reset_memory(self) -> None:
        """Reset memory for components that have it."""
        if hasattr(self, "components_with_memory"):
            for name in self.components_with_memory:
                if name in self.components and hasattr(self.components[name], "reset_memory"):
                    self.components[name].reset_memory()

    def get_memory(self) -> dict:
        """Get memory state from components."""
        if hasattr(self, "components_with_memory"):
            return {
                name: self.components[name].get_memory()
                for name in self.components_with_memory
                if name in self.components and hasattr(self.components[name], "get_memory")
            }
        return {}

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if hasattr(self, "clip_range") and self.clip_range > 0:
            if hasattr(self.legacy_agent, "_apply_to_components"):
                self.legacy_agent._apply_to_components("clip_weights")

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss."""
        if hasattr(self.legacy_agent, "l2_init_loss"):
            return self.legacy_agent.l2_init_loss()
        return torch.tensor(0.0, dtype=torch.float32)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies."""
        if hasattr(self.legacy_agent, "update_l2_init_weight_copy"):
            self.legacy_agent.update_l2_init_weight_copy()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics."""
        if hasattr(self.legacy_agent, "compute_weight_metrics"):
            return self.legacy_agent.compute_weight_metrics(delta)
        return []

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert action to logit index."""
        if hasattr(self.legacy_agent, "_convert_action_to_logit_index"):
            return self.legacy_agent._convert_action_to_logit_index(flattened_action)
        # Fallback implementation using stored tensors
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return action_type_numbers + cumulative_sum + action_params

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert logit index to action."""
        if hasattr(self.legacy_agent, "_convert_logit_index_to_action"):
            return self.legacy_agent._convert_logit_index_to_action(logit_indices)
        # Fallback using action_index_tensor
        return self.action_index_tensor[logit_indices]

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings if the component exists."""
        if hasattr(self, "components") and "_action_embeds_" in self.components:
            self.components["_action_embeds_"].activate_actions(full_action_names, device)

    def _apply_feature_remapping(self, remap_tensor: torch.Tensor):
        """Apply feature remapping to observation component."""
        if hasattr(self, "components") and "_obs_" in self.components:
            obs_component = self.components["_obs_"]
            if hasattr(obs_component, "update_feature_remapping"):
                obs_component.update_feature_remapping(remap_tensor)

    def update_normalization_factors(self, features: dict[str, dict], original_feature_mapping: dict[str, int] | None):
        """Update normalization factors - delegate to legacy agent if it has the method."""
        if hasattr(self.legacy_agent, "_update_normalization_factors"):
            # Call the private method on legacy agent
            self.legacy_agent._update_normalization_factors(features)

    @property
    def lstm(self):
        """Access to LSTM component if it exists."""
        if hasattr(self.legacy_agent, "lstm"):
            return self.legacy_agent.lstm
        if hasattr(self, "components") and "_core_" in self.components:
            core = self.components["_core_"]
            if hasattr(core, "_net"):
                return core._net
        return None
