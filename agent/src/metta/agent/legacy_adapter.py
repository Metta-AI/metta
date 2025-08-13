"""Adapter for backwards compatibility with old MettaAgent checkpoints."""

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

logger = logging.getLogger("legacy_adapter")


class LegacyMettaAgentAdapter(nn.Module):
    """Wraps old MettaAgent checkpoints to work with new architecture without circular references."""

    def __init__(self, legacy_agent):
        super().__init__()
        self.legacy_agent = legacy_agent

        # Copy attributes the new system expects
        for attr in [
            "components",
            "components_with_memory",
            "clip_range",
            "cum_action_max_params",
            "action_index_tensor",
        ]:
            if hasattr(legacy_agent, attr):
                setattr(self, attr, getattr(legacy_agent, attr))

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass - delegate to legacy agent's forward methods."""
        # Track if we need to reshape back
        needs_reshape = td.batch_dims > 1

        # Set up batch/bptt info like old MettaAgent did
        if needs_reshape:
            B = td.batch_size[0]
            TT = td.batch_size[1]
            td = td.reshape(B * TT)
            td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
        else:
            B = td.batch_size.numel()
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
            result = self.legacy_agent.forward_training(td, action)
            # Reshape back for training if needed
            if needs_reshape:
                B = td["batch"][0].item()
                T = td["bptt"][0].item()
                result = result.reshape(B, T)
            return result
        else:
            # Fallback - return td with minimal required fields
            # This path is for legacy agents that don't have the expected methods
            flat_batch = td.batch_size.numel()
            td.setdefault("actions", torch.zeros((flat_batch, 2), dtype=torch.long, device=td.device))
            td.setdefault("values", torch.zeros(flat_batch, device=td.device))
            return td

    def reset_memory(self) -> None:
        """Reset memory for components that have it."""
        for name in getattr(self, "components_with_memory", []):
            if name in self.components and hasattr(self.components[name], "reset_memory"):
                self.components[name].reset_memory()

    def get_memory(self) -> dict:
        """Get memory state from components."""
        return {
            name: self.components[name].get_memory()
            for name in getattr(self, "components_with_memory", [])
            if name in self.components and hasattr(self.components[name], "get_memory")
        }

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if getattr(self, "clip_range", 0) > 0 and hasattr(self.legacy_agent, "_apply_to_components"):
            self.legacy_agent._apply_to_components("clip_weights")

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss."""
        return getattr(self.legacy_agent, "l2_init_loss", lambda: torch.tensor(0.0))()

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies."""
        if hasattr(self.legacy_agent, "update_l2_init_weight_copy"):
            self.legacy_agent.update_l2_init_weight_copy()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics."""
        return getattr(self.legacy_agent, "compute_weight_metrics", lambda d: [])(delta)

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert action to logit index."""
        if hasattr(self.legacy_agent, "_convert_action_to_logit_index"):
            return self.legacy_agent._convert_action_to_logit_index(flattened_action)
        # Fallback using stored tensors
        act_type, act_param = flattened_action[:, 0].long(), flattened_action[:, 1].long()
        return act_type + self.cum_action_max_params[act_type] + act_param

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert logit index to action."""
        if hasattr(self.legacy_agent, "_convert_logit_index_to_action"):
            return self.legacy_agent._convert_logit_index_to_action(logit_indices)
        return self.action_index_tensor[logit_indices]

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings if the component exists."""
        if hasattr(self, "components") and "_action_embeds_" in self.components:
            self.components["_action_embeds_"].activate_actions(full_action_names, device)

    def _apply_feature_remapping(self, remap_tensor: torch.Tensor):
        """Apply feature remapping to observation component."""
        if hasattr(self, "components") and "_obs_" in self.components:
            obs = self.components["_obs_"]
            if hasattr(obs, "update_feature_remapping"):
                obs.update_feature_remapping(remap_tensor)

    def update_normalization_factors(self, features: dict[str, dict], original_feature_mapping: dict[str, int] | None):
        """Update normalization factors."""
        if hasattr(self.legacy_agent, "_update_normalization_factors"):
            self.legacy_agent._update_normalization_factors(features)

    @property
    def lstm(self):
        """Access to LSTM component if it exists."""
        if hasattr(self.legacy_agent, "lstm"):
            return self.legacy_agent.lstm
        if hasattr(self, "components") and "_core_" in self.components:
            core = self.components["_core_"]
            return getattr(core, "_net", None)
        return None
