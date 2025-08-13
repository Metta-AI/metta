"""Adapter for backwards compatibility with old MettaAgent checkpoints."""

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

logger = logging.getLogger("legacy_adapter")


class LegacyMettaAgentAdapter(nn.Module):
    """Wraps old MettaAgent checkpoints to work with new architecture without circular references."""

    def __init__(self, legacy_agent):
        super().__init__()

        # Break any circular references in the legacy agent before storing it
        self._break_circular_references(legacy_agent)

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

    def _break_circular_references(self, agent):
        """Break circular references that could cause infinite recursion."""
        # Remove self-references that might exist in old checkpoints
        if hasattr(agent, "policy") and agent.policy is agent:
            agent.policy = None
            logger.info("Broke circular reference: agent.policy = agent")

        # Remove any module references that point back to the agent itself
        if hasattr(agent, "_modules"):
            for name, module in list(agent._modules.items()):
                if module is agent:
                    agent._modules[name] = None
                    logger.info(f"Broke circular reference in _modules[{name}]")

        # Check components for circular references
        if hasattr(agent, "components"):
            if isinstance(agent.components, nn.ModuleDict):
                for name in list(agent.components.keys()):
                    if agent.components[name] is agent:
                        del agent.components[name]
                        logger.info(f"Removed circular reference in components[{name}]")
                    # Also check for deep circular references within components
                    elif hasattr(agent.components[name], "_modules"):
                        self._break_component_circular_refs(agent.components[name], agent)

        # Remove any parent references that might cause cycles
        if hasattr(agent, "_parent"):
            agent._parent = None

        # Clean up any other potential circular references in __dict__
        for key, value in list(agent.__dict__.items()):
            if value is agent and key not in ["_modules", "_parameters", "_buffers"]:
                agent.__dict__[key] = None
                logger.info(f"Broke circular reference: agent.{key} = agent")

    def _break_component_circular_refs(self, component, root_agent):
        """Recursively break circular references within a component."""
        if hasattr(component, "_modules"):
            for name, module in list(component._modules.items()):
                if module is root_agent or module is component:
                    component._modules[name] = None
                    logger.info("Broke deep circular reference in component")

        # Check component's __dict__ for references back to root
        if hasattr(component, "__dict__"):
            for key, value in list(component.__dict__.items()):
                if value is root_agent and key not in ["_modules", "_parameters", "_buffers"]:
                    component.__dict__[key] = None
                    logger.info(f"Broke deep circular reference in component.{key}")

    def named_children(self):
        """Override to prevent infinite recursion when traversing module tree.

        This is called by SyncBatchNorm.convert_sync_batchnorm and can cause
        infinite recursion if there are circular references.
        """
        # Only return components that are actual nn.Modules and not self-referential
        seen = set()

        # First yield components if they exist
        if hasattr(self, "components") and isinstance(self.components, nn.ModuleDict):
            for name, module in self.components.items():
                if module is not None and id(module) not in seen and module is not self:
                    seen.add(id(module))
                    yield f"components.{name}", module

        # Don't yield legacy_agent as a child to avoid potential circular traversal
        # The legacy agent's components are already accessible via self.components

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

        # Check if legacy agent has the forward methods (unlikely for real old checkpoints)
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

        # Implement the forward logic for old checkpoints that don't have these methods
        if action is None:
            # Inference mode - sample actions
            return self._forward_inference(td)
        else:
            # Training mode - evaluate actions
            result = self._forward_training(td, action)
            if needs_reshape:
                B = td["batch"][0].item()
                T = td["bptt"][0].item()
                result = result.reshape(B, T)
            return result

    def _forward_inference(self, td: TensorDict) -> TensorDict:
        """Inference mode - sample actions from the policy."""
        # Get value and action logits from components
        value = td.get("_value_")
        logits = td.get("_action_")

        if value is None or logits is None:
            # Fallback if components didn't produce expected outputs
            flat_batch = td.batch_size.numel()
            td["actions"] = torch.zeros((flat_batch, 2), dtype=torch.long, device=td.device)
            td["values"] = torch.zeros(flat_batch, device=td.device)
            return td

        # Sample actions from logits
        action_logit_index, action_log_prob, _, full_log_probs = sample_actions(logits)

        # Convert logit indices to actions
        if hasattr(self, "action_index_tensor") and self.action_index_tensor is not None:
            action = self.action_index_tensor[action_logit_index]
        else:
            # Fallback if action conversion not available
            action = torch.zeros((td.batch_size.numel(), 2), dtype=torch.long, device=td.device)

        # Store outputs in td
        td["actions"] = action
        td["act_log_prob"] = action_log_prob
        td["values"] = value.flatten()
        td["full_log_probs"] = full_log_probs

        return td

    def _forward_training(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        """Training mode - evaluate given actions."""
        # Get value and action logits from components
        value = td.get("_value_")
        logits = td.get("_action_")

        if value is None or logits is None:
            # Fallback if components didn't produce expected outputs
            logger.warning("Components did not produce _value_ or _action_ outputs")
            flat_batch = td.batch_size.numel()
            td["act_log_prob"] = torch.zeros(flat_batch, device=td.device)
            td["value"] = torch.zeros((flat_batch, 1), device=td.device)
            td["entropy"] = torch.zeros(flat_batch, device=td.device)
            return td

        # Handle action reshaping
        if action.dim() == 3:  # [B, T, 2]
            B, T, A = action.shape
            flattened_action = action.view(B * T, A)
        else:  # Already flattened
            flattened_action = action

        # Convert actions to logit indices
        if hasattr(self, "cum_action_max_params") and self.cum_action_max_params is not None:
            action_type_numbers = flattened_action[:, 0].long()
            action_params = flattened_action[:, 1].long()
            action_logit_index = action_type_numbers + self.cum_action_max_params[action_type_numbers] + action_params
        else:
            # Fallback - assume simple action space
            action_logit_index = flattened_action[:, 0]

        # Evaluate actions to get log probs and entropy
        action_log_prob, entropy, full_log_probs = evaluate_actions(logits, action_logit_index)

        # Store outputs in td
        td["act_log_prob"] = action_log_prob
        td["entropy"] = entropy
        td["value"] = value
        td["full_log_probs"] = full_log_probs

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
