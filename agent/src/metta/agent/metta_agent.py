import logging
from typing import TYPE_CHECKING, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.instantiate import instantiate
from metta.agent.policy_base import PolicyBase


from typing import Dict





if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class MettaAgent(nn.Module):
    """Clean and simplified MettaAgent implementation."""

    def __init__(self, components: nn.ModuleDict, config: DictConfig, policy: Optional[PolicyBase] = None, device = 'cpu'):
        super().__init__()
        self.components = components
        self.config = config
        self.device = device

        # Extract key configuration values
        self.hidden_size = self.config.components._core_.output_size
        self.core_num_layers = self.config.components._core_.nn_params.num_layers
        self.clip_range = self.config.clip_range

        # Set default policy if none provided
        if policy is None:
            policy = DefaultPolicy()
        self.set_policy(policy)

        # Move components to device and calculate parameters
        self.components = self.components.to(self.device)
        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def set_policy(self, policy: PolicyBase):
        """Set or change the agent's policy."""
        if not isinstance(policy, PolicyBase):
            raise TypeError("Policy must inherit from PolicyBase")
        self.policy = policy
        self.policy.agent = self

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[PolicyState] = None, action: Optional[torch.Tensor] = None) -> Tuple:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set. Use set_policy() first.")
        return self.policy.forward(self, obs, state, action)

    def initialize_to_environment(self, features: dict[str, dict], action_names: list[str], action_max_params: list[int], device, is_training: bool = True):
        """Initialize the agent to the current environment."""
        self._initialize_observations(features, device, self.training)
        self.activate_actions(action_names, action_max_params, device)

    def _initialize_observations(self, features: dict[str, dict], device, is_training: bool):
        """Initialize observation features and handle feature remapping."""
        self.active_features = features
        self.device = device
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props}

        # Handle feature mapping for model reuse
        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            logger.info(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
        else:
            self._create_feature_remapping(features, is_training)

    def _create_feature_remapping(self, features: dict[str, dict], is_training: bool):
        """Create remapping for feature IDs when environment changes."""
        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}
        unknown_features = []

        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not is_training:
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID
                unknown_features.append(name)
            else:
                self.original_feature_mapping[name] = new_id

        if self.feature_id_remap:
            logger.info(f"Created feature remapping: {len(self.feature_id_remap)} remapped, {len(unknown_features)} unknown")
            self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to components."""
        if "_obs_" in self.components and hasattr(self.components["_obs_"], "update_feature_remapping"):
            remap_tensor = torch.arange(256, dtype=torch.uint8, device=self.device)

            for new_id, original_id in self.feature_id_remap.items():
                remap_tensor[new_id] = original_id

            current_feature_ids = {props["id"] for props in features.values()}
            for feature_id in range(256):
                if feature_id not in self.feature_id_remap and feature_id not in current_feature_ids:
                    remap_tensor[feature_id] = unknown_id

            self.components["_obs_"].update_feature_remapping(remap_tensor)

        self._update_normalization_factors(features)

    def _update_normalization_factors(self, features: dict[str, dict]):
        """Update normalization factors for components."""
        for comp_name, component in self.components.items():
            if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
                norm_tensor = torch.ones(256, dtype=torch.float32)
                for name, props in features.items():
                    if name in self.original_feature_mapping and "normalization" in props:
                        original_id = self.original_feature_mapping[name]
                        norm_tensor[original_id] = props["normalization"]
                component.register_buffer("_norm_factors", norm_tensor)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Initialize action space for the agent."""
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for efficient action conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        # Build full action names and activate embeddings
        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        if "_action_embeds_" in self.components:
            self.components["_action_embeds_"].activate_actions(full_action_names, self.device)

        # Create action index tensor for conversions
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)
        logger.info(f"Actions initialized: {self.active_actions}")


    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for saving."""
        return getattr(self, "original_feature_mapping", None)

    def restore_original_feature_mapping(self, mapping: dict[str, int]) -> None:
        """Restore feature mapping from saved state."""
        self.original_feature_mapping = mapping.copy()
        logger.info(f"Restored feature mapping with {len(mapping)} features")

    @property
    def lstm(self):
        """Access to LSTM component."""
        if "_core_" in self.components and hasattr(self.components["_core_"], "_net"):
            return self.components["_core_"]._net
        return None

    @property
    def total_params(self):
        """Total number of parameters."""
        return getattr(self, '_total_params', sum(p.numel() for p in self.parameters() if p.requires_grad))


    def _apply_to_components(self, method_name, *args, **kwargs) -> list[torch.Tensor]:
        """Apply a method to all components that have it."""
        results = []
        for name, component in self.components.items():
            if hasattr(component, method_name):
                method = getattr(component, method_name)
                if callable(method):
                    result = method(*args, **kwargs)
                    if result is not None:
                        results.append(result)
        return results

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss."""
        losses = self._apply_to_components("l2_init_loss")
        return torch.sum(torch.stack(losses)) if losses else torch.tensor(0.0, device=self.device)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies."""
        self._apply_to_components("update_l2_init_weight_copy")

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        results = {}
        for name, component in self.components.items():
            if hasattr(component, "compute_weight_metrics"):
                result = component.compute_weight_metrics(delta)
                if result is not None:
                    results[name] = result
        return list(results.values())




class DistributedMettaAgent(DistributedDataParallel):
    """Distributed wrapper for MettaAgent."""

    def __init__(self, agent, device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")
        agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        return self.module.activate_actions(action_names, action_max_params, device)

    def initialize_to_environment(self, features: dict[str, dict], action_names: list[str],
                                action_max_params: list[int], device: torch.device, is_training: bool = True) -> None:
        return self.module.initialize_to_environment(features, action_names, action_max_params, device)





PolicyAgent = MettaAgent | DistributedMettaAgent
