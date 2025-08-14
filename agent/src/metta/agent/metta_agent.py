import logging
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.component_policy import ComponentPolicy
from metta.agent.pytorch.agent_mapper import agent_classes
from metta.rl.system_config import SystemConfig

logger = logging.getLogger("metta_agent")


class DistributedMettaAgent(DistributedDataParallel):
    """
    Because this class passes through __getattr__ to its self.module, it implements everything
    MettaAgent does. We only have a need for this class because using the DistributedDataParallel wrapper
    returns an object of almost the same interface: you need to call .module to get the wrapped agent.
    """

    module: "MettaAgent"

    def __init__(self, agent: "MettaAgent", device: torch.device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")

        # Check if the agent might have circular references that would cause recursion
        # This can happen with legacy checkpoints wrapped in LegacyMettaAgentAdapter
        try:
            # Try to convert - this will fail with RecursionError if there are circular refs
            layers_converted_agent: "MettaAgent" = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)  # type: ignore
        except RecursionError:
            logger.warning(
                "RecursionError during SyncBatchNorm conversion - likely due to circular references. "
                "Skipping SyncBatchNorm conversion."
            )
            layers_converted_agent = agent

        # Pass device_ids for GPU, but not for CPU
        if device.type == "cpu":
            super().__init__(module=layers_converted_agent)
        else:
            super().__init__(module=layers_converted_agent, device_ids=[device], output_device=device)

    def __getattr__(self, name: str) -> Any:
        # First try DistributedDataParallel's __getattr__, then self.module's (MettaAgent's)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MettaAgent(nn.Module):
    def __init__(
        self,
        env,
        system_cfg: SystemConfig,
        agent_cfg: DictConfig,
        policy: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = agent_cfg
        self.device = system_cfg.device

        # Create observation space
        self.obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.int32),
            }
        )

        self.obs_width = env.obs_width
        self.obs_height = env.obs_height
        self.action_space = env.single_action_space
        self.feature_normalizations = env.feature_normalizations

        # Create policy if not provided
        if policy is None:
            policy = self._create_policy(agent_cfg, env, system_cfg)

        self.policy = policy
        if self.policy is not None:
            # Move policy to device
            self.policy = self.policy.to(self.device)
            # Also set device attribute if it exists
            if hasattr(self.policy, "device"):
                self.policy.device = self.device

        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def _create_policy(self, agent_cfg: DictConfig, env, system_cfg: SystemConfig) -> nn.Module:
        """Create the appropriate policy based on configuration."""
        if agent_cfg.get("agent_type") in agent_classes:
            # Create PyTorch policy
            AgentClass = agent_classes[agent_cfg.agent_type]
            policy = AgentClass(env=env)
            logger.info(f"Using PyTorch Policy: {policy} (type: {agent_cfg.agent_type})")
        else:
            # Create ComponentPolicy (YAML config)
            policy = ComponentPolicy(
                obs_space=self.obs_space,
                obs_width=self.obs_width,
                obs_height=self.obs_height,
                action_space=self.action_space,
                feature_normalizations=self.feature_normalizations,
                device=system_cfg.device,
                cfg=agent_cfg,
            )
            logger.info(f"Using ComponentPolicy: {type(policy).__name__}")

        return policy

    def forward(self, td: Dict[str, torch.Tensor], state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set during initialization.")

        # Delegate to policy - it handles all cases including legacy
        return self.policy(td, state, action)

    def reset_memory(self) -> None:
        """Reset memory - delegates to policy if it supports memory."""
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def get_memory(self) -> dict:
        """Get memory state - delegates to policy if it supports memory."""
        return getattr(self.policy, "get_memory", lambda: {})()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """Initialize the agent to the current environment."""
        # MettaAgent handles all initialization
        self.activate_actions(action_names, action_max_params, device)
        self.activate_observations(features, device)

    def activate_observations(self, features: dict[str, dict], device):
        """Activate observation features by storing the feature mapping."""
        self.active_features = features
        self.device = device

        # Create quick lookup mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        # Store original feature mapping on first initialization
        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            logger.info(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
        else:
            # Create remapping for subsequent initializations
            self._create_feature_remapping(features)

    def _create_feature_remapping(self, features: dict[str, dict]):
        """Create a remapping dictionary to translate new feature IDs to original ones."""
        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}
        unknown_features = []

        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                # Remap known features to their original IDs
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not self.training:
                # In eval mode, map unknown features to UNKNOWN_FEATURE_ID
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID
                unknown_features.append(name)
            else:
                # In training mode, learn new features
                self.original_feature_mapping[name] = new_id

        if self.feature_id_remap:
            logger.info(
                f"Created feature remapping: {len(self.feature_id_remap)} remapped, {len(unknown_features)} unknown"
            )
            self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to policy if it supports it, and update normalizations.

        This allows policies that understand feature remapping (like ComponentPolicy)
        to update their observation components, while vanilla torch.nn.Module policies
        will simply ignore this.
        """
        # Build complete remapping tensor
        remap_tensor = torch.arange(256, dtype=torch.uint8, device=self.device)

        # Apply explicit remappings
        for new_id, original_id in self.feature_id_remap.items():
            remap_tensor[new_id] = original_id

        # Map unused feature IDs to UNKNOWN
        current_feature_ids = {props["id"] for props in features.values()}
        for feature_id in range(256):
            if feature_id not in self.feature_id_remap and feature_id not in current_feature_ids:
                remap_tensor[feature_id] = unknown_id

        # Delegate feature remapping to policy if it supports it
        # Policies have _apply_feature_remapping that takes just the remap_tensor
        if self.policy is not None and hasattr(self.policy, "_apply_feature_remapping"):
            self.policy._apply_feature_remapping(remap_tensor)
        elif self.policy is None and "_obs_" in self.components:
            # MockAgent case - directly update observation component
            obs_component = self.components["_obs_"]
            if hasattr(obs_component, "update_feature_remapping"):
                obs_component.update_feature_remapping(remap_tensor)

        # Update normalization factors
        self._update_normalization_factors(features)

    def _update_normalization_factors(self, features: dict[str, dict]):
        """Update normalization factors after feature remapping."""
        # Delegate normalization update to policy if it supports it
        if hasattr(self.policy, "update_normalization_factors"):
            self.policy.update_normalization_factors(features, getattr(self, "original_feature_mapping", None))

    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for saving in metadata."""
        return getattr(self, "original_feature_mapping", None)

    def restore_original_feature_mapping(self, mapping: dict[str, int]) -> None:
        """Restore the original feature mapping from metadata."""
        # Make a copy to avoid shared state between agents
        self.original_feature_mapping = mapping.copy()
        logger.info(f"Restored original feature mapping with {len(mapping)} features from metadata")

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Initialize action space for the agent."""
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )

        # Build full action names for embeddings
        full_action_names = [f"{name}_{i}" for name, max_param in self.active_actions for i in range(max_param + 1)]

        # Activate embeddings if policy supports it
        if hasattr(self.policy, "activate_action_embeddings"):
            self.policy.activate_action_embeddings(full_action_names, device)

        # Create action index tensor
        self.action_index_tensor = torch.tensor(
            [[idx, j] for idx, max_param in enumerate(action_max_params) for j in range(max_param + 1)],
            device=device,
            dtype=torch.int32,
        )
        logger.info(f"Actions initialized: {self.active_actions}")

        # Pass tensors to policy if needed
        if self.policy is not None:
            for attr in ["action_index_tensor", "cum_action_max_params"]:
                if hasattr(self.policy, attr):
                    setattr(self.policy, attr, getattr(self, attr))

    @property
    def total_params(self):
        return self._total_params

    @property
    def lstm(self):
        """Access to LSTM component - delegates to policy if it has one."""
        if hasattr(self.policy, "lstm"):
            return self.policy.lstm
        return None

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss - delegates to policy."""
        if hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss()
        return torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def clip_weights(self):
        """Delegate weight clipping to the policy."""
        if self.policy is not None and hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies - delegates to policy."""
        if hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics - delegates to policy."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices - delegates to policy."""
        if hasattr(self.policy, "_convert_action_to_logit_index"):
            return self.policy._convert_action_to_logit_index(flattened_action)
        raise NotImplementedError("Policy does not implement _convert_action_to_logit_index")

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete logit indices back to (action_type, action_param) pairs - delegates to policy."""
        if hasattr(self.policy, "_convert_logit_index_to_action"):
            return self.policy._convert_logit_index_to_action(logit_indices)
        raise NotImplementedError("Policy does not implement _convert_logit_index_to_action")

    def __setstate__(self, state):
        """Restore state from checkpoint."""
        # Check if this is an old checkpoint (has components but no policy)
        # Components could be in state directly or in _modules
        has_components = "components" in state or ("_modules" in state and "components" in state.get("_modules", {}))
        has_policy = "policy" in state or ("_modules" in state and "policy" in state.get("_modules", {}))

        if has_components and not has_policy:
            logger.info("Detected old checkpoint format - converting to new ComponentPolicy structure")

            # Extract the components and related attributes that belong in ComponentPolicy
            from metta.agent.component_policy import ComponentPolicy

            # First, break any circular references in the old state
            if "policy" in state and state.get("policy") is state:
                del state["policy"]
                logger.info("Removed circular reference: state['policy'] = state")

            # Create ComponentPolicy without calling __init__ to avoid rebuilding components
            policy = ComponentPolicy.__new__(ComponentPolicy)

            # Initialize nn.Module base class
            nn.Module.__init__(policy)

            # Extract components from wherever they are
            if "components" in state:
                components = state["components"]
            elif "_modules" in state and "components" in state["_modules"]:
                components = state["_modules"]["components"]
            else:
                components = nn.ModuleDict()

            # Transfer component-related attributes to the policy
            policy.components = components
            policy.components_with_memory = state.get("components_with_memory", [])
            policy.clip_range = state.get("clip_range", 0)
            policy.agent_attributes = state.get("agent_attributes", {})

            # Transfer action conversion tensors if they exist
            if "cum_action_max_params" in state:
                policy.cum_action_max_params = state["cum_action_max_params"]
            if "action_index_tensor" in state:
                policy.action_index_tensor = state["action_index_tensor"]

            # Transfer cfg if it exists
            if "cfg" in state:
                policy.cfg = state["cfg"]

            # Now create a minimal state for MettaAgent itself
            # Don't include "components" in the new state - that belongs to the policy now
            new_state = {}
            for key in state:
                # Skip components and _modules to avoid adding components to MettaAgent
                if key in ["components", "_modules"]:
                    continue
                # Only copy attributes that belong to MettaAgent
                if key in [
                    "obs_width",
                    "obs_height",
                    "action_space",
                    "feature_normalizations",
                    "device",
                    "obs_space",
                    "_total_params",
                    "cfg",
                    "active_features",
                    "feature_id_to_name",
                    "original_feature_mapping",
                    "active_actions",
                    "action_names",
                    "action_max_params",
                    "components_with_memory",
                    "clip_range",
                    "agent_attributes",
                    "cum_action_max_params",
                    "action_index_tensor",
                    "training",
                    "_parameters",
                    "_buffers",
                    "_non_persistent_buffers_set",
                    "_backward_pre_hooks",
                    "_backward_hooks",
                    "_is_full_backward_hook",
                    "_forward_hooks",
                    "_forward_hooks_with_kwargs",
                    "_forward_hooks_always_called",
                    "_forward_pre_hooks",
                    "_forward_pre_hooks_with_kwargs",
                    "_state_dict_hooks",
                    "_state_dict_pre_hooks",
                    "_load_state_dict_pre_hooks",
                    "_load_state_dict_post_hooks",
                ]:
                    new_state[key] = state[key]

            # Update MettaAgent with its state (without components)
            self.__dict__.update(new_state)

            # Ensure _modules dict exists but without components
            if "_modules" not in self.__dict__:
                self._modules = {}

            # Set the converted policy
            self.policy = policy

            # Ensure policy has device attribute if MettaAgent has one
            if hasattr(self, "device") and self.policy is not None:
                self.policy.device = self.device

            logger.info("Successfully converted old checkpoint to new structure")
        else:
            # Normal checkpoint restoration
            self.__dict__.update(state)


PolicyAgent = MettaAgent | DistributedMettaAgent
