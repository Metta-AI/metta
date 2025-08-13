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

        # This maintains the same interface as the input MettaAgent
        layers_converted_agent: "MettaAgent" = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)  # type: ignore

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
    """Wrapper that bridges between environments and policies.

    Architecture:
    - MettaAgent: Handles environment interface, feature remapping, action conversion
    - Policy (ComponentPolicy/PyTorch agents): Handles forward pass and network architecture

    Separation of Concerns:
    - Only MettaAgent has initialize_to_environment() - this is the environment interface
    - Policies only implement forward() and optionally specific methods like:
      - update_feature_remapping() for observation components
      - activate_action_embeddings() for action embeddings
      - clip_weights(), l2_init_loss(), etc. for training utilities
    """

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
        if self.policy is not None and hasattr(self.policy, "device"):
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

    def set_policy(self, policy):
        """Set the agent's policy."""
        self.policy = policy
        if hasattr(self.policy, "device"):
            self.policy.device = self.device
        self.policy.to(self.device)

    def forward(self, td: Dict[str, torch.Tensor], state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set. Use set_policy() first.")

        # Handle old checkpoints where self.policy == self (old MettaAgent WAS the policy)
        if self.policy is self and hasattr(self, "components"):
            # Old MettaAgent sets bptt/batch keys and runs value/action components
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
            if "_value_" in self.components:
                self.components["_value_"](td)
            if "_action_" in self.components:
                self.components["_action_"](td)

            # Delegate to appropriate method
            if action is None and hasattr(self, "forward_inference"):
                return self.forward_inference(td)
            elif action is not None and hasattr(self, "forward_training"):
                return self.forward_training(td, action)
            # Fallback
            td.setdefault("actions", torch.zeros((B, 2), dtype=torch.long))
            return td

        # New policies expect (td, state, action)
        return self.policy(td, state, action)

    def reset_memory(self) -> None:
        """Reset memory - delegates to policy if it supports memory."""
        if self.policy is self and hasattr(self, "components_with_memory"):
            # Old MettaAgent: reset components with memory
            for name in self.components_with_memory:
                if name in self.components and hasattr(self.components[name], "reset_memory"):
                    self.components[name].reset_memory()
        elif hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def get_memory(self) -> dict:
        """Get memory state - delegates to policy if it supports memory."""
        if self.policy is self and hasattr(self, "components_with_memory"):
            # Old MettaAgent: collect memory from components
            return {
                name: self.components[name].get_memory()
                for name in self.components_with_memory
                if name in self.components and hasattr(self.components[name], "get_memory")
            }
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
        """Initialize the agent to the current environment.

        This is the main entry point for environment initialization.
        Only MettaAgent should have this method - policies should not.
        """
        # MettaAgent handles all initialization
        self.activate_actions(action_names, action_max_params, device)
        self.activate_observations(features, device)

    def activate_observations(self, features: dict[str, dict], device):
        """Activate observation features by storing the feature mapping.

        This method handles feature remapping for policies that don't understand
        feature ID changes between training and evaluation environments.

        This is a MettaAgent-level concern - policies shouldn't need to know about
        feature remapping, they just work with the remapped observations.
        """
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
        if hasattr(self.policy, "update_feature_remapping"):
            self.policy.update_feature_remapping(remap_tensor)

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
        """Restore the original feature mapping from metadata.

        This should be called after loading a model from checkpoint but before
        calling initialize_to_environment.
        """
        # Make a copy to avoid shared state between agents
        self.original_feature_mapping = mapping.copy()
        logger.info(f"Restored original feature mapping with {len(mapping)} features from metadata")

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Initialize action space for the agent."""
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for action conversion
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

    def clip_weights(self):
        """Delegate weight clipping to the policy."""
        if self.policy is not None and hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    @property
    def total_params(self):
        """Total number of parameters."""
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
        self.__dict__.update(state)
        if not hasattr(self, "policy"):
            self.policy = self


PolicyAgent = MettaAgent | DistributedMettaAgent
