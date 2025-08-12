import logging
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

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
    def __init__(
        self,
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        obs_width: int,
        obs_height: int,
        action_space: gym.spaces.Space,
        feature_normalizations: dict[int, float],
        device: str,
        cfg,
        policy,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.policy = policy
        if self.policy is not None:
            self.policy.device = self.device

        self.obs_space = obs_space
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.action_space = action_space
        self.feature_normalizations = feature_normalizations

        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def set_policy(self, policy):
        """Set the agent's policy."""
        self.policy = policy
        self.policy.agent = self
        self.policy.device = self.device
        self.policy.to(self.device)

    def forward(self, td: Dict[str, torch.Tensor], state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set. Use set_policy() first.")

        # All policies should accept the same forward signature
        return self.policy(td, state, action)

    def reset_memory(self) -> None:
        """Reset memory - delegates to policy if it supports memory."""
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def get_memory(self) -> dict:
        """Get memory state - delegates to policy if it supports memory."""
        if hasattr(self.policy, "get_memory"):
            return self.policy.get_memory()
        return {}

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
        # MettaAgent handles the initialization for all policy types
        self.activate_actions(action_names, action_max_params, device)
        self._initialize_observations(features, device)

        # Let the policy know about environment initialization if it has such a method
        if hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(features, action_names, action_max_params, device, is_training)

    def _initialize_observations(self, features: dict[str, dict], device):
        """Initialize observation features by storing the feature mapping."""
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
        """Apply feature remapping to observation component and update normalizations."""
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

        # Precompute cumulative sums for efficient action conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        # Build full action names and activate embeddings
        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        # Delegate action embedding activation to policy if it supports it
        if hasattr(self.policy, "activate_action_embeddings"):
            self.policy.activate_action_embeddings(full_action_names, self.device)

        # Create action index tensor for conversions
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)
        logger.info(f"Actions initialized: {self.active_actions}")

        # Activate policy attributes
        if self.policy is not None:
            self.policy.action_index_tensor = self.action_index_tensor
            self.policy.cum_action_max_params = self.cum_action_max_params

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
        return torch.tensor(0.0, device=self.device)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies - delegates to policy."""
        if hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics - delegates to policy."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []


PolicyAgent = MettaAgent | DistributedMettaAgent
