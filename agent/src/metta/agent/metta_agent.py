import logging
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy import Policy
from metta.rl.training.training_environment import TrainingEnvironment

logger = logging.getLogger("metta_agent")


class MettaAgent(nn.Module):
    def __init__(
        self,
        env: TrainingEnvironment,
        policy: Policy,
    ):
        super().__init__()
        self._obs_space = env.single_observation_space
        self._action_space = env.single_action_space
        self._feature_normalizations = env.feature_normalizations
        self._policy = policy

        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = None,
    ):
        """Initialize the agent to the current environment.

        Handles feature remapping to allow agents trained on one environment to work
        on another environment where features may have different IDs but same names.
        """
        self.device = device

        # Auto-detect training context if not explicitly provided
        if is_training is None:
            is_training = self.training
            logger.info(f"Auto-detected {'training' if is_training else 'simulation'} context")

        # Build feature mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self._feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            logger.info(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
        else:
            # Re-initialization - create remapping for agent portability
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
                elif not is_training:
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
                # Apply the remapping
                self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

        # Store action configuration
        self.action_names = action_names
        self.action_max_params = action_max_params

        # Compute action tensors for efficient indexing
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )
        self.action_index_tensor = torch.tensor(
            [[idx, j] for idx, max_param in enumerate(action_max_params) for j in range(max_param + 1)],
            device=device,
            dtype=torch.int32,
        )

        # Generate full action names
        full_action_names = [
            f"{name}_{i}"
            for name, max_param in zip(action_names, action_max_params, strict=False)
            for i in range(max_param + 1)
        ]

        # Initialize policy to environment
        self._policy.initialize_to_environment(full_action_names, device)

        # Share tensors with policy
        self._policy.action_index_tensor = self.action_index_tensor
        self._policy.cum_action_max_params = self.cum_action_max_params

        logger.info(
            f"Environment initialized with {len(features)} features and actions: "
            f"{list(zip(action_names, action_max_params, strict=False))}"
        )

    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to policy for agent portability across environments."""
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

        # Apply remapping to policy
        self._policy._apply_feature_remapping(remap_tensor)

        self._update_normalization_factors(features)

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        if hasattr(self._policy, "_convert_action_to_logit_index"):
            return self._policy._convert_action_to_logit_index(flattened_action)
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_params

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete logit indices back to (action_type, action_param) pairs."""
        if hasattr(self._policy, "_convert_logit_index_to_action"):
            return self._policy._convert_logit_index_to_action(logit_indices)
        return self.action_index_tensor[logit_indices]


class DistributedMettaAgent(DistributedDataParallel):
    """Because this class passes through __getattr__ to its self.module, it implements everything
    MettaAgent does. We only have a need for this class because using the DistributedDataParallel wrapper
    returns an object of almost the same interface: you need to call .module to get the wrapped agent."""

    module: "MettaAgent"

    def __init__(self, agent: "MettaAgent", device: torch.device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")

        layers_converted_agent: "MettaAgent" = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)  # type: ignore

        if device.type == "cpu":  # CPU doesn't need device_ids
            super().__init__(module=layers_converted_agent)
        else:
            super().__init__(module=layers_converted_agent, device_ids=[device], output_device=device)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
