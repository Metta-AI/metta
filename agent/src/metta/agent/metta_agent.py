import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.agent_config import AgentConfig, create_agent
from metta.rl.experience import Experience
from metta.rl.system_config import SystemConfig

logger = logging.getLogger("metta_agent")


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


class DistributedMettaAgent(DistributedDataParallel):
    """Because this class passes through __getattr__ to its self.module, it implements everything
    MettaAgent does. We only have a need for this class because using the DistributedDataParallel wrapper
    returns an object of almost the same interface: you need to call .module to get the wrapped agent."""

    module: "MettaAgent"

    def __init__(self, agent: "MettaAgent", device: torch.device):
        log_on_master("Converting BatchNorm layers to SyncBatchNorm for distributed training...")

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


class MettaAgent(nn.Module):
    def __init__(
        self,
        env,
        system_cfg: SystemConfig,
        policy_architecture_cfg: AgentConfig,
        policy: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = policy_architecture_cfg
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
            policy = create_agent(
                config=policy_architecture_cfg,
                obs_space=self.obs_space,
                obs_width=self.obs_width,
                obs_height=self.obs_height,
                feature_normalizations=self.feature_normalizations,
                env=env,
            )
            logger.info(f"Using agent: {policy_architecture_cfg.name}")

        self.policy = policy
        if self.policy is not None:
            self.policy = self.policy.to(self.device)
            self.policy.device = self.device

        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set during initialization.")

        # Internal policies want tensor dicts, external policies want tensors
        if hasattr(self.policy, "wants_td") and self.policy.wants_td:
            return self.policy(td, state, action)
        else:
            x = td["env_obs"]
            # assume we only run external policies in simulation. otherwise we need to unpack return tuple
            action = self.policy(x, state, action)
            td["actions"] = action
            return td

    def get_cfg(self) -> AgentConfig:
        return self.cfg

    def on_new_training_run(self):
        if hasattr(self.policy, "on_new_training_run"):
            self.policy.on_new_training_run()

    def on_rollout_start(self):
        if hasattr(self.policy, "on_rollout_start"):
            self.policy.on_rollout_start()

    def on_train_mb_start(self):
        if hasattr(self.policy, "on_train_mb_start"):
            self.policy.on_train_mb_start()

    def on_eval_start(self):
        if hasattr(self.policy, "on_eval_start"):
            self.policy.on_eval_start()

    # need to revisit these methods
    def reset_memory(self) -> None:
        """Reset memory - delegates to policy if it supports memory."""
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def get_memory(self) -> dict:
        """Get memory state - delegates to policy if it supports memory."""
        return getattr(self.policy, "get_memory", lambda: {})()

    def get_agent_experience_spec(self) -> Composite:
        if hasattr(self.policy, "get_agent_experience_spec"):
            return self.policy.get_agent_experience_spec()
        else:
            return Composite(
                env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
                dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            )

    def attach_replay_buffer(self, experience: Experience):
        """Losses expect to find a replay buffer in the policy."""
        self.replay = experience

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
            log_on_master(f"Auto-detected {'training' if is_training else 'simulation'} context")

        # Build feature mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            log_on_master(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
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

        # Share tensors with policy
        self.policy.action_index_tensor = self.action_index_tensor
        self.policy.cum_action_max_params = self.cum_action_max_params

        # Initialize policy to environment
        self.policy.initialize_to_environment(full_action_names, device)

        log_on_master(
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
        self.policy._apply_feature_remapping(remap_tensor)

        self._update_normalization_factors(features)

    def _update_normalization_factors(self, features: dict[str, dict]):
        """Update normalization factors after feature remapping."""
        self.policy.update_normalization_factors(features, getattr(self, "original_feature_mapping", None))

    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for saving in metadata."""
        return getattr(self, "original_feature_mapping", None)

    @property
    def total_params(self):
        return self._total_params


PolicyAgent = MettaAgent | DistributedMettaAgent
