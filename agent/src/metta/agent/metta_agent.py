import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.agent_config import AgentConfig, create_agent
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

        # Convert BatchNorm layers to SyncBatchNorm for distributed training
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
            # Move policy to device - this matches how main branch handled it
            self.policy = self.policy.to(self.device)
            # Set device attribute if the policy supports it
            if hasattr(self.policy, "device"):
                self.policy.device = self.device

        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set during initialization.")

        return self.policy(td, state, action)

    def reset_memory(self) -> None:
        """Reset memory - delegates to policy."""
        self.policy.reset_memory()

    def get_memory(self) -> dict:
        """Get memory state - delegates to policy if it supports memory."""
        return getattr(self.policy, "get_memory", lambda: {})()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            action=UnboundedDiscrete(shape=torch.Size([2]), dtype=torch.int64),
            action_dist=UnboundedContinuous(shape=torch.Size([2, 3]), dtype=torch.float32),
        )

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
    ):
        """Initialize the agent to the current environment."""
        self.device = device

        # Build feature mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

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
        self.policy.initialize_to_environment(full_action_names, device)

        # Share tensors with policy
        self.policy.action_index_tensor = self.action_index_tensor
        self.policy.cum_action_max_params = self.cum_action_max_params

        log_on_master(
            f"Environment initialized with {len(features)} features and actions: "
            f"{list(zip(action_names, action_max_params, strict=False))}"
        )


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
        return self.policy.l2_init_loss()

    def clip_weights(self):
        """Clip weights to prevent large updates - delegates to policy."""
        self.policy.clip_weights()

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies - delegates to policy."""
        self.policy.update_l2_init_weight_copy()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics - delegates to policy."""
        return self.policy.compute_weight_metrics(delta)

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        if hasattr(self.policy, "_convert_action_to_logit_index"):
            return self.policy._convert_action_to_logit_index(flattened_action)
        # Default implementation using MettaAgent's action tensors
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_params

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete logit indices back to (action_type, action_param) pairs."""
        if hasattr(self.policy, "_convert_logit_index_to_action"):
            return self.policy._convert_logit_index_to_action(logit_indices)
        # Default implementation using MettaAgent's action tensors
        return self.action_index_tensor[logit_indices]


PolicyAgent = MettaAgent | DistributedMettaAgent
