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
                # obs_space=self.obs_space,
                # obs_width=self.obs_width,
                # obs_height=self.obs_height,
                # feature_normalizations=self.feature_normalizations,
                env=env,
            )
            logger.info(f"Using agent: {policy_architecture_cfg.name}")

        self.policy = policy
        if self.policy is not None:
            self.policy = self.policy.to(self.device)
            self.policy.device = self.device

        # self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set during initialization.")

        # Internal policies want tensor dicts, external policies want tensors
        if hasattr(self.policy, "wants_td") and self.policy.wants_td:
            return self.policy(td, action=action)
        else:
            x = td["env_obs"]
            # TODO: can add token to box shaper here
            # assume we only run external policies in simulation. otherwise we need to unpack return tuple
            action = self.policy(x, state, action)
            td["actions"] = action
            return td

    def get_cfg(self) -> AgentConfig:
        return self.cfg

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
                actions=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
                last_actions=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
                truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
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
    ) -> None:
        logs = self.policy.initialize_to_environment(features, action_names, action_max_params, device, is_training)
        for log in logs:
            if log is not None:
                log_on_master(log)

    @property
    def total_params(self):
        return self._total_params


PolicyAgent = MettaAgent | DistributedMettaAgent
