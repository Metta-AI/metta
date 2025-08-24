"""Trainer agent builder for creating policy agents during training."""

import torch

from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.agent.policy_loader import AgentBuilder
from metta.agent.policy_metadata import PolicyMetadata
from metta.core.distributed import TorchDistributedConfig
from metta.mettagrid import MettaGridEnv
from metta.rl.policy_management import initialize_policy_for_environment, wrap_agent_distributed
from metta.rl.system_config import SystemConfig


class TrainerAgentBuilder(AgentBuilder):
    """Agent builder for training scenarios."""

    def __init__(
        self,
        metta_grid_env: MettaGridEnv,
        system_cfg: SystemConfig,
        agent_cfg: AgentConfig,
        device: torch.device,
        torch_dist_cfg: TorchDistributedConfig,
    ):
        """Initialize the trainer agent builder.

        Args:
            metta_grid_env: The MettaGrid environment
            system_cfg: System configuration
            agent_cfg: Agent configuration
            device: Device to use for the agent
            torch_dist_cfg: Distributed training configuration
        """
        self.metta_grid_env = metta_grid_env
        self.system_cfg = system_cfg
        self.agent_cfg = agent_cfg
        self.device = device
        self.torch_dist_cfg = torch_dist_cfg

    def initialize_agent(self, policy_metadata: PolicyMetadata, weights: dict[str, torch.Tensor] | None) -> PolicyAgent:
        """Initialize a policy agent for training.

        Args:
            policy_metadata: Policy metadata
            weights: Optional weights to load into the agent

        Returns:
            Initialized policy agent
        """
        agent = MettaAgent(self.metta_grid_env, self.system_cfg, self.agent_cfg)
        agent.initialize_to_environment(
            features=self.metta_grid_env.get_observation_features(),
            action_names=self.metta_grid_env.action_names,
            action_max_params=self.metta_grid_env.max_action_args,
            device=self.device,
            is_training=True,
        )
        if weights:
            agent.load_state_dict(weights)

        # Wrap in DDP if distributed
        if torch.distributed.is_initialized():
            if self.torch_dist_cfg.is_master:
                import logging

                logger = logging.getLogger(__name__)
                logger.info("Initializing DistributedDataParallel")
            torch.distributed.barrier()
            agent = wrap_agent_distributed(agent, self.device)
            torch.distributed.barrier()

        # Initialize policy to environment after distributed wrapping
        # This must happen after wrapping to ensure all ranks do it at the same time
        initialize_policy_for_environment(
            policy=agent,
            metta_grid_env=self.metta_grid_env,
            device=self.device,
            restore_feature_mapping=True,
            metadata=policy_metadata,
        )

        return agent
