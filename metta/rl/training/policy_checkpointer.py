"""Policy checkpoint management component."""

import logging
from typing import Any, Dict, Optional

import torch

from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.mettagrid import MettaGridEnv
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import ComponentConfig, MasterComponent
from metta.rl.training.distributed_helper import DistributedHelper

logger = logging.getLogger(__name__)


class PolicyCheckpointerConfig(ComponentConfig):
    """Configuration for policy checkpointing."""

    interval: int = 100
    """How often to save policy checkpoints (in epochs)"""


class PolicyCheckpointer(MasterComponent):
    """Manages policy checkpointing with distributed awareness and URI support."""

    def __init__(
        self,
        config: PolicyCheckpointerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
    ):
        """Initialize policy checkpointer.

        Args:
            config: Policy checkpointer configuration
            checkpoint_manager: Checkpoint manager for saving/loading
            distributed_helper: Helper for distributed training
        """
        super().__init__(config)
        self.checkpoint_manager = checkpoint_manager
        self.distributed = distributed_helper
        self.config = config

    def load_or_create_agent(
        self,
        env: MettaGridEnv,
        system_cfg: Any,
        agent_cfg: Any,
        device: torch.device,
        policy_uri: Optional[str] = None,
    ) -> PolicyAgent:
        """Load agent from checkpoint/URI or create new one.

        Args:
            env: Environment for agent initialization
            system_cfg: System configuration
            agent_cfg: Agent configuration
            device: Device to load agent on
            policy_uri: Optional URI to load policy from (e.g., 'wandb://...' or 'file://...')

        Returns:
            PolicyAgent
        """
        existing_agent = None

        if self.distributed.is_master():
            # Try to load from URI first if provided
            if policy_uri:
                logger.info(f"Loading policy from URI: {policy_uri}")
                try:
                    existing_agent = self.checkpoint_manager.load_agent_from_uri(uri=policy_uri, device=device)
                except Exception as e:
                    logger.error(f"Failed to load from URI: {e}")
                    raise

        # Broadcast agent from master to all workers
        existing_agent = self.distributed.broadcast_from_master(existing_agent)

        if existing_agent:
            logger.info("Using loaded agent")
            return existing_agent

        # Create new agent if no checkpoint exists
        logger.info("Creating new agent from scratch")
        new_agent = MettaAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=agent_cfg,
            system_config=system_cfg,
            device=device,
        )
        return new_agent

    def save_policy(
        self,
        policy: PolicyAgent,
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Save policy checkpoint.

        Args:
            policy: Policy to save
            epoch: Current epoch
            metadata: Optional metadata to save with checkpoint
            force: Force save even if not at interval

        Returns:
            Checkpoint URI if saved, else None
        """
        if not self.distributed.should_checkpoint():
            return None

        if not force and epoch % self.config.interval != 0:
            return None

        # Save checkpoint
        checkpoint_uri = self.checkpoint_manager.save_checkpoint(
            agent=policy,
            epoch=epoch,
            metadata=metadata or {},
        )

        if checkpoint_uri:
            logger.info(f"Saved policy checkpoint at epoch {epoch}: {checkpoint_uri}")

        return checkpoint_uri

    def save_policy_to_buffer(self, policy: PolicyAgent) -> bytes:
        """Save policy to bytes buffer.

        Args:
            policy: Policy to save

        Returns:
            Policy as bytes
        """
        return self.checkpoint_manager.save_agent_to_buffer(policy)

    def get_latest_policy_uri(self) -> Optional[str]:
        """Get URI for the latest policy checkpoint.

        Returns:
            Policy checkpoint URI or None if no checkpoint exists
        """
        checkpoint_uris = self.checkpoint_manager.select_checkpoints("latest", count=1)
        return checkpoint_uris[0] if checkpoint_uris else None

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Save policy checkpoint at epoch end if due."""

        # Build metadata
        metadata = {
            "epoch": epoch,
            "agent_step": trainer.trainer_state.agent_step,
        }

        # Add evaluation scores if available
        if hasattr(trainer, "evaluator") and trainer.evaluator:
            eval_scores = trainer.evaluator.get_latest_scores()
            if eval_scores and (eval_scores.category_scores or eval_scores.simulation_scores):
                metadata.update(
                    {
                        "score": eval_scores.avg_simulation_score,
                        "avg_reward": eval_scores.avg_category_score,
                    }
                )

        # Save policy
        self.save_policy(
            policy=trainer.policy,
            epoch=epoch,
            metadata=metadata,
        )
