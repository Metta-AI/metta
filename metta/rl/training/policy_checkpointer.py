"""Policy checkpoint management component."""

import logging
from typing import Any, Dict, Optional

from metta.agent.policy import Policy, PolicyArchitecture
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import TrainerComponent
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.training_environment import EnvironmentMetaData

logger = logging.getLogger(__name__)


class PolicyCheckpointerConfig(Config):
    """Configuration for policy checkpointing."""

    epoch_interval: int = 100
    """How often to save policy checkpoints (in epochs)"""


class PolicyCheckpointer(TrainerComponent):
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

    def load_or_create_policy(
        self,
        env: EnvironmentMetaData,
        policy_architecture: PolicyArchitecture,
        policy_uri: Optional[str] = None,
    ) -> Policy:
        """Load policy from checkpoint/URI or create new one.

        Args:
            env: Environment for agent initialization
            policy_architecture_cfg: Policy architecture configuration
            policy_uri: Optional URI to load policy from (e.g., 'wandb://...' or 'file://...')

        Returns:
            Policy
        """
        existing_policy = None

        if self.distributed.is_master():
            # Try to load from URI first if provided
            if policy_uri:
                logger.info(f"Loading policy from URI: {policy_uri}")
                try:
                    existing_policy = self.checkpoint_manager.load_policy_from_uri(uri=policy_uri)
                except Exception as e:
                    logger.error(f"Failed to load from URI: {e}")
                    raise

        # Broadcast agent from master to all workers
        existing_policy = self.distributed.broadcast_from_master(existing_policy)

        if existing_policy:
            logger.info("Using loaded policy")
            return existing_policy

        # Create new agent if no checkpoint exists
        logger.info("Creating new agent from scratch")
        new_policy = policy_architecture.make_policy(
            env_metadata=env.meta_data,
            policy_architecture_cfg=policy_architecture,
        )
        return new_policy

    def save_policy(
        self,
        policy: Policy,
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

        if not force and epoch % self.config.epoch_interval != 0:
            return None

        # Save checkpoint
        checkpoint_uri = self.checkpoint_manager.save_policy_checkpoint(
            policy=policy,
            epoch=epoch,
            metadata=metadata or {},
        )

        if checkpoint_uri:
            logger.info(f"Saved policy checkpoint at epoch {epoch}: {checkpoint_uri}")

        return checkpoint_uri

    def save_policy_to_buffer(self, policy: Policy) -> bytes:
        """Save policy to bytes buffer.

        Args:
            policy: Policy to save

        Returns:
            Policy as bytes
        """
        return self.checkpoint_manager.save_policy_checkpoint_to_buffer(policy)

    def get_latest_policy_uri(self) -> Optional[str]:
        """Get URI for the latest policy checkpoint.

        Returns:
            Policy checkpoint URI or None if no checkpoint exists
        """
        checkpoint_uris = self.checkpoint_manager.select_policy_checkpoints("latest", count=1)
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

    def on_training_complete(self, trainer: Any) -> None:
        """Save final policy checkpoint when training completes.

        Args:
            trainer: The trainer instance
        """
        # Build final metadata
        metadata = {
            "agent_step": trainer.trainer_state.agent_step,
            "epoch": trainer.trainer_state.epoch,
            "total_time": trainer.timer.get_elapsed(),
            "total_train_time": (
                trainer.timer.get_all_elapsed().get("_rollout", 0) + trainer.timer.get_all_elapsed().get("_train", 0)
            ),
            "is_final": True,
            "upload_to_wandb": False,
        }

        # Add final evaluation scores if available
        if hasattr(trainer, "evaluator") and trainer.evaluator:
            eval_scores = trainer.evaluator.get_latest_scores()
            if eval_scores and (eval_scores.category_scores or eval_scores.simulation_scores):
                metadata.update(
                    {
                        "score": eval_scores.avg_simulation_score,
                        "avg_reward": eval_scores.avg_category_score,
                    }
                )

        # Save final policy checkpoint
        self.save_policy(
            policy=trainer.policy,
            epoch=trainer.trainer_state.epoch,
            metadata=metadata,
            force=True,  # Force save final checkpoint
        )
