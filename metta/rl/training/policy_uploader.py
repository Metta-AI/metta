"""Policy upload management component for wandb and other destinations."""

import logging
from typing import Any, Dict, Optional

from metta.common.wandb.wandb_context import WandbRun
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import TrainerComponent
from metta.rl.training.distributed_helper import DistributedHelper

logger = logging.getLogger(__name__)


class PolicyUploaderConfig(Config):
    """Configuration for policy uploading."""

    epoch_interval: int = 1000
    """How often to upload policy to wandb (in epochs)"""


class PolicyUploader(TrainerComponent):
    """Manages uploading policies to wandb and other destinations."""

    def __init__(
        self,
        config: PolicyUploaderConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        wandb_run: Optional[WandbRun] = None,
    ):
        """Initialize policy uploader.

        Args:
            config: Policy uploader configuration
            checkpoint_manager: Checkpoint manager for uploading
            distributed_helper: Helper for distributed training
            wandb_run: Optional wandb run for uploading
        """
        super().__init__(config)
        self.checkpoint_manager = checkpoint_manager
        self.distributed = distributed_helper
        self.wandb_run = wandb_run
        self.config = config

    def upload_to_wandb(
        self,
        checkpoint_uri: str,
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Upload checkpoint to wandb.

        Args:
            checkpoint_uri: URI of checkpoint to upload
            epoch: Current epoch
            metadata: Optional metadata to include
            force: Force upload even if not at interval

        Returns:
            Wandb URI if uploaded, else None
        """
        if not self.distributed.should_checkpoint():
            return None

        if not self.wandb_run:
            return None

        if not force and epoch % self.config.epoch_interval != 0:
            return None

        try:
            artifact_name = f"policy-{epoch}"
            wandb_uri = self.checkpoint_manager.upload_checkpoint_to_wandb(
                checkpoint_uri=checkpoint_uri,
                wandb_run=self.wandb_run,
                artifact_name=artifact_name,
                metadata=metadata,
            )
            logger.info(f"Uploaded policy to wandb: {wandb_uri}")
            return wandb_uri
        except Exception as e:
            logger.error(f"Failed to upload to wandb: {e}")
            return None

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Check if policy should be uploaded at epoch end."""

        # Check if we should upload
        if epoch % self.config.epoch_interval != 0:
            return

        # Get latest checkpoint URI from policy checkpointer
        checkpoint_uri = trainer.policy_checkpointer.get_latest_policy_uri()
        if not checkpoint_uri:
            logger.debug(f"No checkpoint available to upload at epoch {epoch}")
            return

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

        # Upload to wandb
        self.upload_to_wandb(
            checkpoint_uri=checkpoint_uri,
            epoch=epoch,
            metadata=metadata,
        )

    def on_training_complete(self, trainer: Any) -> None:
        """Upload final policy on training completion."""
        epoch = trainer.trainer_state.epoch

        # Get latest checkpoint URI
        checkpoint_uri = trainer.policy_checkpointer.get_latest_policy_uri()
        if not checkpoint_uri:
            logger.debug("No checkpoint available for final upload")
            return

        # Build metadata
        metadata = {
            "epoch": epoch,
            "agent_step": trainer.trainer_state.agent_step,
            "final": True,
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

        # Force upload final policy
        self.upload_to_wandb(
            checkpoint_uri=checkpoint_uri,
            epoch=epoch,
            metadata=metadata,
            force=True,
        )
