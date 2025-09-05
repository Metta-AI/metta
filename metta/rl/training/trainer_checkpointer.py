"""Trainer state checkpoint management component."""

import logging
from typing import Any, Dict, Optional, Tuple

from pydantic import Field

from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import TrainerComponent
from metta.rl.training.distributed_helper import DistributedHelper

logger = logging.getLogger(__name__)


class TrainerCheckpointerConfig(Config):
    """Configuration for trainer state checkpointing."""

    checkpoint_dir: str | None = Field(default=None)
    """Directory to save trainer state checkpoints"""

    epoch_interval: int = 50
    """How often to save trainer state checkpoints (in epochs)"""

    keep_last_n: int = 5
    """Number of recent trainer checkpoints to keep"""


class TrainerCheckpointer(TrainerComponent):
    """Manages trainer state checkpointing (optimizer, epoch, etc.)."""

    def __init__(
        self,
        config: TrainerCheckpointerConfig,
        distributed_helper: DistributedHelper,
    ):
        """Initialize trainer checkpointer.

        Args:
            config: Trainer checkpointer configuration
            checkpoint_manager: Checkpoint manager for saving/loading
            distributed_helper: Helper for distributed training
        """
        super().__init__(config)
        self.checkpoint_manager = CheckpointManager(run_dir=config.checkpoint_dir)
        self.distributed = distributed_helper
        self.config = config

    def load_trainer_state(self) -> Optional[Dict[str, Any]]:
        """Load trainer state from checkpoint.

        Returns:
            Trainer state dictionary or None if no checkpoint exists
        """
        if not self.distributed.is_master():
            # Only master loads, will be broadcast
            return None

        # Try to load latest trainer checkpoint
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_data and "trainer_state" in checkpoint_data:
            logger.info(f"Loading trainer state from epoch {checkpoint_data.get('epoch', 'unknown')}")
            return checkpoint_data["trainer_state"]

        return None

    def restore(self, trainer: Any) -> None:
        """Restore trainer state from checkpoint.

        Args:
            trainer: The trainer instance to restore state to
        """
        # Get checkpoint info
        _, trainer_state = self.get_checkpoint_info()

        if trainer_state:
            # Restore trainer state
            trainer.trainer_state.agent_step = trainer_state.get("agent_step", 0)
            trainer.trainer_state.epoch = trainer_state.get("epoch", 0)
            trainer.latest_saved_epoch = trainer.trainer_state.epoch

            # Restore timer state if available
            if "stopwatch_state" in trainer_state and hasattr(trainer, "timer"):
                trainer.timer.load_state(trainer_state["stopwatch_state"], resume_running=True)

            logger.info(
                f"Restored trainer state: epoch={trainer.trainer_state.epoch}, "
                f"agent_step={trainer.trainer_state.agent_step}"
            )

    def save_trainer_state(
        self,
        epoch: int,
        agent_step: int,
        optimizer_state_dict: Dict[str, Any],
        timer_state: Optional[Dict[str, Any]] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """Save trainer state checkpoint.

        Args:
            epoch: Current epoch
            agent_step: Current agent step
            optimizer_state_dict: Optimizer state dictionary
            timer_state: Optional timer/stopwatch state
            additional_state: Any additional state to save
            force: Force save even if not at interval

        Returns:
            True if saved successfully
        """
        if not self.distributed.should_checkpoint():
            return False

        if not force and epoch % self.config.epoch_interval != 0:
            return False

        trainer_state = {
            "epoch": epoch,
            "agent_step": agent_step,
            "optimizer_state": optimizer_state_dict,
        }

        if timer_state:
            trainer_state["stopwatch_state"] = timer_state

        if additional_state:
            trainer_state.update(additional_state)

        try:
            # Save trainer state through checkpoint manager
            # We'll save it as part of the metadata in the checkpoint system
            self.checkpoint_manager.save_metadata(epoch=epoch, metadata={"trainer_state": trainer_state})
            logger.info(f"Saved trainer state at epoch {epoch}")

            # Clean up old trainer checkpoints if needed
            self._cleanup_old_checkpoints(epoch)

            return True
        except Exception as e:
            logger.error(f"Failed to save trainer state: {e}")
            return False

    def _cleanup_old_checkpoints(self, current_epoch: int) -> None:
        """Remove old trainer checkpoints beyond keep_last_n.

        Args:
            current_epoch: Current epoch number
        """
        if self.config.keep_last_n <= 0:
            return

        # Get list of trainer checkpoints and remove old ones
        # This would need implementation in checkpoint_manager
        # For now, just log the intent
        logger.debug(f"Would clean up trainer checkpoints older than {current_epoch - self.config.keep_last_n}")

    def get_checkpoint_info(self) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """Get checkpoint epoch and trainer state if available.

        Returns:
            Tuple of (checkpoint_epoch, trainer_state) or (None, None)
        """
        trainer_state = None
        checkpoint_epoch = None

        if self.distributed.is_master():
            # Try to load existing trainer state
            trainer_state = self.load_trainer_state()
            if trainer_state:
                checkpoint_epoch = trainer_state.get("epoch", 0)
                logger.info(f"Found existing trainer checkpoint at epoch {checkpoint_epoch}")

        # Broadcast from master to all workers
        checkpoint_epoch = self.distributed.broadcast_from_master(checkpoint_epoch)
        trainer_state = self.distributed.broadcast_from_master(trainer_state)

        return checkpoint_epoch, trainer_state

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Save trainer state checkpoint at epoch end if due."""

        # Get timer state if available
        timer_state = None
        if hasattr(trainer, "timer") and hasattr(trainer.timer, "save_state"):
            timer_state = trainer.timer.save_state()

        # Save trainer state
        self.save_trainer_state(
            epoch=epoch,
            agent_step=trainer.trainer_state.agent_step,
            optimizer_state_dict=trainer.optimizer.state_dict(),
            timer_state=timer_state,
        )

    def on_training_complete(self, trainer: Any) -> None:
        """Save final trainer state on training completion."""
        # Force save final state
        timer_state = None
        if hasattr(trainer, "timer") and hasattr(trainer.timer, "save_state"):
            timer_state = trainer.timer.save_state()

        self.save_trainer_state(
            epoch=trainer.trainer_state.epoch,
            agent_step=trainer.trainer_state.agent_step,
            optimizer_state_dict=trainer.optimizer.state_dict(),
            timer_state=timer_state,
            force=True,  # Always save final state
        )

        """Finalize training."""
        if not self.distributed_helper.is_master():
            return

        logger.info("Training complete!")

        # Save final checkpoint
        metadata = {
            "agent_step": self._agent_step,
            "total_time": self.timer.get_elapsed(),
            "total_train_time": (
                self.timer.get_all_elapsed().get("_rollout", 0) + self.timer.get_all_elapsed().get("_train", 0)
            ),
            "is_final": True,
            "upload_to_wandb": False,
        }

        # Add final evaluation scores
        if self.evaluator:
            eval_scores = self.evaluator.get_latest_scores()
            if eval_scores.category_scores or eval_scores.simulation_scores:
                metadata.update(
                    {
                        "score": eval_scores.avg_simulation_score,
                        "avg_reward": eval_scores.avg_category_score,
                    }
                )

        # Final saves are handled by component callbacks (on_training_complete)

        # Log timing summary
        if self.distributed_helper.is_master():
            timing_summary = self.timer.get_all_summaries()
            logger.info("Timing Summary:")
            for name, summary in timing_summary.items():
                logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")
