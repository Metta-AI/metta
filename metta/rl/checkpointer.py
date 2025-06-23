"""Simple checkpoint management for training."""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from metta.rl.trainer_checkpoint import TrainerCheckpoint

if TYPE_CHECKING:
    from metta.agent import BaseAgent
    from metta.agent.policy_store import PolicyRecord, PolicyStore

logger = logging.getLogger(__name__)


class TrainingCheckpointer:
    """Simple checkpointer for training state and policies."""

    def __init__(
        self,
        checkpoint_dir: str,
        policy_store: "PolicyStore",
        wandb_run=None,
    ):
        """Initialize checkpointer.

        Args:
            checkpoint_dir: Directory to save checkpoints
            policy_store: Policy storage backend
            wandb_run: Optional wandb run for artifact tracking
        """
        self.checkpoint_dir = checkpoint_dir
        self.policy_store = policy_store
        self.wandb_run = wandb_run
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_policy(
        self,
        policy: "BaseAgent",
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyRecord":
        """Save policy checkpoint.

        Args:
            policy: Policy to save
            epoch: Current epoch
            metadata: Optional metadata to store

        Returns:
            PolicyRecord for the saved policy
        """
        name = f"policy_epoch_{epoch}"
        path = os.path.join(self.checkpoint_dir, name)

        # Add epoch to metadata
        if metadata is None:
            metadata = {}
        metadata["epoch"] = epoch

        # Save through policy store
        policy_record = self.policy_store.save(
            name=name,
            path=path,
            policy=policy,
            metadata=metadata,
        )

        logger.info(f"Saved policy checkpoint: {name}")
        return policy_record

    def save_trainer_state(
        self,
        agent_step: int,
        epoch: int,
        optimizer_state_dict: Dict[str, Any],
        run_dir: str,
        **extra_args,
    ):
        """Save trainer state checkpoint.

        Args:
            agent_step: Current agent steps
            epoch: Current epoch
            optimizer_state_dict: Optimizer state
            run_dir: Run directory
            **extra_args: Additional state to save
        """
        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            total_agent_step=agent_step,  # Simplified
            optimizer_state_dict=optimizer_state_dict,
            extra_args=extra_args,
        )
        checkpoint.save(run_dir)
        logger.info(f"Saved trainer state at epoch {epoch}")

    def load_trainer_state(self, run_dir: str) -> TrainerCheckpoint:
        """Load trainer state checkpoint.

        Args:
            run_dir: Run directory

        Returns:
            Loaded checkpoint
        """
        return TrainerCheckpoint.load(run_dir)

    def load_policy(
        self,
        checkpoint: TrainerCheckpoint,
        env,
        initial_policy_uri: Optional[str] = None,
    ) -> "PolicyRecord":
        """Load policy from checkpoint or create new.

        Args:
            checkpoint: Trainer checkpoint
            env: Environment for creating new policy
            initial_policy_uri: Optional initial policy URI

        Returns:
            Loaded or created policy record
        """
        # Try checkpoint first
        if checkpoint.policy_path:
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            return self.policy_store.policy(checkpoint.policy_path)

        # Try initial policy URI
        if initial_policy_uri:
            logger.info(f"Loading initial policy: {initial_policy_uri}")
            return self.policy_store.policy(initial_policy_uri)

        # Try default checkpoint location
        default_path = os.path.join(self.checkpoint_dir, "policy_epoch_0")
        if os.path.exists(default_path):
            logger.info(f"Loading policy from default path: {default_path}")
            return self.policy_store.policy(default_path)

        # Create new policy
        logger.info("Creating new policy")
        return self.policy_store.create(env)

    def save_to_wandb(self, policy_record: "PolicyRecord"):
        """Save policy to wandb as artifact.

        Args:
            policy_record: Policy record to save
        """
        if self.wandb_run is None:
            return

        try:
            self.policy_store.add_to_wandb_run(
                self.wandb_run.name,
                policy_record,
            )
            logger.info(f"Saved policy {policy_record.name} to wandb")
        except Exception as e:
            logger.warning(f"Failed to save to wandb: {e}")


class AutoCheckpointer(TrainingCheckpointer):
    """Checkpointer with automatic interval-based saving."""

    def __init__(
        self,
        checkpoint_dir: str,
        policy_store: "PolicyStore",
        wandb_run=None,
        is_master: bool = True,
        checkpoint_interval: int = 100,
        wandb_interval: int = 500,
    ):
        """Initialize auto-checkpointer.

        Args:
            checkpoint_dir: Directory to save checkpoints
            policy_store: Policy storage backend
            wandb_run: Optional wandb run
            is_master: Whether this is the master process
            checkpoint_interval: Epochs between checkpoints
            wandb_interval: Epochs between wandb saves
        """
        super().__init__(checkpoint_dir, policy_store, wandb_run)
        self.is_master = is_master
        self.checkpoint_interval = checkpoint_interval
        self.wandb_interval = wandb_interval

    def should_checkpoint(self, epoch: int) -> bool:
        """Check if should checkpoint at this epoch."""
        return self.is_master and epoch % self.checkpoint_interval == 0

    def should_save_to_wandb(self, epoch: int) -> bool:
        """Check if should save to wandb at this epoch."""
        return self.is_master and epoch % self.wandb_interval == 0
