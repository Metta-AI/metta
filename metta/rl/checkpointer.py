"""Checkpointing components for saving and loading training state."""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.util.wandb.wandb_context import WandbRun

if TYPE_CHECKING:
    from metta.agent import BaseAgent
    from metta.agent.policy_store import PolicyRecord, PolicyStore
    from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


class TrainingCheckpointer:
    """Handles checkpointing of training state and policies.

    This class manages saving and loading of trainer state, policy models,
    and integration with wandb for model versioning.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        policy_store: "PolicyStore",
        wandb_run: Optional[WandbRun] = None,
        is_master: bool = True,
    ):
        """Initialize the checkpointer.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            policy_store: Store for saving/loading policies
            wandb_run: Optional wandb run for artifact tracking
            is_master: Whether this is the master process (for distributed training)
        """
        self.checkpoint_dir = checkpoint_dir
        self.policy_store = policy_store
        self.wandb_run = wandb_run
        self.is_master = is_master

        # Create checkpoint directory
        if self.is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def save_trainer_state(
        self,
        run_dir: str,
        agent_step: int,
        epoch: int,
        optimizer_state_dict: Dict[str, Any],
        world_size: int = 1,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save trainer checkpoint.

        Args:
            run_dir: Run directory for checkpoint
            agent_step: Current agent steps
            epoch: Current epoch
            optimizer_state_dict: Optimizer state
            world_size: Number of distributed processes
            extra_args: Additional checkpoint data
        """
        if not self.is_master:
            return

        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            total_agent_step=agent_step * world_size,
            optimizer_state_dict=optimizer_state_dict,
            extra_args=extra_args or {},
        )
        checkpoint.save(run_dir)

    def save_policy(
        self,
        policy: "BaseAgent",
        epoch: int,
        agent_step: int,
        env: "MettaGridEnv",
        metadata: Optional[Dict[str, Any]] = None,
        initial_pr: Optional["PolicyRecord"] = None,
    ) -> Optional["PolicyRecord"]:
        """Save policy checkpoint.

        Args:
            policy: Policy to save
            epoch: Current epoch
            agent_step: Current agent steps
            env: Environment (for action names)
            metadata: Additional metadata to save
            initial_pr: Initial policy record (for generation tracking)

        Returns:
            PolicyRecord for the saved policy
        """
        if not self.is_master:
            return None

        name = self.policy_store.make_model_name(epoch)

        # Build metadata
        save_metadata = {
            "agent_step": agent_step,
            "epoch": epoch,
            "action_names": env.action_names,
        }

        # Add generation info
        if initial_pr:
            save_metadata["generation"] = initial_pr.metadata.get("generation", 0) + 1
            save_metadata["initial_uri"] = initial_pr.uri

        # Add custom metadata
        if metadata:
            save_metadata.update(metadata)

        # Save policy
        return self.policy_store.save(
            name,
            os.path.join(self.checkpoint_dir, name),
            policy,
            metadata=save_metadata,
        )

    def save_to_wandb(self, policy_record: "PolicyRecord") -> None:
        """Save policy to wandb as an artifact.

        Args:
            policy_record: Policy record to save
        """
        if not self.is_master or self.wandb_run is None:
            return

        self.policy_store.add_to_wandb_run(self.wandb_run.name, policy_record)

    def load_trainer_state(self, run_dir: str) -> TrainerCheckpoint:
        """Load trainer checkpoint.

        Args:
            run_dir: Run directory containing checkpoint

        Returns:
            Loaded trainer checkpoint
        """
        return TrainerCheckpoint.load(run_dir)

    def load_policy(
        self,
        checkpoint: TrainerCheckpoint,
        env: "MettaGridEnv",
        initial_policy_uri: Optional[str] = None,
    ) -> "PolicyRecord":
        """Load policy from checkpoint or create new.

        Args:
            checkpoint: Trainer checkpoint
            env: Environment for creating new policy
            initial_policy_uri: Optional URI for initial policy

        Returns:
            Loaded or created policy record
        """
        import time

        for attempt in range(10):
            # Try loading from checkpoint
            if checkpoint.policy_path:
                logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
                return self.policy_store.policy(checkpoint.policy_path)

            # Try loading from initial URI
            elif initial_policy_uri is not None:
                logger.info(f"Loading initial policy URI: {initial_policy_uri}")
                return self.policy_store.policy(initial_policy_uri)

            # Try default checkpoint path
            else:
                policy_path = os.path.join(self.checkpoint_dir, self.policy_store.make_model_name(0))
                if os.path.exists(policy_path):
                    logger.info(f"Loading policy from checkpoint: {policy_path}")
                    return self.policy_store.policy(policy_path)
                elif self.is_master:
                    logger.info("Creating new policy!")
                    return self.policy_store.create(env)

            time.sleep(5)

        raise RuntimeError("Failed to load policy after 10 attempts")


class AutoCheckpointer(TrainingCheckpointer):
    """Automatic checkpointing with configurable intervals.

    This extends TrainingCheckpointer with automatic checkpointing
    based on epochs or time intervals.
    """

    def __init__(
        self,
        *args,
        checkpoint_interval: int = 100,
        wandb_interval: int = 500,
        time_interval_minutes: Optional[int] = None,
        **kwargs,
    ):
        """Initialize auto checkpointer.

        Args:
            checkpoint_interval: Epochs between checkpoints
            wandb_interval: Epochs between wandb saves
            time_interval_minutes: Optional time-based checkpointing
            *args, **kwargs: Arguments for TrainingCheckpointer
        """
        super().__init__(*args, **kwargs)
        self.checkpoint_interval = checkpoint_interval
        self.wandb_interval = wandb_interval
        self.time_interval_minutes = time_interval_minutes
        self._last_checkpoint_time = None

    def should_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint is needed.

        Args:
            epoch: Current epoch

        Returns:
            Whether to checkpoint now
        """
        # Epoch-based checkpointing
        if epoch % self.checkpoint_interval == 0:
            return True

        # Time-based checkpointing
        if self.time_interval_minutes is not None:
            import time

            current_time = time.time()
            if self._last_checkpoint_time is None:
                self._last_checkpoint_time = current_time
                return True
            elif (current_time - self._last_checkpoint_time) / 60 >= self.time_interval_minutes:
                self._last_checkpoint_time = current_time
                return True

        return False

    def should_save_to_wandb(self, epoch: int) -> bool:
        """Check if wandb save is needed.

        Args:
            epoch: Current epoch

        Returns:
            Whether to save to wandb now
        """
        return epoch % self.wandb_interval == 0
