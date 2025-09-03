"""Base class for all trainer callbacks."""

from typing import TYPE_CHECKING

import torch

from metta.rl.trainer_state import TrainerState

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class TrainerCallback:
    """Base class for all trainer callbacks.

    Callbacks are components that get invoked at specific points during training.
    This includes both losses (which compute gradients) and services (checkpointing,
    metrics, evaluation, etc). All callbacks share the same lifecycle methods.
    """

    __slots__ = (
        "trainer_cfg",
        "device",
        "instance_name",
        "critical",
    )

    def __init__(
        self,
        trainer_cfg: "TrainerConfig",
        device: torch.device,
        instance_name: str = "callback",
        critical: bool = False,
    ):
        """Initialize the callback with shared configuration.

        Args:
            trainer_cfg: The trainer configuration object
            device: The torch device to use for operations
            instance_name: Unique name for this callback instance
            critical: If True, failures in this callback will stop training
        """
        self.trainer_cfg = trainer_cfg
        self.device = device
        self.instance_name = instance_name
        self.critical = critical

    # ======================================================================
    # ============================ CALLBACK METHODS ========================
    # These methods are called at specific points during the training loop.

    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        """Called at the start of training.

        This is where callbacks can perform initialization tasks like:
        - Setting up monitoring systems
        - Loading previous state if resuming
        - Initializing data structures
        """
        pass

    def on_rollout_start(self, trainer_state: TrainerState) -> None:
        """Called before rollout phase begins.

        This callback occurs before the environment starts collecting
        experience for the current epoch.
        """
        pass

    def on_rollout_end(self, trainer_state: TrainerState) -> None:
        """Called after rollout phase completes.

        At this point:
        - trainer_state.rollout_stats contains the rollout statistics
        - Experience has been collected but training hasn't started
        """
        pass

    def on_mb_end(self, trainer_state: TrainerState) -> None:
        """Called after each minibatch.

        This callback occurs after each minibatch update during the
        training phase. Useful for fine-grained monitoring.
        """
        pass

    def on_train_phase_end(self, trainer_state: TrainerState) -> None:
        """Called after training phase completes.

        This callback occurs after all minibatches have been processed
        for the current epoch but before epoch increment.
        """
        pass

    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each epoch.

        At this point:
        - trainer_state.loss_stats contains aggregated loss statistics
        - trainer_state.eval_scores may contain evaluation results
        - trainer_state.latest_checkpoint_uri may contain checkpoint location
        - The epoch counter has been incremented

        This is typically where callbacks perform major operations like:
        - Saving checkpoints
        - Running evaluations
        - Logging metrics
        - Updating monitoring systems
        """
        pass

    def on_training_end(self, trainer_state: TrainerState) -> None:
        """Called when training completes.

        This is the final callback, useful for:
        - Final checkpoint saving
        - Cleanup operations
        - Final reporting
        """
        pass

    # ============================ END CONTROL FLOW ============================
    # ==========================================================================

    # ------------------------ UTILITY METHODS -----------------------------

    def should_run(self, epoch: int, interval: int) -> bool:
        """Utility to check if an operation should run based on interval.

        Args:
            epoch: Current epoch number
            interval: How often to run (e.g., every N epochs)

        Returns:
            True if the operation should run this epoch
        """
        if interval <= 0:
            return False
        return epoch > 0 and epoch % interval == 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(instance_name='{self.instance_name}')"

    # ------------------------ END UTILITY METHODS -----------------------------
