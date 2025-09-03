"""Base class for trainer hooks, following the same pattern as BaseLoss."""

import torch

from metta.rl.trainer_state import TrainerState


class TrainerHook:
    """Base class for trainer hooks, following BaseLoss patterns.

    Hooks provide a way to extract non-training concerns from the Trainer class
    while maintaining consistency with the existing callback system. Hooks can
    coexist with losses and use the same TrainerState-based data flow pattern.
    """

    __slots__ = (
        "trainer_cfg",
        "device",
        "instance_name",
        "critical",
    )

    def __init__(
        self,
        trainer_cfg,
        device: torch.device,
        instance_name: str = "hook",
        critical: bool = False,
    ):
        """Initialize the hook with shared configuration.

        Args:
            trainer_cfg: The trainer configuration object
            device: The torch device to use for operations
            instance_name: Unique name for this hook instance
            critical: If True, failures in this hook will stop training
        """
        self.trainer_cfg = trainer_cfg
        self.device = device
        self.instance_name = instance_name
        self.critical = critical

    # ======================================================================
    # ============================ CONTROL FLOW ============================
    # Following the same pattern as BaseLoss, hooks provide callbacks for
    # different stages of the training loop.

    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        """Called at the start of training.

        This is where hooks can perform initialization tasks like:
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

        This is typically where hooks perform major operations like:
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
