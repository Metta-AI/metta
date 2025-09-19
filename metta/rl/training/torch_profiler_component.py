"""Torch profiler component for training."""

import logging
from typing import Any

from metta.rl.torch_profiler import TorchProfiler
from metta.rl.training.component import TrainerComponent
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class TorchProfilerConfig(Config):
    """Configuration for torch profiler."""

    epoch_interval: int = 1


class TorchProfilerComponent(TrainerComponent):
    """Manages torch profiling during training."""

    def __init__(self, config: TorchProfilerConfig):
        """Initialize torch profiler component.

        Args:
            config: Profiler configuration
        """
        super().__init__(config)
        self.config = config
        self.torch_profiler = None
        self._original_train_epoch = None
        self._wandb_run = None
        self._run_dir = None

    def register(self, trainer: Any) -> None:
        """Register this component with the trainer and wrap training methods.

        Args:
            trainer: The trainer instance to register with
        """
        super().register(trainer)

        # Create the torch profiler with trainer context
        if self.torch_profiler is None:
            # Get wandb run and run_dir from component attributes or trainer
            wandb_run = self._wandb_run if hasattr(self, "_wandb_run") else getattr(trainer, "_wandb_run", None)
            run_dir = self._run_dir if hasattr(self, "_run_dir") else getattr(trainer, "run_dir", None)

            # Use distributed helper to check if master
            is_master = trainer.distributed_helper.is_master() if hasattr(trainer, "distributed_helper") else True

            # Get profiler config from trainer if not provided
            profiler_config = self.config.profiler_config
            if profiler_config is None and hasattr(trainer, "trainer") and hasattr(trainer.trainer, "profiler"):
                profiler_config = trainer.trainer.profiler
            self.torch_profiler = TorchProfiler(
                master=is_master,
                profiler_config=profiler_config,
                wandb_run=wandb_run,
                run_dir=run_dir,
            )

        # Wrap the _train_epoch method to add profiling context
        if hasattr(trainer, "_train_epoch"):
            original_train_epoch = trainer._train_epoch

            def wrapped_train_epoch():
                with self.torch_profiler:
                    return original_train_epoch()

            trainer._train_epoch = wrapped_train_epoch
            self._original_train_epoch = original_train_epoch

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Step the profiler at epoch end."""
        if self.torch_profiler:
            self.torch_profiler.on_epoch_end(epoch)

    def on_training_complete(self, trainer: Any) -> None:
        """Finalize profiling on training completion."""
        # TorchProfiler handles cleanup in its context manager
        # Restore original method if we wrapped it
        if self._original_train_epoch and hasattr(trainer, "_train_epoch"):
            trainer._train_epoch = self._original_train_epoch
