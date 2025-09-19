"""Torch profiler component for training."""

import logging
from typing import Any, Optional

from metta.common.wandb.wandb_context import WandbRun
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.training.component import TrainerComponent
from metta.rl.training.context import TrainerContext

logger = logging.getLogger(__name__)


class TorchProfilerComponent(TrainerComponent):
    """Manages torch profiling during training."""

    def __init__(
        self,
        *,
        profiler_config: Any,
        wandb_run: Optional[WandbRun] = None,
        run_dir: Optional[str] = None,
        is_master: bool = True,
    ) -> None:
        interval = getattr(profiler_config, "interval_epochs", 0)
        super().__init__(epoch_interval=max(1, interval) if interval else 0)
        self._config = profiler_config
        self._wandb_run = wandb_run
        self._run_dir = run_dir
        self._is_master = is_master
        self._torch_profiler: Optional[TorchProfiler] = None
        self._original_train_epoch = None
        self._master_only = True

    def register(self, context: TrainerContext) -> None:  # type: ignore[override]
        super().register(context)
        interval = getattr(self._config, "interval_epochs", 0)
        if not interval:
            return

        if self._torch_profiler is None:
            run_dir = self._run_dir or getattr(context, "run_dir", None)
            self._torch_profiler = TorchProfiler(
                master=self._is_master,
                profiler_config=self._config,
                wandb_run=self._wandb_run,
                run_dir=run_dir,
            )

        trainer = context.trainer
        original_train_epoch = trainer._train_epoch  # type: ignore[attr-defined]

        def wrapped_train_epoch():
            if self._torch_profiler is None:
                return original_train_epoch()
            with self._torch_profiler:
                return original_train_epoch()

        trainer._train_epoch = wrapped_train_epoch  # type: ignore[attr-defined]
        self._original_train_epoch = original_train_epoch

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if self._torch_profiler:
            self._torch_profiler.on_epoch_end(epoch)

    def on_training_complete(self) -> None:  # type: ignore[override]
        if self._original_train_epoch is not None:
            self.context.trainer._train_epoch = self._original_train_epoch  # type: ignore[attr-defined]
