"""Torch profiler component for training."""

import logging
from typing import Any

from metta.rl.torch_profiler import TorchProfiler
from metta.rl.training.component import ComponentConfig, MasterComponent

logger = logging.getLogger(__name__)


class TorchProfilerConfig(ComponentConfig):
    """Configuration for torch profiler."""

    interval: int = 1
    """How often to step the profiler (in epochs)"""


class TorchProfilerComponent(MasterComponent):
    """Manages torch profiling during training."""

    def __init__(self, config: TorchProfilerConfig, torch_profiler: TorchProfiler):
        """Initialize torch profiler component.

        Args:
            config: Profiler configuration
            torch_profiler: Torch profiler instance
        """
        super().__init__(config)
        self.torch_profiler = torch_profiler
        self.config = config

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Step the profiler at epoch end."""
        self.torch_profiler.on_epoch_end(epoch)

    def on_training_complete(self, trainer: Any) -> None:
        """Finalize profiling on training completion."""
        # TorchProfiler handles cleanup in its context manager
        pass
