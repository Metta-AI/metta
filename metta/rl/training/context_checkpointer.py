"""Trainer state checkpoint management component."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import TrainerComponent
from metta.rl.training.component_context import ComponentContext
from metta.rl.training.distributed_helper import DistributedHelper
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class ContextCheckpointerConfig(Config):
    """Configuration for trainer state checkpointing."""

    epoch_interval: int = 50
    """How often to save trainer state checkpoints (in epochs)."""

    keep_last_n: int = 5
    """Number of trainer checkpoints to retain locally."""

    checkpoint_dir: str | None = None
    """Optional explicit directory for checkpoint artifacts."""


class ContextCheckpointer(TrainerComponent):
    """Persist and restore optimizer/timing state alongside policy checkpoints."""

    trainer_attr = "trainer_checkpointer"

    def __init__(
        self,
        *,
        config: ContextCheckpointerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        explicit_dir = self._config.checkpoint_dir
        if explicit_dir:
            Path(explicit_dir).mkdir(parents=True, exist_ok=True)
            logger.debug("Trainer checkpoints will be written to %s", explicit_dir)

    # ------------------------------------------------------------------
    # Public API used by Trainer
    # ------------------------------------------------------------------
    def restore(self, context: ComponentContext) -> None:
        """Load trainer state if checkpoints exist and broadcast to all ranks."""
        payload: Optional[Dict[str, Any]] = None

        if self._distributed.is_master():
            raw = self._checkpoint_manager.load_trainer_state()
            if raw:
                logger.info(
                    "Restoring trainer state from epoch=%s agent_step=%s", raw.get("epoch"), raw.get("agent_step")
                )
                payload = {
                    "agent_step": raw.get("agent_step", 0),
                    "epoch": raw.get("epoch", 0),
                    "optimizer_state": raw.get("optimizer_state", {}),
                    "stopwatch_state": raw.get("stopwatch_state"),
                }

        payload = self._distributed.broadcast_from_master(payload)
        if payload is None:
            return

        restored_epoch = payload["epoch"]
        context.agent_step = payload["agent_step"]
        context.epoch = restored_epoch
        context.latest_saved_policy_epoch = restored_epoch

        optimizer_state = payload.get("optimizer_state")
        context.state.optimizer_state = optimizer_state
        if optimizer_state:
            try:
                context.optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:  # pragma: no cover - mismatch rare but we log it
                logger.warning("Failed to load optimizer state from checkpoint: %s", exc)

        stopwatch_state = payload.get("stopwatch_state")
        context.state.stopwatch_state = stopwatch_state
        wall_time_baseline = 0.0
        if stopwatch_state:
            try:
                context.stopwatch.load_state(stopwatch_state, resume_running=True)
                wall_time_baseline = context.stopwatch.get_elapsed()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to restore stopwatch state: %s", exc)

        context.timing_baseline = {
            "agent_step": context.agent_step,
            "wall_time": wall_time_baseline,
        }

    # ------------------------------------------------------------------
    # Callback entry-points
    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self._distributed.should_checkpoint():
            return

        if epoch % self._config.epoch_interval != 0:
            return

        self._save_state()

    def on_training_complete(self) -> None:  # type: ignore[override]
        if not self._distributed.should_checkpoint():
            return

        self._save_state(force=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _save_state(self, *, force: bool = False) -> None:
        context = self.context
        current_epoch = context.epoch
        agent_step = context.agent_step

        if not force and self._config.epoch_interval and current_epoch % self._config.epoch_interval != 0:
            return

        try:
            context.state.stopwatch_state = context.stopwatch.save_state()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unable to capture stopwatch state: %s", exc)
            context.state.stopwatch_state = None

        context.state.optimizer_state = context.optimizer.state_dict()

        self._checkpoint_manager.save_trainer_state(
            context.optimizer,
            current_epoch,
            agent_step,
            stopwatch_state=context.state.stopwatch_state,
        )
