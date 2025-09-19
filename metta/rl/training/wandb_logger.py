"""Trainer component for logging metrics to wandb."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from metta.common.wandb.context import WandbRun
from metta.rl.training.component import TrainerComponent
from metta.rl.wandb import log_model_parameters, setup_wandb_metrics

if TYPE_CHECKING:
    from metta.rl.trainer import Trainer


class WandbLoggerComponent(TrainerComponent):
    """Logs core training metrics to wandb at epoch boundaries."""

    def __init__(self, wandb_run: WandbRun, epoch_interval: int = 1):
        super().__init__(epoch_interval=epoch_interval)
        self._wandb_run = wandb_run
        self._last_agent_step = 0

    def register(self, trainer: "Trainer") -> None:
        super().register(trainer)
        setup_wandb_metrics(self._wandb_run)
        log_model_parameters(trainer._policy, self._wandb_run)

    def on_epoch_end(self, epoch: int) -> None:  # noqa: D401 - documented in base class
        trainer = self._trainer

        payload: Dict[str, float] = {
            "metric/agent_step": float(trainer._agent_step),
            "metric/epoch": float(trainer._epoch),
            "metric/train_time": float(trainer.timer.get_last_elapsed("_train")),
            "metric/rollout_time": float(trainer.timer.get_last_elapsed("_rollout")),
            "metric/stats_time": float(trainer.timer.get_last_elapsed("_process_stats")),
        }

        total_time = payload["metric/train_time"] + payload["metric/rollout_time"] + payload["metric/stats_time"]
        steps_delta = trainer._agent_step - self._last_agent_step
        if total_time > 0 and steps_delta > 0:
            payload["overview/steps_per_second"] = float(steps_delta / total_time)

        self._last_agent_step = trainer._agent_step

        for key, value in getattr(trainer, "latest_losses_stats", {}).items():
            metric_key = key if "/" in key else f"loss/{key}"
            payload[metric_key] = float(value)

        self._wandb_run.log(payload)

    def on_training_complete(self) -> None:  # noqa: D401
        self._log_status("completed")

    def on_failure(self) -> None:  # noqa: D401
        self._log_status("failed")

    def _log_status(self, status: str) -> None:
        self._wandb_run.summary["training/status"] = status
