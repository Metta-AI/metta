"""Trainer component for logging metrics to wandb."""

from typing import Dict

from metta.common.wandb.context import WandbRun
from metta.rl.training import TrainerComponent
from metta.rl.wandb import log_model_parameters, setup_wandb_metrics


class WandbLogger(TrainerComponent):
    """Logs core training metrics to wandb at epoch boundaries."""

    def __init__(self, wandb_run: WandbRun, epoch_interval: int = 1):
        super().__init__(epoch_interval=epoch_interval)
        self._wandb_run = wandb_run
        self._last_agent_step = 0

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        setup_wandb_metrics(self._wandb_run)
        log_model_parameters(self.context.policy, self._wandb_run)

    def on_epoch_end(self, epoch: int) -> None:  # noqa: D401 - documented in base class
        context = self.context
        payload: Dict[str, float] = {
            "metric/agent_step": float(context.agent_step),
            "metric/epoch": float(context.epoch),
            "metric/train_time": float(context.stopwatch.get_last_elapsed("_train")),
            "metric/rollout_time": float(context.stopwatch.get_last_elapsed("_rollout")),
            "metric/stats_time": float(context.stopwatch.get_last_elapsed("_process_stats")),
        }

        total_time = payload["metric/train_time"] + payload["metric/rollout_time"] + payload["metric/stats_time"]
        steps_delta = context.agent_step - self._last_agent_step
        if total_time > 0 and steps_delta > 0:
            payload["overview/steps_per_second"] = float(steps_delta / total_time)

        self._last_agent_step = context.agent_step

        for key, value in context.latest_losses_stats.items():
            metric_key = key if "/" in key else f"loss/{key}"
            payload[metric_key] = float(value)

        self._wandb_run.log(payload)

    def on_training_complete(self) -> None:  # noqa: D401
        self._log_status("completed")

    def on_failure(self) -> None:  # noqa: D401
        self._log_status("failed")

    def _log_status(self, status: str) -> None:
        self._wandb_run.summary["training/status"] = status
