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
        # Track cumulative elapsed times to compute per-epoch deltas robustly
        self._prev_elapsed: Dict[str, float] = {}

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

        # Add rollout breakdown metrics as per-epoch deltas of cumulative elapsed time
        elapsed = context.stopwatch.get_all_elapsed()

        def _delta(timer_name: str) -> float:
            cur = float(elapsed.get(timer_name, 0.0))
            prev = float(self._prev_elapsed.get(timer_name, 0.0))
            return max(0.0, cur - prev)

        env_wait = _delta("_rollout.env_wait")
        td_prep = _delta("_rollout.td_prep")
        inference = _delta("_rollout.inference")
        send = _delta("_rollout.send")

        payload.update(
            {
                "metric/rollout_env_wait_time": env_wait,
                "metric/rollout_td_prep_time": td_prep,
                "metric/rollout_inference_time": inference,
                "metric/rollout_send_time": send,
            }
        )

        total_time = payload["metric/train_time"] + payload["metric/rollout_time"] + payload["metric/stats_time"]
        steps_delta = context.agent_step - self._last_agent_step
        if total_time > 0 and steps_delta > 0:
            payload["overview/steps_per_second"] = float(steps_delta / total_time)

        self._last_agent_step = context.agent_step
        # Update baseline after computing deltas
        for k, v in elapsed.items():
            self._prev_elapsed[k] = float(v)

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
