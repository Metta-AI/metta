"""Trainer component for logging metrics to wandb."""

import time
from typing import Dict

from metta.common.wandb.context import WandbRun
from metta.rl.training import TrainerComponent
from metta.rl.wandb import log_model_parameters, setup_wandb_metrics


class WandbLogger(TrainerComponent):
    """Logs core training metrics to wandb.

    Epoch-boundary logs are the canonical training metrics. We also emit a small
    periodic "heartbeat" log during rollouts to keep long/slow epochs from
    looking stale to sweep controllers that rely on WandB summary timestamps.
    """

    def __init__(
        self,
        wandb_run: WandbRun,
        *,
        epoch_interval: int = 1,
        step_interval: int = 50_000,
        heartbeat_interval_s: float = 120.0,
    ):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._wandb_run = wandb_run
        self._last_agent_step = 0
        # Track cumulative elapsed times to compute per-epoch deltas robustly
        self._prev_elapsed: Dict[str, float] = {}
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._last_heartbeat_time = 0.0
        self._last_heartbeat_step = 0

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        setup_wandb_metrics(self._wandb_run)
        log_model_parameters(self.context.policy, self._wandb_run)
        # Emit an initial timestamped log so newly-started runs show up as "alive"
        # even before the first epoch completes.
        self._wandb_run.log(
            {
                "metric/agent_step": float(self.context.agent_step),
                "metric/epoch": float(self.context.epoch),
            }
        )
        self._last_heartbeat_time = time.monotonic()
        self._last_heartbeat_step = int(self.context.agent_step)

    def on_step(self, infos: list[dict]) -> None:  # type: ignore[override]
        now = time.monotonic()
        if now - self._last_heartbeat_time < self._heartbeat_interval_s:
            return

        agent_step = int(self.context.agent_step)
        step_delta = agent_step - self._last_heartbeat_step
        time_delta = max(1e-6, now - self._last_heartbeat_time)
        sps = float(step_delta / time_delta) if step_delta > 0 else 0.0

        self._wandb_run.log(
            {
                "metric/agent_step": float(agent_step),
                "metric/epoch": float(self.context.epoch),
                "overview/steps_per_second": sps,
            }
        )

        self._last_heartbeat_time = now
        self._last_heartbeat_step = agent_step

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

        for timer_name, current_elapsed in elapsed.items():
            if timer_name.startswith("_rollout."):
                prev_elapsed = self._prev_elapsed.get(timer_name, 0.0)
                delta = max(0.0, float(current_elapsed) - float(prev_elapsed))

                # Convert "_rollout.env_wait" to "metric/rollout_env_wait_time"
                metric_name = f"metric/rollout_{timer_name[9:].replace('.', '_')}_time"
                payload[metric_name] = delta

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
