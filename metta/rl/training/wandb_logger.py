"""Trainer component for logging metrics to wandb."""

from typing import Any, Dict

import wandb

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

        # Update curriculum environment epoch for visualization
        self._update_curriculum_epoch(epoch)

        # Check for task pool visualization data in latest rollout info
        self._log_task_pool_visualizations()

        self._wandb_run.log(payload)

    def on_training_complete(self) -> None:  # noqa: D401
        self._log_status("completed")

    def on_failure(self) -> None:  # noqa: D401
        self._log_status("failed")

    def _log_status(self, status: str) -> None:
        self._wandb_run.summary["training/status"] = status

    def _update_curriculum_epoch(self, epoch: int) -> None:
        """Update curriculum environments with current epoch for visualization."""
        try:
            # Access the training environment and update curriculum epoch
            env = getattr(self.context, "env", None)
            if env is not None:
                # For vectorized environments, access the driver environment
                driver_env = getattr(env, "driver_env", None)
                if driver_env is not None and hasattr(driver_env, "set_epoch"):
                    driver_env.set_epoch(epoch)
        except Exception:
            # Fail silently - visualization is optional
            pass

    def _log_task_pool_visualizations(self) -> None:
        """Log task pool visualization histograms to wandb."""
        try:
            # Access visualization data directly from the environment
            env = getattr(self.context, "env", None)
            if env is not None:
                # For vectorized environments, access the driver environment
                driver_env = getattr(env, "driver_env", None)
                if driver_env is not None and hasattr(driver_env, "get_latest_viz_data"):
                    viz_data = driver_env.get_latest_viz_data()
                    if viz_data:
                        self._log_histograms(viz_data)
        except Exception:
            # Fail silently - visualization is optional
            pass

    def _log_histograms(self, viz_data: Dict[str, Any]) -> None:
        """Log histogram data to wandb."""
        for hist_name, hist_data in viz_data.items():
            try:
                if hist_name in ["task_scores", "task_completions"]:
                    # Simple numpy array histograms
                    if len(hist_data) > 0:
                        self._wandb_run.log({f"curriculum_viz/{hist_name}": wandb.Histogram(hist_data)})

                elif hist_name == "label_counts":
                    # Bar chart for label counts
                    if "labels" in hist_data and "counts" in hist_data:
                        labels = hist_data["labels"]
                        counts = hist_data["counts"]
                        if len(labels) > 0 and len(counts) > 0:
                            # Create a simple bar chart data
                            bar_data = [[label, count] for label, count in zip(labels, counts, strict=False)]
                            table = wandb.Table(data=bar_data, columns=["label", "count"])
                            self._wandb_run.log(
                                {
                                    f"curriculum_viz/{hist_name}": wandb.plot.bar(
                                        table, "label", "count", title="Task Label Counts"
                                    )
                                }
                            )

                elif hist_name == "mean_scores_by_label":
                    # Bar chart for mean scores by label
                    if "labels" in hist_data and "means" in hist_data:
                        labels = hist_data["labels"]
                        means = hist_data["means"]
                        if len(labels) > 0 and len(means) > 0:
                            # Create a simple bar chart data
                            bar_data = [[label, mean] for label, mean in zip(labels, means, strict=False)]
                            table = wandb.Table(data=bar_data, columns=["label", "mean_score"])
                            self._wandb_run.log(
                                {
                                    f"curriculum_viz/{hist_name}": wandb.plot.bar(
                                        table, "label", "mean_score", title="Mean Scores by Label"
                                    )
                                }
                            )

            except Exception:
                # Individual histogram failures shouldn't break training
                continue
