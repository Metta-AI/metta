"""
Trainer component for logging GAMMA alignment metrics.

Integrates with Metta's training loop to compute and log alignment metrics.
"""

import logging
from typing import Any

import numpy as np

from metta.alignment.integration.gamma_evaluator import GAMMAEvaluator
from metta.alignment.integration.mettagrid_adapter import MettaGridAdapter
from metta.alignment.integration.trajectory_collector import TrajectoryCollector
from metta.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class GAMMALogger(TrainerComponent):
    """
    Logs GAMMA alignment metrics during training.

    This component:
    - Collects agent trajectories during episodes
    - Computes GAMMA metrics at epoch boundaries
    - Logs results to wandb and stats reporter

    Args:
        num_agents: Number of agents in the environment
        epoch_interval: How often to compute metrics (default: every 10 epochs)
        alpha: Dispersion penalty for GAMMA_α
        enabled: Whether to enable GAMMA logging (can be expensive)
    """

    def __init__(
        self,
        num_agents: int,
        epoch_interval: int = 10,
        alpha: float = 0.1,
        enabled: bool = True,
        collect_during_rollout: bool = True,
    ):
        """Initialize GAMMA logger."""
        super().__init__(epoch_interval=epoch_interval)
        self.num_agents = num_agents
        self.alpha = alpha
        self.enabled = enabled
        self.collect_during_rollout = collect_during_rollout

        if self.enabled:
            self.collector = TrajectoryCollector(num_agents=num_agents)
            self.evaluator = GAMMAEvaluator(alpha=alpha)
            self.adapter = MettaGridAdapter(grid_to_continuous_scale=1.0)
            self._episode_count = 0
            self._gamma_history: list[float] = []
            self._step_count = 0
            self._last_positions = None

    def on_rollout_start(self) -> None:
        """Called at the start of rollout phase."""
        if self.enabled and self.collect_during_rollout:
            self.collector.reset()
            self._step_count = 0
            self._last_positions = None

    def collect_step_data(self, dt: float = 0.1) -> None:
        """
        Collect trajectory data for current step.

        This should be called during the rollout loop with access to the environment.

        Args:
            dt: Time step size
        """
        if not self.enabled or not self.collect_during_rollout:
            return

        try:
            # Access environment from context
            env = self.context.env

            # Get the underlying MettaGrid environment
            # Handle vectorized environments
            if hasattr(env, "_vecenv"):
                # For vectorized envs, we'll collect from the first environment
                # This is a simplification - full implementation would collect from all
                if hasattr(env._vecenv, "envs") and len(env._vecenv.envs) > 0:
                    mettagrid_env = env._vecenv.envs[0]
                else:
                    logger.debug("Cannot access MettaGrid environment from vectorized wrapper")
                    return
            else:
                mettagrid_env = env

            # Extract agent positions using adapter
            positions = self.adapter.extract_agent_positions(mettagrid_env)

            # Compute task directions (toward resources as default)
            task_directions = self.adapter.compute_task_directions_to_resources(
                mettagrid_env, resource_types=["generator", "converter", "altar"]
            )

            # Record step
            self.collector.record_step(positions=positions, task_directions=task_directions, dt=dt)

            self._step_count += 1

        except Exception as e:
            logger.debug(f"Failed to collect trajectory data: {e}")

    def on_step(self, infos: dict[str, Any] | list[dict[str, Any]]) -> None:
        """
        Called after each environment step.

        Args:
            infos: Step information from environment
        """
        if self.enabled and self.collect_during_rollout:
            self.collect_step_data(dt=0.1)

    def on_epoch_end(self, epoch: int) -> None:
        """
        Compute and log GAMMA metrics at epoch end.

        Args:
            epoch: Current epoch number
        """
        if not self.enabled:
            return

        try:
            # Get collected trajectories
            trajectories = self.collector.get_trajectories()
            dt = self.collector.get_mean_dt()

            # Check if we have any data
            if len(trajectories) == 0 or all(len(t["positions"]) == 0 for t in trajectories):
                logger.debug(f"No trajectory data collected for epoch {epoch}")
                return

            # Compute GAMMA metrics
            results = self.evaluator.evaluate(trajectories, dt=dt, goals=None)

            # Store history
            self._gamma_history.append(results["GAMMA"])
            self._episode_count += 1

            # Log to console
            logger.info(
                f"GAMMA Alignment - Epoch {epoch}: "
                f"GAMMA={results['GAMMA']:.3f}, "
                f"IAM_mean={results['IAM_mean']:.3f}, "
                f"CV={results['CV']:.3f}"
            )

            # Format for wandb
            wandb_metrics = self.evaluator.format_for_wandb(results, prefix="alignment")

            # Add to context for other components to log
            if hasattr(self.context, "alignment_metrics"):
                self.context.alignment_metrics.update(wandb_metrics)
            else:
                self.context.alignment_metrics = wandb_metrics

        except Exception as e:
            logger.warning(f"Failed to compute GAMMA metrics: {e}", exc_info=True)

    def get_summary_stats(self) -> dict[str, float]:
        """
        Get summary statistics for the training run.

        Returns:
            Dictionary with summary metrics
        """
        if not self.enabled or len(self._gamma_history) == 0:
            return {}

        return {
            "alignment/GAMMA_mean": float(np.mean(self._gamma_history)),
            "alignment/GAMMA_std": float(np.std(self._gamma_history)),
            "alignment/GAMMA_final": self._gamma_history[-1],
            "alignment/episodes_evaluated": self._episode_count,
        }
