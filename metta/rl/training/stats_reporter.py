"""Statistics reporting and aggregation."""

import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Protocol
from uuid import UUID

import numpy as np
import torch
import torch.nn as nn
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.wandb.context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.model_analysis import compute_dormant_neuron_stats
from metta.rl.stats import accumulate_rollout_stats, compute_timing_stats, process_training_stats
from metta.rl.training.component import TrainerComponent
from metta.rl.utils import should_run
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class Timer(Protocol):
    def __call__(self, name: str) -> ContextManager[Any]: ...


def _to_scalar(value: Any) -> Optional[float]:
    """Convert supported numeric types to float, skipping non-scalars."""

    if isinstance(value, (int, float, bool, np.number)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None
    return None


def build_wandb_payload(
    processed_stats: dict[str, Any],
    curriculum_stats: dict[str, Any],
    timing_info: dict[str, Any],
    weight_stats: dict[str, Any],
    grad_stats: dict[str, float],
    system_stats: dict[str, Any],
    memory_stats: dict[str, Any],
    parameters: dict[str, Any],
    hyperparameters: dict[str, Any],
    evals: EvalRewardSummary,
    *,
    agent_step: int,
    epoch: int,
) -> dict[str, float]:
    """Create a flattened stats dictionary ready for wandb logging."""

    overview: dict[str, Any] = {
        "sps": timing_info.get("epoch_steps_per_second", 0.0),
        "steps_per_second": timing_info.get("steps_per_second", 0.0),
        "epoch_steps_per_second": timing_info.get("epoch_steps_per_second", 0.0),
        **processed_stats.get("overview", {}),
    }
    for category, score in evals.category_scores.items():
        overview[f"{category}_score"] = score
    if "reward" in overview:
        overview["reward_vs_total_time"] = overview["reward"]

    payload: dict[str, float] = {
        "metric/agent_step": float(agent_step),
        "metric/epoch": float(epoch),
        "metric/total_time": float(timing_info.get("wall_time", 0.0)),
        "metric/train_time": float(timing_info.get("train_time", 0.0)),
    }

    def _update(items: dict[str, Any], *, prefix: str = "") -> None:
        for key, value in items.items():
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            metric_key = f"{prefix}{key}" if prefix else key
            payload[metric_key] = scalar

    _update(overview, prefix="overview/")
    _update(processed_stats.get("losses_stats", {}), prefix="losses/")

    # Get experience stats and compute area under reward
    experience_stats = processed_stats.get("experience_stats", {})
    _update(experience_stats, prefix="experience/")

    _update(processed_stats.get("environment_stats", {}))
    _update(parameters, prefix="parameters/")
    _update(hyperparameters, prefix="hyperparameters/")

    eval_metrics = evals.to_wandb_metrics_format()
    for key, value in eval_metrics.items():
        scalar = _to_scalar(value)
        if scalar is None:
            continue
        payload[f"eval_{key}"] = scalar

    _update(system_stats)
    _update({f"trainer_memory/{k}": v for k, v in memory_stats.items()})
    _update(weight_stats)
    _update(grad_stats)
    _update(timing_info.get("timing_stats", {}))

    # Add curriculum stats (already has proper prefixes)
    _update(curriculum_stats)

    return payload


class StatsReporterConfig(Config):
    """Configuration for stats reporting."""

    report_to_wandb: bool = True
    report_to_stats_client: bool = True
    report_to_console: bool = True
    grad_mean_variance_interval: int = 50
    interval: int = 1
    """How often to report stats (in epochs)"""
    analyze_weights_interval: int = 0
    """How often to compute weight metrics (0 disables)."""
    dormant_neuron_threshold: float = 1e-6
    """Threshold for considering a neuron dormant based on mean absolute weight magnitude."""


class StatsReporterState(Config):
    """State for statistics tracking."""

    rollout_stats: dict = Field(default_factory=lambda: defaultdict(list))
    grad_stats: dict = Field(default_factory=dict)
    eval_scores: EvalRewardSummary = Field(default_factory=EvalRewardSummary)
    stats_run_id: Optional[UUID] = None
    area_under_reward: float = 0.0
    """Cumulative area under the reward curve"""


class NoOpStatsReporter(TrainerComponent):
    """No-op stats reporter for when stats are disabled."""

    def __init__(self):
        """Initialize no-op stats reporter."""
        # Create a minimal config for the no-op reporter
        config = StatsReporterConfig(report_to_wandb=False, report_to_stats_client=False, interval=999999)
        super().__init__(epoch_interval=config.interval)
        self.wandb_run = None
        self.stats_run_id = None

    def on_step(self, infos: list[dict[str, Any]]) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def on_training_complete(self) -> None:
        pass

    def on_failure(self) -> None:
        pass


class StatsReporter(TrainerComponent):
    """Aggregates and reports statistics to multiple backends."""

    @classmethod
    def from_config(
        cls,
        config: Optional[StatsReporterConfig],
        stats_client: Optional[StatsClient] = None,
        wandb_run: Optional[WandbRun] = None,
    ) -> TrainerComponent:
        """Create a StatsReporter from optional config, returning no-op if None."""
        if config is None:
            return NoOpStatsReporter()
        return cls(config=config, stats_client=stats_client, wandb_run=wandb_run)

    def __init__(
        self,
        config: StatsReporterConfig,
        stats_client: Optional[StatsClient] = None,
        wandb_run: Optional[WandbRun] = None,
    ):
        super().__init__(epoch_interval=config.interval)
        self._config = config
        self._stats_client = stats_client
        self._wandb_run = wandb_run
        self._state = StatsReporterState()
        self._latest_payload: dict[str, float] | None = None

        # Initialize stats run if client is available
        if self._stats_client and self._config.report_to_stats_client:
            self._initialize_stats_run()

    @property
    def wandb_run(self) -> WandbRun | None:
        return self._wandb_run

    @wandb_run.setter
    def wandb_run(self, run: WandbRun | None) -> None:
        self._wandb_run = run

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        context.stats_reporter = self

    def _initialize_stats_run(self) -> None:
        """Initialize stats run with the stats client."""
        if not self._stats_client:
            return

        # Extract wandb attributes with defaults
        name = url = "unknown"
        description: Optional[str] = None
        tags: Optional[list[str]] = None

        if self._wandb_run:
            name = self._wandb_run.name or name
            url = self._wandb_run.url
            if self._wandb_run.tags:
                tags = list(self._wandb_run.tags)
            description = self._wandb_run.notes

        try:
            result = self._stats_client.create_training_run(name=name, url=url, description=description, tags=tags)
            self._state.stats_run_id = result.id
            logger.info(f"Created stats run with ID: {self._state.stats_run_id}")
        except Exception as e:
            logger.warning(f"Failed to create training run: {e}", exc_info=True)

    @property
    def state(self) -> StatsReporterState:
        """Get the state for external access."""
        return self._state

    def process_rollout(self, raw_infos: list[dict[str, Any]]) -> None:
        if not raw_infos:
            return
        accumulate_rollout_stats(raw_infos, self._state.rollout_stats)

    def report_epoch(
        self,
        epoch: int,
        agent_step: int,
        losses_stats: dict[str, float],
        experience: Any,
        policy: Any,
        timer: Timer | None,
        trainer_cfg: Any,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        timing_context = timer("_process_stats") if callable(timer) else nullcontext()

        with timing_context:
            payload = self._build_wandb_payload(
                losses_stats=losses_stats,
                experience=experience,
                trainer_cfg=trainer_cfg,
                policy=policy,
                agent_step=agent_step,
                epoch=epoch,
                timer=timer,
                optimizer=optimizer,
            )

            # Update area under reward curve
            # Uses the current reward value and accumulates it over time
            if "experience/rewards" in payload:
                # Assuming each epoch represents a fixed time interval
                self._state.area_under_reward += payload["experience/rewards"]
                payload["experience/area_under_reward"] = self._state.area_under_reward

            if self._wandb_run and self._config.report_to_wandb and payload:
                self._wandb_run.log(payload, step=agent_step)

            self._latest_payload = payload.copy() if payload else None

            # Clear stats after processing
            self.clear_rollout_stats()
            self.clear_grad_stats()

    def update_eval_scores(self, scores: EvalRewardSummary) -> None:
        self._state.eval_scores = scores
        if self._context is not None:
            self.context.latest_eval_scores = scores

    def clear_rollout_stats(self) -> None:
        """Clear rollout statistics."""
        self._state.rollout_stats.clear()
        self._state.rollout_stats = defaultdict(list)

    def clear_grad_stats(self) -> None:
        """Clear gradient statistics."""
        self._state.grad_stats.clear()

    def update_grad_stats(self, grad_stats: dict[str, float]) -> None:
        self._state.grad_stats = grad_stats

    def create_epoch(
        self,
        run_id: UUID,
        start_epoch: int,
        end_epoch: int,
        attributes: dict[str, Any] | None = None,
    ) -> Optional[UUID]:
        if not self._stats_client or not self._config.report_to_stats_client:
            return None

        try:
            result = self._stats_client.create_epoch(
                run_id=run_id,
                start_training_epoch=start_epoch,
                end_training_epoch=end_epoch,
                attributes=attributes or {},
            )
            return result.id
        except Exception as e:
            logger.warning(f"Failed to create epoch: {e}", exc_info=True)
            return None

    def finalize(self, status: str = "completed") -> None:
        """Finalize stats reporting.

        Args:
            status: Final status of the training run
        """
        if self._stats_client and self._state.stats_run_id and self._config.report_to_stats_client:
            try:
                self._stats_client.update_training_run_status(self._state.stats_run_id, status)
                logger.info(f"Training run status updated to '{status}'")
            except Exception as e:
                logger.warning(f"Failed to update training run status: {e}", exc_info=True)
        self._latest_payload = None

    def on_step(self, infos: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Accumulate step infos.

        Args:
            infos: Step information from environment
        """
        self.accumulate_infos(infos)

    def get_latest_payload(self) -> Optional[dict[str, float]]:
        if self._latest_payload is None:
            return None
        return self._latest_payload.copy()

    def on_epoch_end(self, epoch: int) -> None:
        """Report stats at epoch end.

        Args:
        """
        ctx = self.context

        self.report_epoch(
            epoch=ctx.epoch,
            agent_step=ctx.agent_step,
            losses_stats=ctx.latest_losses_stats,
            experience=ctx.experience,
            policy=ctx.policy,
            timer=ctx.stopwatch,
            trainer_cfg=ctx.config,
            optimizer=ctx.optimizer,
        )

    def on_training_complete(self) -> None:
        """Handle training completion.

        Args:
        """
        self.finalize(status="completed")

    def on_failure(self) -> None:
        """Handle training failure.

        Args:
            trainer: The trainer instance
        """
        self.finalize(status="failed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def accumulate_infos(self, info: dict[str, Any] | list[dict[str, Any]] | None) -> None:
        """Accumulate rollout info dictionaries for later aggregation."""
        if not info:
            return
        if isinstance(info, list):
            filtered = [i for i in info if i]
            if not filtered:
                return
            self.process_rollout(filtered)
            return

        self.process_rollout([info])

    def _build_wandb_payload(
        self,
        *,
        losses_stats: dict[str, float],
        experience: Any,
        trainer_cfg: Any,
        policy: Any,
        agent_step: int,
        epoch: int,
        timer: Any,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Convert collected stats into a flat wandb payload."""

        if experience is None:
            return {}

        processed = process_training_stats(
            raw_stats=self._state.rollout_stats,
            losses_stats=losses_stats,
            experience=experience,
            trainer_config=trainer_cfg,
        )

        # Collect curriculum stats at epoch level (centralized)
        curriculum_stats = self._collect_curriculum_stats()

        timing_info = compute_timing_stats(timer=timer, agent_step=agent_step)
        self._normalize_steps_per_second(timing_info, agent_step)

        weight_stats = self._collect_weight_stats(policy=policy, epoch=epoch)
        dormant_stats = self._compute_dormant_neuron_stats(policy=policy)
        if dormant_stats:
            weight_stats.update(dormant_stats)
        system_stats = self._collect_system_stats()
        memory_stats = self._collect_memory_stats()
        parameters = self._collect_parameters(
            experience=experience,
            optimizer=optimizer,
            timing_info=timing_info,
        )
        hyperparameters = self._collect_hyperparameters(trainer_cfg=trainer_cfg, parameters=parameters)

        return build_wandb_payload(
            processed_stats=processed,
            curriculum_stats=curriculum_stats,
            timing_info=timing_info,
            weight_stats=weight_stats,
            grad_stats=self._state.grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters=hyperparameters,
            evals=self._state.eval_scores,
            agent_step=agent_step,
            epoch=epoch,
        )

    def _normalize_steps_per_second(self, timing_info: dict[str, Any], agent_step: int) -> None:
        """Adjust SPS to account for agent steps accumulated before a resume."""

        context = self._context
        if context is None:
            return

        baseline = getattr(context, "timing_baseline", None)
        if not isinstance(baseline, dict):
            return

        baseline_steps = baseline.get("agent_step", 0)
        baseline_wall = baseline.get("wall_time", 0.0)

        effective_elapsed = timing_info.get("wall_time", 0.0) - baseline_wall
        effective_steps = agent_step - baseline_steps

        if effective_elapsed <= 0 or effective_steps <= 0:
            return

        sps = effective_steps / effective_elapsed
        timing_info["steps_per_second"] = sps

        timing_stats = timing_info.get("timing_stats")
        if isinstance(timing_stats, dict):
            timing_stats["timing_cumulative/sps"] = sps

    def _collect_weight_stats(self, *, policy: Any, epoch: int) -> dict[str, float]:
        interval = self._config.analyze_weights_interval
        if not interval:
            policy_config = getattr(policy, "config", None)
            interval = getattr(policy_config, "analyze_weights_interval", 0) if policy_config else 0

        if not interval or not should_run(epoch, interval):
            return {}

        if not hasattr(policy, "compute_weight_metrics"):
            return {}

        weight_stats: dict[str, float] = {}
        try:
            for metrics in policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key == "name":
                        continue
                    scalar = _to_scalar(value)
                    if scalar is None:
                        continue
                    weight_stats[f"weights/{key}/{name}"] = scalar
        except Exception as exc:  # pragma: no cover - safeguard against model-specific failures
            logger.warning("Failed to compute weight metrics: %s", exc, exc_info=True)
        return weight_stats

    def _compute_dormant_neuron_stats(self, *, policy: Any) -> dict[str, float]:
        if not isinstance(policy, nn.Module):
            return {}
        threshold = getattr(self._config, "dormant_neuron_threshold", 1e-6)
        try:
            return compute_dormant_neuron_stats(policy, threshold=threshold)
        except Exception as exc:  # pragma: no cover - safeguard against model-specific failures
            logger.debug("Failed to compute dormant neuron stats: %s", exc, exc_info=True)
            return {}

    def _collect_system_stats(self) -> dict[str, Any]:
        system_monitor = getattr(self.context, "system_monitor", None)
        if system_monitor is None:
            return {}
        try:
            return system_monitor.stats()
        except Exception as exc:  # pragma: no cover
            logger.debug("System monitor stats failed: %s", exc, exc_info=True)
            return {}

    def _collect_memory_stats(self) -> dict[str, Any]:
        memory_monitor = getattr(self.context, "memory_monitor", None)
        if memory_monitor is None:
            return {}
        try:
            return memory_monitor.stats()
        except Exception as exc:  # pragma: no cover
            logger.debug("Memory monitor stats failed: %s", exc, exc_info=True)
            return {}

    def _collect_parameters(
        self,
        *,
        experience: Any,
        optimizer: torch.optim.Optimizer,
        timing_info: dict[str, Any],
    ) -> dict[str, Any]:
        learning_rate = getattr(self.context.config.optimizer, "learning_rate", 0)
        if optimizer and optimizer.param_groups:
            learning_rate = optimizer.param_groups[0].get("lr", learning_rate)

        parameters: dict[str, Any] = {
            "learning_rate": learning_rate,
            "epoch_steps": timing_info.get("epoch_steps", 0),
            "num_minibatches": getattr(experience, "num_minibatches", 0),
        }

        # Add ScheduleFree optimizer information
        if optimizer and optimizer.param_groups:
            param_group = optimizer.param_groups[0]
            is_schedulefree = "train_mode" in param_group

            if is_schedulefree:
                scheduled_lr = param_group.get("scheduled_lr")
                if scheduled_lr is not None:
                    parameters["schedulefree_scheduled_lr"] = scheduled_lr
                lr_max = param_group.get("lr_max")
                if lr_max is not None:
                    parameters["schedulefree_lr_max"] = lr_max

        return parameters

    def _collect_hyperparameters(
        self,
        *,
        trainer_cfg: Any,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        hyperparameters: dict[str, Any] = {}
        if "learning_rate" in parameters:
            hyperparameters["learning_rate"] = parameters["learning_rate"]

        optimizer_cfg = getattr(trainer_cfg, "optimizer", None)
        if optimizer_cfg:
            hyperparameters["optimizer_type"] = optimizer_cfg.type
            if "schedulefree" in optimizer_cfg.type:
                warmup_steps = getattr(optimizer_cfg, "warmup_steps", None)
                if warmup_steps is not None:
                    hyperparameters["schedulefree_warmup_steps"] = warmup_steps

        losses = getattr(trainer_cfg, "losses", None)
        loss_configs = getattr(losses, "loss_configs", {}) if losses else {}
        ppo_cfg = loss_configs.get("ppo") if isinstance(loss_configs, dict) else None
        if ppo_cfg is not None:
            for attr in (
                "clip_coef",
                "vf_clip_coef",
                "ent_coef",
                "l2_reg_loss_coef",
                "l2_init_loss_coef",
            ):
                value = getattr(ppo_cfg, attr, None)
                if value is None:
                    continue
                hyperparameters[f"ppo_{attr}"] = value
        return hyperparameters

    def _collect_curriculum_stats(self) -> dict[str, float]:
        """Collect curriculum statistics directly at epoch boundary.

        This replaces the batched per-environment logging approach with
        centralized collection, providing smooth and consistent logging.
        """
        if not self.context or not hasattr(self.context, "curriculum") or not self.context.curriculum:
            logger.info("No curriculum found in context, skipping curriculum stats collection")
            return {}

        curriculum = self.context.curriculum
        logger.info(f"Collecting curriculum stats from curriculum: {type(curriculum).__name__}")
        stats = {}

        # Get base curriculum stats
        curriculum_stats = curriculum.stats()
        logger.info(f"Got {len(curriculum_stats)} base curriculum stats")

        # Debug: Log sample of stat keys to see what we're getting
        sample_keys = list(curriculum_stats.keys())[:10]
        logger.info(f"Sample curriculum stat keys: {sample_keys}")

        # ===== GROUP A: Global Curriculum Stats =====
        if "num_completed" in curriculum_stats:
            stats["curriculum_stats/total_completions"] = float(curriculum_stats["num_completed"])

        if "num_evicted" in curriculum_stats:
            stats["curriculum_stats/num_evicted"] = float(curriculum_stats["num_evicted"])

        if "algorithm/mean_lp_score" in curriculum_stats:
            stats["curriculum_stats/mean_pool_lp_score"] = float(curriculum_stats["algorithm/mean_lp_score"])

        # Pool composition fractions
        total_pool_size = curriculum_stats.get("num_active_tasks", 0)
        if total_pool_size > 0:
            for key, value in curriculum_stats.items():
                if key.startswith("algorithm/pool_composition/"):
                    label = key.replace("algorithm/pool_composition/", "")
                    stats[f"curriculum_stats/pool_composition_fraction/{label}"] = float(value / total_pool_size)

        # Per-label aggregate evictions
        for key, value in curriculum_stats.items():
            if key.startswith("algorithm/eviction_counts/"):
                label = key.replace("algorithm/eviction_counts/", "")
                stats[f"curriculum_stats/per_label_aggregate_evictions/{label}"] = float(value)

        # ===== Per-Label LP Scores (ALWAYS collected for Gini calculations) =====
        # These are required for sampling_gini, so collect them unconditionally
        per_label_lp_stats = self._get_per_label_lp_stats(curriculum)
        logger.info(f"Collected {len(per_label_lp_stats)} per-label LP stats for Gini calculations")
        stats.update(per_label_lp_stats)

        # ===== GROUP C & D: Additional Troubleshooting Stats (if enabled) =====
        # NOTE: Per-label LP scores are now collected above unconditionally
        if self._should_enable_curriculum_troubleshooting():
            troubleshooting_stats = self._collect_curriculum_troubleshooting_stats(curriculum, curriculum_stats)
            logger.info(f"Collected {len(troubleshooting_stats)} additional troubleshooting stats")
            stats.update(troubleshooting_stats)

        # ===== GROUP B: Gini Coefficients =====
        # There are two types: per-label (terrain type distribution) and per-task (individual task inequality)

        # DIAGNOSTIC: Log what's available in stats
        logger.info(f"Stats keys before Gini calculations: {len(stats)} total")
        lp_prob_keys = [k for k in stats.keys() if "per_label_lp_probs" in k]
        logger.info(f"  - Found {len(lp_prob_keys)} per_label_lp_probs keys")

        # DIAGNOSTIC: Check rollout stats
        rollout_keys_sample = list(self._state.rollout_stats.keys())[:10]
        logger.info(f"Rollout stats keys (sample): {rollout_keys_sample}")

        # === PER-LABEL GINI COEFFICIENTS (Label-Aggregated Metrics) ===

        # 1. Pool composition gini - inequality in which terrain types are represented in task pool
        # Uses the fractions that were computed above (lines 628-634)
        pool_comp_fraction_keys = [
            k for k in stats.keys() if k.startswith("curriculum_stats/pool_composition_fraction/")
        ]
        pool_comp_fractions = [stats[k] for k in pool_comp_fraction_keys]
        if pool_comp_fractions:
            stats["curriculum_stats/pool_composition_gini"] = self._calculate_gini_coefficient(pool_comp_fractions)
            logger.info(
                f"Pool composition gini: {stats['curriculum_stats/pool_composition_gini']:.3f} "
                f"({len(pool_comp_fractions)} labels, fractions={pool_comp_fractions[:5]})"
            )
        else:
            logger.warning("No pool composition fractions found for gini calculation")

        # 2. Sampling gini - inequality in LP-based sampling probabilities by terrain type
        # Uses per_label_lp_probs which are the actual sampling probabilities aggregated by label
        sampling_prob_keys = [k for k in stats.keys() if k.startswith("curriculum_stats/per_label_lp_probs/")]
        sampling_probs = [stats[k] for k in sampling_prob_keys]
        if sampling_probs:
            stats["curriculum_stats/sampling_gini"] = self._calculate_gini_coefficient(sampling_probs)
            logger.info(
                f"Sampling gini: {stats['curriculum_stats/sampling_gini']:.3f} "
                f"({len(sampling_probs)} labels, probs={sampling_probs[:5]})"
            )
        else:
            logger.warning(
                f"No sampling probabilities found for gini calculation (found {len(sampling_prob_keys)} keys)"
            )

        # 3. Eviction gini - inequality in which terrain types are evicted THIS EPOCH
        # Get directly from curriculum (bypasses rollout_stats issues)
        per_label_evictions = self._get_per_epoch_evictions_from_curriculum(curriculum)
        logger.info(f"Got evictions from curriculum: {len(per_label_evictions)} labels")
        if per_label_evictions:
            epoch_eviction_counts = list(per_label_evictions.values())
            logger.info(f"  - Eviction counts: {epoch_eviction_counts[:5]}")
            stats["curriculum_stats/eviction_gini"] = self._calculate_gini_coefficient(epoch_eviction_counts)
            logger.info(
                f"Eviction gini: {stats['curriculum_stats/eviction_gini']:.3f} "
                f"({len(epoch_eviction_counts)} labels, counts={epoch_eviction_counts[:5]})"
            )
        else:
            logger.info("No evictions this epoch (expected early in training)")

        # 4. Per-epoch samples gini - inequality in this epoch's episode completions by terrain type
        # Get directly from curriculum algorithm (bypasses rollout_stats issues)
        per_label_samples = self._get_per_epoch_samples_from_curriculum(curriculum)
        logger.info(f"Got samples from curriculum: {len(per_label_samples)} labels")
        if per_label_samples:
            epoch_sample_counts = list(per_label_samples.values())
            logger.info(f"  - Sample counts: {epoch_sample_counts[:5]}")
            # Check if all values are equal (uniform distribution)
            if len(set(epoch_sample_counts)) > 1:
                stats["curriculum_stats/per_epoch_samples_gini"] = self._calculate_gini_coefficient(epoch_sample_counts)
                logger.info(
                    f"Per-epoch samples gini: {stats['curriculum_stats/per_epoch_samples_gini']:.3f} "
                    f"({len(epoch_sample_counts)} labels, counts={epoch_sample_counts[:5]})"
                )
            else:
                # All equal - legitimate zero Gini (perfect equality)
                stats["curriculum_stats/per_epoch_samples_gini"] = 0.0
                logger.info(f"Per-epoch samples are uniform (all = {epoch_sample_counts[0]}), gini=0.0 by definition")
        else:
            logger.info("No samples this epoch (expected if no episodes completed)")

        # === PER-TASK GINI COEFFICIENTS (Individual Task Inequality) ===
        # These are calculated by the algorithm and measure inequality across individual tasks

        # 5. Task sampling gini - inequality in how often individual tasks are sampled
        # Uses completion counts (empirical sampling distribution)
        if "algorithm/curriculum_gini/pool_occupancy" in curriculum_stats:
            gini_value = float(curriculum_stats["algorithm/curriculum_gini/pool_occupancy"])
            stats["curriculum_stats/task_sampling_gini"] = gini_value
            logger.info(f"Task sampling gini: {gini_value:.3f} (from algorithm/curriculum_gini/pool_occupancy)")
        else:
            logger.warning("algorithm/curriculum_gini/pool_occupancy not found in curriculum_stats")

        # 6. Task LP gini - inequality in learning progress scores across all individual tasks
        # Uses raw LP scores (before z-score normalization)
        if "algorithm/curriculum_gini/raw_lp_scores" in curriculum_stats:
            gini_value = float(curriculum_stats["algorithm/curriculum_gini/raw_lp_scores"])
            stats["curriculum_stats/task_lp_gini"] = gini_value
            logger.info(f"Task LP gini: {gini_value:.3f} (from algorithm/curriculum_gini/raw_lp_scores)")

            # Log debug stats if available
            if "algorithm/debug/raw_lp_unique_count" in curriculum_stats:
                unique_count = curriculum_stats["algorithm/debug/raw_lp_unique_count"]
                total_count = curriculum_stats.get("algorithm/debug/raw_lp_total_count", 0)
                logger.info(
                    f"  Raw LP: {unique_count:.0f} unique values out of {total_count:.0f} tasks "
                    f"(mean={curriculum_stats.get('algorithm/debug/raw_lp_mean', 0):.4f}, "
                    f"std={curriculum_stats.get('algorithm/debug/raw_lp_std', 0):.4f})"
                )
        else:
            logger.warning("algorithm/curriculum_gini/raw_lp_scores not found in curriculum_stats")

        # 7. Pass through all curriculum gini coefficients from the algorithm
        # These are comprehensive gini stats calculated at different pipeline stages:
        # - algorithm/curriculum_gini/pool_occupancy: task completion count inequality
        # - algorithm/curriculum_gini/raw_lp_scores: raw LP score inequality (task-level)
        # - algorithm/curriculum_gini/raw_lp_by_label: raw LP inequality aggregated by label
        # - algorithm/curriculum_gini/sampling_probs_by_label: final probability inequality by label
        # - algorithm/curriculum_gini/sampling_by_label: actual sampling inequality by label
        # - algorithm/curriculum_gini/zscored_lp_scores: z-scored LP inequality
        # - algorithm/curriculum_gini/evictions_by_label: eviction inequality by label
        # - algorithm/curriculum_gini/pool_composition_by_label: pool composition inequality
        gini_keys_found = []
        for key, value in curriculum_stats.items():
            if key.startswith("algorithm/curriculum_gini/"):
                # Pass through with the full key name (maintains alignment with task_dependency_simulator)
                stats[key] = float(value)
                gini_type = key.replace("algorithm/curriculum_gini/", "")
                gini_keys_found.append(gini_type)

        if gini_keys_found:
            logger.info(f"Passed through {len(gini_keys_found)} gini stats: {gini_keys_found[:5]}")

        # 8. Pass through all debug stats
        for key, value in curriculum_stats.items():
            if key.startswith("algorithm/debug/"):
                stats[key] = float(value)

        logger.info(f"Total curriculum stats collected: {len(stats)}")
        return stats

    @staticmethod
    def _reconstruct_dict_from_flattened_keys(rollout_stats: dict, prefix: str) -> dict[str, float]:
        """Reconstruct dictionary from flattened keys.

        When stats are emitted as nested dicts like:
            {"env_curriculum_stats/per_label_samples_this_epoch": {"label1": 2, "label2": 3}}

        They get flattened by unroll_nested_dict to:
            {"env_curriculum_stats/per_label_samples_this_epoch/label1": 2,
             "env_curriculum_stats/per_label_samples_this_epoch/label2": 3}

        This function reconstructs the original dict structure for Gini calculations.

        Args:
            rollout_stats: Dictionary with flattened keys
            prefix: The prefix to search for (e.g., "env_curriculum_stats/per_label_samples_this_epoch")

        Returns:
            Dictionary mapping label -> value
        """
        result = {}
        prefix_with_slash = f"{prefix}/"
        for key, value in rollout_stats.items():
            if key.startswith(prefix_with_slash):
                label = key[len(prefix_with_slash) :]
                # Value might be a list (accumulated across rollout steps) or scalar
                if isinstance(value, list):
                    result[label] = float(sum(value))  # Sum for count-based stats
                else:
                    result[label] = float(value)
        return result

    def _get_per_epoch_evictions_from_curriculum(self, curriculum: Any) -> dict[str, int]:
        """Get per-epoch evictions directly from curriculum.

        This bypasses rollout_stats and gets the data directly from the source,
        avoiding issues with info dict handling in the rollout loop.
        """
        if curriculum is None:
            return {}
        if not hasattr(curriculum, "get_and_reset_evictions_this_epoch"):
            return {}
        try:
            return curriculum.get_and_reset_evictions_this_epoch()
        except Exception as e:
            logger.warning(f"Failed to get evictions from curriculum: {e}")
            return {}

    def _get_per_epoch_samples_from_curriculum(self, curriculum: Any) -> dict[str, int]:
        """Get per-epoch sampling counts directly from curriculum algorithm.

        This bypasses rollout_stats and gets the data directly from the source,
        avoiding issues with info dict handling in the rollout loop.
        """
        if curriculum is None:
            return {}
        algorithm = getattr(curriculum, "_algorithm", None)
        if algorithm is None:
            return {}
        if not hasattr(algorithm, "get_and_reset_sampling_counts_this_epoch"):
            return {}
        try:
            return algorithm.get_and_reset_sampling_counts_this_epoch()
        except Exception as e:
            logger.warning(f"Failed to get sampling counts from curriculum: {e}")
            return {}

    def _should_enable_curriculum_troubleshooting(self) -> bool:
        """Check if curriculum troubleshooting logging is enabled.

        Checks the curriculum algorithm's hyperparameters for the
        show_curriculum_troubleshooting_logging flag.

        Context Access: self.context.curriculum is set in trainer from env._curriculum
        Flag Location: LearningProgressAlgorithmHypers.show_curriculum_troubleshooting_logging
        """
        if not self.context or not hasattr(self.context, "curriculum"):
            logger.info("Troubleshooting disabled: No context or curriculum")
            return False

        curriculum = self.context.curriculum
        if not curriculum or not hasattr(curriculum, "_algorithm"):
            logger.info("Troubleshooting disabled: No curriculum or algorithm")
            return False

        algorithm = curriculum._algorithm
        if not algorithm or not hasattr(algorithm, "hypers"):
            logger.info("Troubleshooting disabled: No algorithm or hypers")
            return False

        flag_value = getattr(algorithm.hypers, "show_curriculum_troubleshooting_logging", False)
        logger.info(f"Troubleshooting flag value: {flag_value}")
        return flag_value

    def _collect_curriculum_troubleshooting_stats(self, curriculum: Any, curriculum_stats: dict) -> dict[str, float]:
        """Collect detailed troubleshooting stats for curriculum debugging.

        Includes:
        - Tracked task dynamics (first 3 tasks)

        Note: Per-label LP scores are now collected unconditionally in _collect_curriculum_stats
        for Gini calculation purposes.
        """
        stats = {}

        # GROUP C: Tracked task dynamics
        # Get first 3 task IDs from the curriculum's task pool
        tracked_task_ids = self._get_tracked_task_ids(curriculum)
        logger.info(f"Tracked task IDs: {tracked_task_ids}")

        if tracked_task_ids:
            for i, task_id in enumerate(tracked_task_ids):
                lp_score = curriculum.get_task_lp_score(task_id)
                stats[f"curriculum_stats/tracked_task_lp_scores/task_{i}"] = float(lp_score)

                # Completion counts this epoch from accumulated info dicts
                completion_key = f"curriculum_stats/tracked_task_completions_this_epoch/task_{i}"
                if completion_key in self._state.rollout_stats:
                    stats[completion_key] = float(np.sum(self._state.rollout_stats[completion_key]))

        return stats

    def _get_tracked_task_ids(self, curriculum: Any) -> list[int]:
        """Get first 3 task IDs for detailed tracking."""
        # Get active tasks from curriculum (stored in _tasks dict)
        if not hasattr(curriculum, "_tasks"):
            logger.warning("Curriculum has no _tasks attribute")
            return []

        tasks = curriculum._tasks
        if not tasks:
            logger.info("Curriculum task pool is empty (early in training)")
            return []

        # Return first 3 task IDs (or fewer if pool is smaller)
        task_ids = list(tasks.keys())[:3]
        logger.info(f"Found {len(tasks)} total tasks, tracking first 3: {task_ids}")
        return task_ids

    def _get_per_label_lp_stats(self, curriculum: Any) -> dict[str, float]:
        """Get per-label LP scores from curriculum algorithm.

        Requires: Step 5 implementation (get_per_label_lp_scores method on algorithm)
        """
        stats = {}

        algorithm = getattr(curriculum, "_algorithm", None)
        if not algorithm:
            logger.info("No algorithm found in curriculum")
            return stats

        # Get task pool from curriculum (stored in _tasks dict)
        tasks = getattr(curriculum, "_tasks", None)
        if not tasks:
            logger.info("No tasks found in curriculum for per-label stats")
            return stats

        # Get per-label aggregated scores from algorithm (Step 5 provides this)
        if hasattr(algorithm, "get_per_label_lp_scores"):
            per_label_scores = algorithm.get_per_label_lp_scores(tasks)
            logger.info(f"Algorithm returned per-label scores for {len(per_label_scores)} labels")
            for label, score_dict in per_label_scores.items():
                stats[f"curriculum_stats/per_label_lp_scores/{label}"] = float(score_dict.get("raw", 0.0))
                stats[f"curriculum_stats/per_label_postzscored_lp_scores/{label}"] = float(
                    score_dict.get("postzscored", 0.0)
                )
                stats[f"curriculum_stats/per_label_lp_probs/{label}"] = float(score_dict.get("prob", 0.0))
        else:
            logger.warning("Algorithm does not have get_per_label_lp_scores method")

        return stats

    @staticmethod
    def _calculate_gini_coefficient(values: list[float]) -> float:
        """Calculate Gini coefficient for a distribution.

        Measures inequality in sampling across labels:
        - 0 = perfect equality (all labels sampled equally)
        - 1 = perfect inequality (all samples from one label)
        """
        if not values or len(values) == 0:
            return 0.0

        if sum(values) == 0:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
        total = sum(sorted_values)
        gini = (2.0 * cumsum) / (n * total) - (n + 1.0) / n

        return float(gini)
