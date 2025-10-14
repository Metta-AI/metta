"""Grid-search scheduler for categorical parameter sweeps.

Minimal design: enumerate the Cartesian product of categorical parameters,
schedule evals for runs with training complete, and issue new training jobs
for the next unseen combinations while respecting slot limits. No optimizer
or complex state; we infer progress from the runs' recorded suggestions.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.utils import create_eval_job, create_training_job, generate_run_id
from metta.sweep.core import CategoricalParameterConfig
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class GridSearchSchedulerConfig(Config):
    """Configuration for the grid-search scheduler.

    Provide nested categorical parameters via `parameters`. Values may be
    `CategoricalParameterConfig` or plain lists of choices.
    """

    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    stats_server_uri: str | None = None
    gpus: int = 1
    nodes: int = 1
    experiment_id: str = "grid_search"
    # Optional cap; if None, runs through entire grid
    max_trials: int | None = None
    # Max concurrent evaluations; None means unlimited
    max_concurrent_evals: Optional[int] = None
    # Nested dict of categorical parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GridSearchScheduler:
    """Scheduler that enumerates a fixed grid of categorical suggestions."""

    def __init__(self, config: GridSearchSchedulerConfig):
        self.config = config
        # Precompute full grid suggestions
        dims = self._flatten_dims(config.parameters)
        self._dim_names: list[str] = list(dims.keys())
        self._grid: list[dict[str, Any]] = self._cartesian_product(dims)
        logger.info("[GridSearchScheduler] Initialized with grid size=%s", len(self._grid))

    # ---------- Scheduling API ----------
    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs: list[JobDefinition] = []

        # 1) Schedule evals for any runs with training done (throttled)
        in_eval = [r for r in runs if r.status == JobStatus.IN_EVAL]
        eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]

        eval_capacity: Optional[int] = None
        if self.config.max_concurrent_evals is not None:
            eval_capacity = max(0, self.config.max_concurrent_evals - len(in_eval))

        if eval_candidates:
            to_schedule = eval_candidates if eval_capacity is None else eval_candidates[:eval_capacity]
            for run in to_schedule:
                job = create_eval_job(
                    run_id=run.run_id,
                    experiment_id=self.config.experiment_id,
                    recipe_module=self.config.recipe_module,
                    eval_entrypoint=self.config.eval_entrypoint,
                    stats_server_uri=self.config.stats_server_uri,
                    eval_overrides=self.config.eval_overrides,
                )
                jobs.append(job)
                logger.info("[GridSearchScheduler] Scheduling evaluation for %s", run.run_id)

        # 2) Respect training capacity
        remaining_capacity = max(0, available_training_slots)
        if remaining_capacity <= 0:
            return jobs

        # Grid search does not block training on eval backlog; proceed to schedule training as slots allow.

        # 3) Determine remaining suggestions to launch based on used suggestions
        target_total = (
            len(self._grid) if self.config.max_trials is None else min(self.config.max_trials, len(self._grid))
        )
        used_keys = self._collect_used_suggestion_keys(runs)
        used_count = min(len(used_keys), target_total)
        remaining_to_create = target_total - used_count
        if remaining_to_create <= 0:
            return jobs

        to_launch = min(remaining_capacity, remaining_to_create)
        base_trial_num = len(runs)

        # 4) Launch next unseen suggestions in grid order
        launched = 0
        for suggestion in self._grid:
            if launched >= to_launch:
                break
            if self._suggestion_key(suggestion) in used_keys:
                continue
            trial_num = base_trial_num + launched + 1
            run_id = generate_run_id(self.config.experiment_id, trial_num)

            merged_overrides = dict(self.config.train_overrides)
            merged_overrides.update(suggestion)
            job = create_training_job(
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                train_entrypoint=self.config.train_entrypoint,
                gpus=self.config.gpus,
                nodes=self.config.nodes,
                stats_server_uri=self.config.stats_server_uri,
                train_overrides=merged_overrides,
            )
            job.metadata["sweep/suggestion"] = suggestion
            jobs.append(job)
            launched += 1
            logger.info("[GridSearchScheduler] Scheduling training %s", run_id)

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        finished = [r for r in runs if r.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE)]
        target = len(self._grid) if self.config.max_trials is None else min(self.config.max_trials, len(self._grid))
        is_done = len(finished) >= target
        if is_done:
            completed = [r for r in runs if r.status == JobStatus.COMPLETED]
            logger.info(
                "[GridSearchScheduler] Experiment complete! %s/%s trials finished (%s successful)",
                len(finished),
                target,
                len(completed),
            )
        return is_done

    # ---------- Helpers ----------
    def _extract_suggestion(self, run: RunInfo) -> dict[str, Any] | None:
        if run.summary and isinstance(run.summary, dict):
            sg = run.summary.get("sweep/suggestion")
            if isinstance(sg, dict):
                return dict(sg)
        return None

    def _suggestion_key(self, suggestion: dict[str, Any]) -> Tuple[Any, ...]:
        # Build a key tuple ordered by dimension names
        return tuple(suggestion.get(name) for name in self._dim_names)

    def _collect_used_suggestion_keys(self, runs: list[RunInfo]) -> set[Tuple[Any, ...]]:
        keys: set[Tuple[Any, ...]] = set()
        for r in runs:
            sg = self._extract_suggestion(r)
            if not sg:
                continue
            try:
                key = self._suggestion_key(sg)
            except Exception:
                continue
            keys.add(key)
        return keys

    def _flatten_dims(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, List[Any]]:
        """Extract flattened categorical dimensions from nested params.

        Returns a mapping from dotted parameter name to list of allowed choices.
        """
        dims: dict[str, list[Any]] = {}
        for key, value in params.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, CategoricalParameterConfig):
                if not value.choices:
                    raise ValueError(f"Categorical parameter '{full}' must have at least one choice")
                dims[full] = list(value.choices)
            elif isinstance(value, list):
                if not value:
                    raise ValueError(f"Categorical parameter '{full}' must have at least one choice")
                dims[full] = list(value)
            elif isinstance(value, dict):
                dims.update(self._flatten_dims(value, full))
            else:
                raise TypeError(
                    f"GridSearchScheduler only supports categorical parameters (lists or CategoricalParameterConfig). "
                    f"Got unsupported type at '{full}': {type(value)}"
                )
        if not dims:
            raise ValueError("No categorical parameters provided for grid search")
        return dims

    def _cartesian_product(self, dims: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Build the Cartesian product of dimensions as a list of suggestion dicts."""
        names = list(dims.keys())
        values: Iterable[Tuple[Any, ...]] = itertools.product(*(dims[name] for name in names))
        suggestions: list[dict[str, Any]] = []
        for combo in values:
            s: dict[str, Any] = {}
            for name, val in zip(names, combo, strict=False):
                s[name] = val
            suggestions.append(s)
        return suggestions
