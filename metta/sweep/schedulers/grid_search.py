"""Grid-search scheduler for categorical parameter sweeps.

This scheduler enumerates the Cartesian product of provided categorical
parameter choices and schedules training/eval jobs while respecting resource
constraints. It does not use an optimizer.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

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
    # Nested dict of categorical parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class GridSearchState:
    """State for grid search scheduling."""

    runs_in_training: set[str] = field(default_factory=set)
    runs_in_eval: set[str] = field(default_factory=set)
    runs_completed: set[str] = field(default_factory=set)
    # Index of next suggestion to dispatch
    next_index: int = 0
    # Mapping run_id -> suggestion dict
    in_progress_suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "runs_in_training": list(self.runs_in_training),
            "runs_in_eval": list(self.runs_in_eval),
            "runs_completed": list(self.runs_completed),
            "next_index": self.next_index,
            "in_progress_suggestions": self.in_progress_suggestions,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "GridSearchState":
        return cls(
            runs_in_training=set(data.get("runs_in_training", [])),
            runs_in_eval=set(data.get("runs_in_eval", [])),
            runs_completed=set(data.get("runs_completed", [])),
            next_index=int(data.get("next_index", 0)),
            in_progress_suggestions=dict(data.get("in_progress_suggestions", {})),
        )


class GridSearchScheduler:
    """Scheduler that enumerates a fixed grid of categorical suggestions."""

    def __init__(self, config: GridSearchSchedulerConfig, state: GridSearchState | None = None):
        self.config = config
        self.state = state or GridSearchState()
        # Precompute full grid suggestions
        dims = self._flatten_dims(config.parameters)
        self._dim_names: list[str] = list(dims.keys())
        self._grid: list[dict[str, Any]] = self._cartesian_product(dims)
        logger.info("[GridSearchScheduler] Initialized with grid size=%s", len(self._grid))

    # ---------- Scheduling API ----------
    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs: list[JobDefinition] = []
        run_status = {r.run_id: r.status for r in runs}

        # Schedule evals for any runs that completed training
        eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
        for run in eval_candidates:
            job = create_eval_job(
                run_id=run.run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                eval_entrypoint=self.config.eval_entrypoint,
                stats_server_uri=self.config.stats_server_uri,
                eval_overrides=self.config.eval_overrides,
            )
            jobs.append(job)
            self.state.runs_in_training.discard(run.run_id)
            self.state.runs_in_eval.add(run.run_id)
            logger.info("[GridSearchScheduler] Scheduling evaluation for %s", run.run_id)

        # Update state transitions for eval completions/failures
        for run_id in list(self.state.runs_in_eval):
            st = run_status.get(run_id)
            if st in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                self.state.runs_in_eval.discard(run_id)
                self.state.runs_completed.add(run_id)

        # Respect training capacity
        remaining_capacity = max(0, available_training_slots)
        if remaining_capacity <= 0:
            return jobs

        # Determine remaining suggestions to launch
        max_total = len(self._grid) if self.config.max_trials is None else min(self.config.max_trials, len(self._grid))
        already_created = len(runs)
        remaining_to_create = max_total - already_created
        if remaining_to_create <= 0:
            return jobs

        to_launch = min(remaining_capacity, remaining_to_create)
        base_trial_num = already_created

        # Launch next chunk of suggestions
        for i in range(to_launch):
            if self.state.next_index >= max_total:
                break
            suggestion = dict(self._grid[self.state.next_index])
            trial_num = base_trial_num + i + 1
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
            # Persist suggestion for store consumers
            job.metadata["sweep/suggestion"] = suggestion
            jobs.append(job)

            # Update state
            self.state.runs_in_training.add(run_id)
            self.state.in_progress_suggestions[run_id] = suggestion
            self.state.next_index += 1
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
            for name, val in zip(names, combo):
                s[name] = val
            suggestions.append(s)
        return suggestions

