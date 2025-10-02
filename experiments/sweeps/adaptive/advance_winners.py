"""Advance winners: resume top-performers in a WandB group to a target step count.

This experiment selects the top N% runs in a WandB `group` by a chosen
`metric` and resumes training for each until `trainer.total_timesteps == target`.

It uses the lightweight AdaptiveController + a custom scheduler, and dispatches
local training jobs (via `uv run ./tools/run.py ...`). No Bayesian tooling.

Example:
  uv run ./tools/run.py experiments.sweeps.adaptive.advance_winners.advance \
    group=ak.vit_sweep.10012346 perc=10 target=5000000000 \
    metric=evaluator/eval_arena/score
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, List

from metta.adaptive import AdaptiveConfig, AdaptiveController
from metta.adaptive.dispatcher import LocalDispatcher
from metta.adaptive.models import JobDefinition, JobTypes, RunInfo
from metta.adaptive.protocols import ExperimentScheduler
from metta.adaptive.stores import WandbStore
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


@dataclass
class AdvanceWinnersConfig:
    group: str
    metric: str
    target_timesteps: int
    percentile: float  # e.g., 10 for top 10%
    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped"
    train_entrypoint: str = "train"
    gpus: int = 1
    nodes: int = 1


class AdvanceWinnersScheduler(ExperimentScheduler):
    """Scheduler that resumes the top-`percentile` runs until `target_timesteps`.

    - Selects winners once based on the provided `metric`.
    - Schedules LAUNCH_TRAINING jobs for winners whose current_steps < target.
    - Uses the original run_id so TrainTool resumes the same run directory/state.
    - Sets override `trainer.total_timesteps = target` so training extends to target.
    - Stops when all winners have reached the target.
    """

    def __init__(self, cfg: AdvanceWinnersConfig):
        self.cfg = cfg
        self._selected_run_ids: set[str] | None = None
        self._scheduled_run_ids: set[str] = set()

    def _select_winners(self, runs: List[RunInfo]) -> List[RunInfo]:
        # Filter runs that have the metric present
        with_scores: List[RunInfo] = []
        for r in runs:
            summary = r.summary or {}
            score = summary.get(self.cfg.metric)
            if score is None:
                continue
            try:
                float(score)
            except Exception:
                continue
            with_scores.append(r)

        if not with_scores:
            return []

        # Sort descending by metric
        with_scores.sort(key=lambda rr: float((rr.summary or {}).get(self.cfg.metric, float("-inf"))), reverse=True)

        # Determine how many to keep by percentile
        perc = self.cfg.percentile
        perc = perc * 100.0 if 0.0 < perc <= 1.0 else perc  # allow 0..1 or 0..100 inputs
        k = max(1, ceil(len(with_scores) * (perc / 100.0)))
        return with_scores[:k]

    def _ensure_selection(self, runs: List[RunInfo]) -> None:
        if self._selected_run_ids is not None:
            return
        winners = self._select_winners(runs)
        self._selected_run_ids = {w.run_id for w in winners}

    def schedule(self, runs: List[RunInfo], available_training_slots: int) -> List[JobDefinition]:
        self._ensure_selection(runs)
        selected = self._selected_run_ids or set()
        if not selected or available_training_slots <= 0:
            return []

        # Build a map for quick lookup
        run_map = {r.run_id: r for r in runs if r.run_id in selected}

        jobs: List[JobDefinition] = []
        for run_id in selected:
            if len(jobs) >= available_training_slots:
                break

            r = run_map.get(run_id)
            if r is None:
                continue

            # Skip if already at/over target
            if r.current_steps is not None and int(r.current_steps) >= int(self.cfg.target_timesteps):
                continue

            # Skip if we already scheduled this run in a prior cycle
            if run_id in self._scheduled_run_ids:
                continue

            # Skip if it appears actively training
            if r.has_started_training and not r.has_completed_training:
                continue

            # Create a training job that resumes the same run_id and extends target timesteps
            job = JobDefinition(
                run_id=run_id,
                cmd=f"{self.cfg.recipe_module}.{self.cfg.train_entrypoint}",
                gpus=self.cfg.gpus,
                nodes=self.cfg.nodes,
                args={
                    "run": run_id,  # critical: keep same run_id to resume
                    "group": self.cfg.group,
                },
                overrides={
                    "trainer.total_timesteps": int(self.cfg.target_timesteps),
                },
                type=JobTypes.LAUNCH_TRAINING,
                metadata={
                    "advance_winners/metric": self.cfg.metric,
                    "advance_winners/target": int(self.cfg.target_timesteps),
                },
            )

            jobs.append(job)
            self._scheduled_run_ids.add(run_id)

        return jobs

    def is_experiment_complete(self, runs: List[RunInfo]) -> bool:
        self._ensure_selection(runs)
        selected = self._selected_run_ids or set()
        if not selected:
            # Nothing to do
            return True

        for r in runs:
            if r.run_id not in selected:
                continue
            # Continue until every selected run has reached target
            if r.current_steps is None or int(r.current_steps) < int(self.cfg.target_timesteps):
                return False
        return True


def advance(
    *,
    group: str,
    perc: float,
    target: int,
    metric: str,
    # Optional knobs
    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped",
    train_entrypoint: str = "train",
    max_parallel: int = 4,
    monitoring_interval: int = 60,
) -> None:
    """Advance the top-`perc` runs in `group` to `target` timesteps.

    Args:
        group: WandB group containing prior runs
        perc: Percentile of best runs to resume (e.g., 10 for top 10%). 0..1 also accepted
        target: New absolute `trainer.total_timesteps` to reach
        metric: Name of the score in run.summary used for ranking
        recipe_module: Training recipe module to use when resuming
        train_entrypoint: Training entrypoint function within the recipe module
        max_parallel: Max concurrent training jobs
        monitoring_interval: Seconds between controller polling cycles
    """

    # Store: use default entity/project from repo constants
    store = WandbStore(entity=METTA_WANDB_ENTITY, project=METTA_WANDB_PROJECT)

    # Simple local dispatcher by default
    dispatcher = LocalDispatcher(capture_output=True)

    # Custom scheduler configured for this advance
    sched = AdvanceWinnersScheduler(
        AdvanceWinnersConfig(
            group=group,
            metric=metric,
            target_timesteps=int(target),
            percentile=float(perc),
            recipe_module=recipe_module,
            train_entrypoint=train_entrypoint,
        )
    )

    controller = AdaptiveController(
        experiment_id=group,
        scheduler=sched,
        dispatcher=dispatcher,
        store=store,
        config=AdaptiveConfig(
            max_parallel=max_parallel,
            monitoring_interval=monitoring_interval,
            resume=True,  # fetch immediately on first cycle
        ),
    )

    controller.run()

