"""Tests for AsyncCappedOptimizingScheduler behavior."""

from datetime import datetime, timezone
from typing import Any

from metta.adaptive.models import JobStatus, JobTypes, RunInfo
from metta.adaptive.run_phase import RunPhaseManager
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.async_capped import (
    AsyncCappedOptimizingScheduler,
    AsyncCappedSchedulerConfig,
)


class MockStore:
    """Mock store that records update calls."""

    def __init__(self):
        self.summaries: dict[str, dict[str, Any]] = {}

    def update_run_summary(self, run_id: str, data: dict[str, Any]) -> bool:
        if run_id not in self.summaries:
            self.summaries[run_id] = {}
        self.summaries[run_id].update(data)
        return True


def _basic_protein_config() -> ProteinConfig:
    return ProteinConfig(
        metric="test_metric",
        goal="maximize",
        parameters={
            "lr": ParameterConfig(
                min=0.001,
                max=0.01,
                distribution="log_normal",
                mean=0.003,
                scale="auto",
            )
        },
    )


class TestAsyncCappedOptimizingScheduler:
    def test_training_fills_slots(self):
        config = AsyncCappedSchedulerConfig(
            max_trials=5,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_async",
            protein_config=_basic_protein_config(),
            max_concurrent_evals=1,
        )
        store = MockStore()
        phase_manager = RunPhaseManager(store)
        scheduler = AsyncCappedOptimizingScheduler(config, phase_manager)

        # Stub optimizer to avoid importing heavy dependencies
        scheduler.optimizer.suggest = lambda observations, n_suggestions=1: [
            {"lr": 0.003} for _ in range(n_suggestions)
        ]  # type: ignore

        jobs = scheduler.schedule([], available_training_slots=3)

        assert len(jobs) == 3
        assert all(job.type == JobTypes.LAUNCH_TRAINING for job in jobs)

    def test_eval_capped_one_at_a_time(self):
        config = AsyncCappedSchedulerConfig(
            max_trials=10,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_async",
            protein_config=_basic_protein_config(),
            max_concurrent_evals=1,
        )
        store = MockStore()
        phase_manager = RunPhaseManager(store)
        scheduler = AsyncCappedOptimizingScheduler(config, phase_manager)
        # Stub optimizer
        scheduler.optimizer.suggest = lambda observations, n_suggestions=1: [
            {"lr": 0.003} for _ in range(n_suggestions)
        ]  # type: ignore

        now = datetime.now(timezone.utc)
        runs = [
            RunInfo(
                run_id="r1",
                has_started_training=True,
                has_completed_training=True,
                has_failed=False,
                created_at=now,
                last_updated_at=now,
                summary={},  # No sweep/eval_started
            ),
            RunInfo(
                run_id="r2",
                has_started_training=True,
                has_completed_training=True,
                has_failed=False,
                created_at=now,
                last_updated_at=now,
                summary={},  # No sweep/eval_started
            ),
        ]

        # First schedule should produce exactly one eval
        jobs_1 = scheduler.schedule(runs, available_training_slots=0)
        assert len(jobs_1) == 1
        assert jobs_1[0].type == JobTypes.LAUNCH_EVAL

        # Subsequent schedule with same inputs should produce no further evals
        jobs_2 = scheduler.schedule(runs, available_training_slots=0)
        assert len(jobs_2) == 0

    def test_fantasies_include_pending(self):
        config = AsyncCappedSchedulerConfig(
            max_trials=10,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_async",
            protein_config=_basic_protein_config(),
            max_concurrent_evals=1,
            liar_strategy="mean",
        )
        store = MockStore()
        phase_manager = RunPhaseManager(store)
        scheduler = AsyncCappedOptimizingScheduler(config, phase_manager)

        now = datetime.now(timezone.utc)
        # One completed run with observation
        completed = RunInfo(
            run_id="done",
            has_started_training=True,
            has_completed_training=True,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary={
                "sweep/score": 0.5,
                "sweep/cost": 100.0,
                "sweep/suggestion": {"lr": 0.004},
                "sweep/eval_started": True,  # Mark eval as started and completed
            },
        )

        # Two pending runs with suggestions in summary
        pending1 = RunInfo(
            run_id="p1",
            has_started_training=True,
            has_completed_training=False,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary={"sweep/suggestion": {"lr": 0.002}},
        )
        pending2 = RunInfo(
            run_id="p2",
            has_started_training=True,
            has_completed_training=False,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary={"sweep/suggestion": {"lr": 0.006}},
        )

        runs = [completed, pending1, pending2]

        # Monkeypatch optimizer.suggest to capture observations length
        captured = {}

        def fake_suggest(observations, n_suggestions=1):
            captured["num_obs"] = len(observations)
            return [{"lr": 0.003} for _ in range(n_suggestions)]

        scheduler.optimizer.suggest = fake_suggest  # type: ignore

        jobs = scheduler.schedule(runs, available_training_slots=1)

        # One completed obs + 2 fantasies expected
        assert captured.get("num_obs") == 3
        assert len(jobs) == 1
        assert jobs[0].type == JobTypes.LAUNCH_TRAINING

    def test_skip_evaluation_schedules_training(self):
        """When skip_evaluation=True, scheduler should schedule training for
        runs in TRAINING_DONE_NO_EVAL state instead of evaluation."""
        config = AsyncCappedSchedulerConfig(
            max_trials=10,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_async",
            protein_config=_basic_protein_config(),
            max_concurrent_evals=1,
            skip_evaluation=True,
        )
        store = MockStore()
        phase_manager = RunPhaseManager(store)
        scheduler = AsyncCappedOptimizingScheduler(config, phase_manager)
        scheduler.optimizer.suggest = lambda observations, n_suggestions=1: [
            {"lr": 0.003} for _ in range(n_suggestions)
        ]  # type: ignore

        now = datetime.now(timezone.utc)
        # One run that has finished training but no eval started
        runs = [
            RunInfo(
                run_id="r1",
                has_started_training=True,
                has_completed_training=True,
                has_failed=False,
                created_at=now,
                last_updated_at=now,
                summary={"sweep/suggestion": {"lr": 0.003}},
            ),
        ]

        # With skip_evaluation=True, should schedule training (not eval)
        jobs = scheduler.schedule(runs, available_training_slots=2)

        # Should get training jobs, not eval jobs
        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
        eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]
        assert len(eval_jobs) == 0, "Should not schedule eval jobs when skip_evaluation=True"
        assert len(training_jobs) > 0, "Should schedule training jobs"
