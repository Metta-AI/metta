"""Tests for AsyncCappedOptimizingScheduler behavior."""

from datetime import datetime, timezone

from metta.adaptive.models import JobTypes, RunInfo
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.async_capped import (
    AsyncCappedOptimizingScheduler,
    AsyncCappedSchedulerConfig,
)


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
        scheduler = AsyncCappedOptimizingScheduler(config)

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
        scheduler = AsyncCappedOptimizingScheduler(config)
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
                has_started_eval=False,
                has_been_evaluated=False,
                has_failed=False,
                created_at=now,
                last_updated_at=now,
            ),
            RunInfo(
                run_id="r2",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                has_been_evaluated=False,
                has_failed=False,
                created_at=now,
                last_updated_at=now,
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
        scheduler = AsyncCappedOptimizingScheduler(config)

        now = datetime.now(timezone.utc)
        # One completed run with observation
        completed = RunInfo(
            run_id="done",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary={
                "sweep/score": 0.5,
                "sweep/cost": 100.0,
                "sweep/suggestion": {"lr": 0.004},
            },
        )

        # Two pending runs with suggestions in summary
        pending1 = RunInfo(
            run_id="p1",
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary={"sweep/suggestion": {"lr": 0.002}},
        )
        pending2 = RunInfo(
            run_id="p2",
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
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

