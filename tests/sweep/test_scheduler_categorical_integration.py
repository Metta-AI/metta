"""Scheduler-level categorical integration test.

Covers end-to-end flow through BatchedSyncedOptimizingScheduler with a
categorical parameter, ensuring:
- Training jobs embed string categorical values in suggestions
- Completed runs with string categorical suggestions are collected as
  observations and produce subsequent valid suggestions
"""

from datetime import datetime, timezone
from typing import Any

from metta.adaptive.models import JobTypes, RunInfo
from metta.sweep.core import CategoricalParameterConfig
from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.sweep.schedulers.batched_synced import (
    BatchedSyncedOptimizingScheduler,
    BatchedSyncedSchedulerConfig,
)


def _make_protein_config() -> ProteinConfig:
    return ProteinConfig(
        metric="test/metric",
        goal="maximize",
        parameters={
            "model": {
                "color": CategoricalParameterConfig(choices=["red", "blue", "green"]),
            },
            "trainer": {
                "optimizer": {
                    "learning_rate": ParameterConfig(
                        min=1e-5,
                        max=1e-3,
                        distribution="log_normal",
                        mean=1e-4,
                        scale="auto",
                    ),
                }
            },
        },
        settings=ProteinSettings(
            num_random_samples=0,  # seed with search center first
            seed_with_search_center=True,
        ),
    )


def test_scheduler_categorical_integration_batched_synced() -> None:
    protein_config = _make_protein_config()
    scheduler_config = BatchedSyncedSchedulerConfig(
        max_trials=4,
        batch_size=2,
        experiment_id="test_sweep_cat",
        recipe_module="test.recipe",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        protein_config=protein_config,
    )

    scheduler = BatchedSyncedOptimizingScheduler(scheduler_config)

    # First schedule: with no runs, should launch a batch of training jobs
    jobs_train = scheduler.schedule([], available_training_slots=2)
    assert len(jobs_train) == 2
    for job in jobs_train:
        assert job.type == JobTypes.LAUNCH_TRAINING
        assert "sweep/suggestion" in job.metadata
        suggestion: dict[str, Any] = job.metadata["sweep/suggestion"]
        # Suggestions use flat keys with dot notation
        assert suggestion["model.color"] in {"red", "blue", "green"}
        lr = suggestion["trainer.optimizer.learning_rate"]
        assert 1e-5 <= lr <= 1e-3

    # Simulate training done; schedule evals
    now = datetime.now(timezone.utc)
    runs_train_done = [
        RunInfo(
            run_id=job.run_id,
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary=job.metadata,  # includes sweep/suggestion
        )
        for job in jobs_train
    ]

    jobs_eval = scheduler.schedule(runs_train_done, available_training_slots=0)
    assert len(jobs_eval) == 2
    assert all(job.type == JobTypes.LAUNCH_EVAL for job in jobs_eval)

    # Simulate eval completed with scores; schedule next batch
    runs_completed = [
        RunInfo(
            run_id=job.run_id,
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            created_at=now,
            last_updated_at=now,
            summary={
                **job.metadata,  # keep sweep/suggestion present
                "sweep/score": 0.75 + i * 0.05,
                "sweep/cost": 100.0 + i,
            },
        )
        for i, job in enumerate(jobs_train)
    ]

    jobs_next = scheduler.schedule(runs_completed, available_training_slots=2)
    assert len(jobs_next) == 2
    for job in jobs_next:
        assert job.type == JobTypes.LAUNCH_TRAINING
        suggestion = job.metadata["sweep/suggestion"]
        assert suggestion["model.color"] in {"red", "blue", "green"}
        lr = suggestion["trainer.optimizer.learning_rate"]
        assert 1e-5 <= lr <= 1e-3
