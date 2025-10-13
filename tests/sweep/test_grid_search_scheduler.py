"""Tests for GridSearchScheduler behavior and integration surface.

Covers:
- Flattening nested categorical parameters (CategoricalParameterConfig and lists)
- Training/eval scheduling with resource constraints
- Respecting max_trials cap
- Rejecting non-categorical parameters
"""

from datetime import datetime, timezone

import pytest

from metta.adaptive.models import JobStatus, JobTypes, RunInfo
from metta.sweep.core import CategoricalParameterConfig
from metta.sweep.schedulers.grid_search import GridSearchScheduler, GridSearchSchedulerConfig


def _now():
    return datetime.now(timezone.utc)


def test_grid_scheduler_basic_flow():
    # Build a 2x2 grid: model.color x trainer.device
    params = {
        "model": {"color": CategoricalParameterConfig(choices=["red", "blue"])},
        "trainer": {"device": ["cpu", "cuda"]},
    }

    cfg = GridSearchSchedulerConfig(
        experiment_id="grid_exp",
        parameters=params,
        max_trials=3,  # cap below full grid of 4
    )
    scheduler = GridSearchScheduler(cfg)

    # First call: request 2 training slots -> should schedule 2 trainings
    jobs = scheduler.schedule([], available_training_slots=2)
    assert len(jobs) == 2
    assert all(job.type == JobTypes.LAUNCH_TRAINING for job in jobs)
    for job in jobs:
        suggestion = job.metadata.get("sweep/suggestion", {})
        assert suggestion["model.color"] in {"red", "blue"}
        assert suggestion["trainer.device"] in {"cpu", "cuda"}

    # Second call: no free slots -> no new jobs
    runs_in_training = [
        RunInfo(
            run_id=jobs[0].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
        RunInfo(
            run_id=jobs[1].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
    ]
    jobs2 = scheduler.schedule(runs_in_training, available_training_slots=0)
    assert len(jobs2) == 0

    # Third call: both runs finished training -> schedule evals
    runs_train_done = [
        RunInfo(
            run_id=jobs[0].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
        RunInfo(
            run_id=jobs[1].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
    ]
    # status property returns TRAINING_DONE_NO_EVAL when completed but not evaluated
    jobs3 = scheduler.schedule(runs_train_done, available_training_slots=0)
    assert len(jobs3) == 2
    assert all(job.type == JobTypes.LAUNCH_EVAL for job in jobs3)

    # Fourth call: runs completed evals (COMPLETED) -> one more training (max_trials=3)
    runs_completed = [
        RunInfo(
            run_id=jobs[0].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            summary=jobs[0].metadata,
        ),
        RunInfo(
            run_id=jobs[1].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            summary=jobs[1].metadata,
        ),
    ]
    jobs4 = scheduler.schedule(runs_completed, available_training_slots=1)
    assert len(jobs4) == 1
    assert jobs4[0].type == JobTypes.LAUNCH_TRAINING

    # After third training is completed, experiment is complete per max_trials
    runs_done = runs_completed + [
        RunInfo(
            run_id=jobs4[0].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
        )
    ]
    assert scheduler.is_experiment_complete(runs_done) is True


def test_grid_scheduler_rejects_non_categorical():
    # Provide an unsupported numeric leaf value to ensure we fail fast
    params = {
        "trainer": {
            "optimizer": {
                "device": ["cpu", "cuda"],
                # Unsupported type (int) to simulate misconfiguration
                "learning_rate": 1e-3,
            }
        }
    }
    cfg = GridSearchSchedulerConfig(experiment_id="grid_bad", parameters=params)
    with pytest.raises(TypeError):
        _ = GridSearchScheduler(cfg)


def test_grid_scheduler_accepts_list_and_nested_config():
    params = {
        # Dotted key as leaf list is allowed
        "trainer.optimizer.device": ["cpu", "cuda"],
        # Nested with CategoricalParameterConfig is allowed
        "model": {"color": CategoricalParameterConfig(choices=["red", "blue"])},
    }
    cfg = GridSearchSchedulerConfig(experiment_id="grid_ok", parameters=params)
    scheduler = GridSearchScheduler(cfg)

    # Should schedule first training job
    jobs = scheduler.schedule([], available_training_slots=1)
    assert len(jobs) == 1
    s = jobs[0].metadata.get("sweep/suggestion", {})
    assert s["trainer.optimizer.device"] in {"cpu", "cuda"}
    assert s["model.color"] in {"red", "blue"}

