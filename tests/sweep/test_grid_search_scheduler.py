"""Tests for GridSearchScheduler behavior and integration surface.

Covers:
- Flattening nested categorical parameters (CategoricalParameterConfig and lists)
- Training/eval scheduling with resource constraints
- Respecting max_trials cap
- Rejecting non-categorical parameters
"""

import datetime

import pytest

import metta.adaptive.models
import metta.sweep.core
import metta.sweep.schedulers.grid_search


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def test_grid_scheduler_basic_flow():
    # Build a 2x2 grid: model.color x trainer.device
    params = {
        "model": {"color": metta.sweep.core.CategoricalParameterConfig(choices=["red", "blue"])},
        "trainer": {"device": ["cpu", "cuda"]},
    }

    cfg = metta.sweep.schedulers.grid_search.GridSearchSchedulerConfig(
        experiment_id="grid_exp",
        parameters=params,
        max_trials=3,  # cap below full grid of 4
    )
    scheduler = metta.sweep.schedulers.grid_search.GridSearchScheduler(cfg)

    # First call: request 2 training slots -> should schedule 2 trainings
    jobs = scheduler.schedule([], available_training_slots=2)
    assert len(jobs) == 2
    assert all(job.type == metta.adaptive.models.JobTypes.LAUNCH_TRAINING for job in jobs)
    for job in jobs:
        suggestion = job.metadata.get("sweep/suggestion", {})
        assert suggestion["model.color"] in {"red", "blue"}
        assert suggestion["trainer.device"] in {"cpu", "cuda"}

    # Second call: no free slots -> no new jobs
    runs_in_training = [
        metta.adaptive.models.RunInfo(
            run_id=jobs[0].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
        metta.adaptive.models.RunInfo(
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
        metta.adaptive.models.RunInfo(
            run_id=jobs[0].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
        metta.adaptive.models.RunInfo(
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
    assert all(job.type == metta.adaptive.models.JobTypes.LAUNCH_EVAL for job in jobs3)

    # Fourth call: runs completed evals (COMPLETED) -> one more training (max_trials=3)
    runs_completed = [
        metta.adaptive.models.RunInfo(
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
        metta.adaptive.models.RunInfo(
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
    assert jobs4[0].type == metta.adaptive.models.JobTypes.LAUNCH_TRAINING

    # After third training is completed, experiment is complete per max_trials
    runs_done = runs_completed + [
        metta.adaptive.models.RunInfo(
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
    cfg = metta.sweep.schedulers.grid_search.GridSearchSchedulerConfig(experiment_id="grid_bad", parameters=params)
    with pytest.raises(TypeError):
        _ = metta.sweep.schedulers.grid_search.GridSearchScheduler(cfg)


def test_grid_scheduler_accepts_list_and_nested_config():
    params = {
        # Dotted key as leaf list is allowed
        "trainer.optimizer.device": ["cpu", "cuda"],
        # Nested with CategoricalParameterConfig is allowed
        "model": {"color": metta.sweep.core.CategoricalParameterConfig(choices=["red", "blue"])},
    }
    cfg = metta.sweep.schedulers.grid_search.GridSearchSchedulerConfig(experiment_id="grid_ok", parameters=params)
    scheduler = metta.sweep.schedulers.grid_search.GridSearchScheduler(cfg)

    # Should schedule first training job
    jobs = scheduler.schedule([], available_training_slots=1)
    assert len(jobs) == 1
    s = jobs[0].metadata.get("sweep/suggestion", {})
    assert s["trainer.optimizer.device"] in {"cpu", "cuda"}
    assert s["model.color"] in {"red", "blue"}


def test_grid_scheduler_resume_hydrates_from_runs():
    params = {
        "model": {"color": metta.sweep.core.CategoricalParameterConfig(choices=["red", "blue"])},
        "trainer": {"device": ["cpu", "cuda"]},
    }
    cfg = metta.sweep.schedulers.grid_search.GridSearchSchedulerConfig(experiment_id="grid_resume", parameters=params)

    # Pretend two runs already completed with specific suggestions
    completed_runs = []
    suggs = [
        {"model.color": "red", "trainer.device": "cpu"},
        {"model.color": "red", "trainer.device": "cuda"},
    ]
    for i, sugg in enumerate(suggs, start=1):
        completed_runs.append(
            metta.adaptive.models.RunInfo(
                run_id=f"grid_resume_trial_{i:04d}",
                created_at=_now(),
                last_updated_at=_now(),
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                has_failed=False,
                summary={"sweep/suggestion": sugg},
            )
        )

    # New scheduler instance (simulating restart)
    scheduler = metta.sweep.schedulers.grid_search.GridSearchScheduler(cfg)

    # Ask to schedule up to 2 trainings; should skip used suggestions and launch remaining grid points
    jobs = scheduler.schedule(completed_runs, available_training_slots=2)
    assert len(jobs) == 2
    new_suggs = [{**j.metadata.get("sweep/suggestion", {})} for j in jobs]
    for ns in new_suggs:
        # Should be the remaining blue-cpu and blue-cuda in some order
        assert ns["model.color"] == "blue"
        assert ns["trainer.device"] in {"cpu", "cuda"}
    # Ensure no duplicates of the already-completed suggestions
    used = {(s["model.color"], s["trainer.device"]) for s in suggs}
    launched = {(s["model.color"], s["trainer.device"]) for s in new_suggs}
    assert used.isdisjoint(launched)


def test_grid_scheduler_eval_throttling():
    params = {
        "model": {"color": metta.sweep.core.CategoricalParameterConfig(choices=["red", "blue"])},
        "trainer": {"device": ["cpu", "cuda"]},
    }
    cfg = metta.sweep.schedulers.grid_search.GridSearchSchedulerConfig(
        experiment_id="grid_throttle", parameters=params, max_concurrent_evals=1
    )
    scheduler = metta.sweep.schedulers.grid_search.GridSearchScheduler(cfg)

    # Two runs finished training => only 1 eval should be scheduled due to throttle
    runs_train_done = [
        metta.adaptive.models.RunInfo(
            run_id="r1",
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
        metta.adaptive.models.RunInfo(
            run_id="r2",
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        ),
    ]
    jobs = scheduler.schedule(runs_train_done, available_training_slots=5)

    # Exactly one eval should be scheduled due to throttle
    eval_jobs = [j for j in jobs if j.type == metta.adaptive.models.JobTypes.LAUNCH_EVAL]
    assert len(eval_jobs) == 1

    # Grid search does not block training on eval backlog: training may be scheduled simultaneously
    train_jobs = [j for j in jobs if j.type == metta.adaptive.models.JobTypes.LAUNCH_TRAINING]
    assert len(train_jobs) >= 1
    assert len(train_jobs) <= 5
