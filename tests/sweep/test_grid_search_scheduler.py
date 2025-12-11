"""Tests for GridSearchScheduler behavior and integration surface.

Covers:
- Flattening nested categorical parameters (CategoricalParameterConfig and lists)
- Training/eval scheduling with resource constraints
- Respecting max_trials cap
- Rejecting non-categorical parameters
"""

from datetime import datetime, timezone
from typing import Any

from metta.adaptive.models import JobTypes, RunInfo
from metta.adaptive.run_phase import RunPhaseManager
from metta.sweep.core import CategoricalParameterConfig
from metta.sweep.schedulers.grid_search import GridSearchScheduler, GridSearchSchedulerConfig


class MockStore:
    """Mock store that records update calls."""

    def __init__(self):
        self.summaries: dict[str, dict[str, Any]] = {}

    def update_run_summary(self, run_id: str, data: dict[str, Any]) -> bool:
        if run_id not in self.summaries:
            self.summaries[run_id] = {}
        self.summaries[run_id].update(data)
        return True


def _now():
    return datetime.now(timezone.utc)


def _make_phase_manager():
    return RunPhaseManager(MockStore())


def test_grid_scheduler_basic_flow():
    # Build a 2x2 grid: model.color x trainer.device
    params = {
        "model.color": CategoricalParameterConfig(choices=["red", "blue"]),
        "trainer.device": ["cpu", "cuda"],
    }

    cfg = GridSearchSchedulerConfig(
        experiment_id="grid_exp",
        parameters=params,
        max_trials=3,  # cap below full grid of 4
    )
    phase_manager = _make_phase_manager()
    scheduler = GridSearchScheduler(cfg, phase_manager)

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
            has_failed=False,
        ),
        RunInfo(
            run_id=jobs[1].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=False,
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
            has_failed=False,
            summary={},  # No sweep/eval_started
        ),
        RunInfo(
            run_id=jobs[1].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_failed=False,
            summary={},  # No sweep/eval_started
        ),
    ]
    # phase computed by phase_manager returns TRAINING_DONE_NO_EVAL
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
            has_failed=False,
            summary={
                **jobs[0].metadata,
                "sweep/eval_started": True,
                "sweep/score": 0.5,  # Observation recorded
            },
        ),
        RunInfo(
            run_id=jobs[1].run_id,
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_failed=False,
            summary={
                **jobs[1].metadata,
                "sweep/eval_started": True,
                "sweep/score": 0.6,  # Observation recorded
            },
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
            has_failed=False,
            summary={
                "sweep/eval_started": True,
                "sweep/score": 0.7,
            },
        )
    ]
    assert scheduler.is_experiment_complete(runs_done) is True


def test_grid_scheduler_ignores_non_categorical_leaves():
    # Non-categorical leaves are ignored; only categorical dimensions are used.
    params = {
        "trainer.optimizer.device": ["cpu", "cuda"],
        "trainer.optimizer.learning_rate": 1e-3,  # ignored
    }
    cfg = GridSearchSchedulerConfig(experiment_id="grid_bad", parameters=params)
    phase_manager = _make_phase_manager()
    scheduler = GridSearchScheduler(cfg, phase_manager)

    jobs = scheduler.schedule([], available_training_slots=1)
    assert len(jobs) == 1
    suggestion = jobs[0].metadata.get("sweep/suggestion", {})
    assert suggestion["trainer.optimizer.device"] in {"cpu", "cuda"}
    assert "trainer.optimizer.learning_rate" not in suggestion


def test_grid_scheduler_accepts_list_and_nested_config():
    params = {
        "trainer.optimizer.device": ["cpu", "cuda"],
        "model.color": CategoricalParameterConfig(choices=["red", "blue"]),
    }
    cfg = GridSearchSchedulerConfig(experiment_id="grid_ok", parameters=params)
    phase_manager = _make_phase_manager()
    scheduler = GridSearchScheduler(cfg, phase_manager)

    # Should schedule first training job
    jobs = scheduler.schedule([], available_training_slots=1)
    assert len(jobs) == 1
    s = jobs[0].metadata.get("sweep/suggestion", {})
    assert s["trainer.optimizer.device"] in {"cpu", "cuda"}
    assert s["model.color"] in {"red", "blue"}


def test_grid_scheduler_resume_hydrates_from_runs():
    params = {
        "model.color": CategoricalParameterConfig(choices=["red", "blue"]),
        "trainer.device": ["cpu", "cuda"],
    }
    cfg = GridSearchSchedulerConfig(experiment_id="grid_resume", parameters=params)

    # Pretend two runs already completed with specific suggestions
    completed_runs = []
    suggs = [
        {"model.color": "red", "trainer.device": "cpu"},
        {"model.color": "red", "trainer.device": "cuda"},
    ]
    for i, sugg in enumerate(suggs, start=1):
        completed_runs.append(
            RunInfo(
                run_id=f"grid_resume_trial_{i:04d}",
                created_at=_now(),
                last_updated_at=_now(),
                has_started_training=True,
                has_completed_training=True,
                has_failed=False,
                summary={
                    "sweep/suggestion": sugg,
                    "sweep/eval_started": True,
                    "sweep/score": 0.5,
                },
            )
        )

    # New scheduler instance (simulating restart)
    phase_manager = _make_phase_manager()
    scheduler = GridSearchScheduler(cfg, phase_manager)

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
        "model.color": CategoricalParameterConfig(choices=["red", "blue"]),
        "trainer.device": ["cpu", "cuda"],
    }
    cfg = GridSearchSchedulerConfig(experiment_id="grid_throttle", parameters=params, max_concurrent_evals=1)
    phase_manager = _make_phase_manager()
    scheduler = GridSearchScheduler(cfg, phase_manager)

    # Two runs finished training => only 1 eval should be scheduled due to throttle
    runs_train_done = [
        RunInfo(
            run_id="r1",
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_failed=False,
            summary={},
        ),
        RunInfo(
            run_id="r2",
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_failed=False,
            summary={},
        ),
    ]
    jobs = scheduler.schedule(runs_train_done, available_training_slots=5)

    # Exactly one eval should be scheduled due to throttle
    eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]
    assert len(eval_jobs) == 1

    # Grid search does not block training on eval backlog: training may be scheduled simultaneously
    train_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
    assert len(train_jobs) >= 1
    assert len(train_jobs) <= 5


def test_grid_scheduler_skips_eval_jobs_when_disabled():
    """When skip_evaluation=True, scheduler should not schedule eval jobs."""
    params = {
        "model.color": CategoricalParameterConfig(choices=["red", "blue"]),
    }
    cfg = GridSearchSchedulerConfig(
        experiment_id="grid_skip_eval",
        parameters=params,
        skip_evaluation=True,
    )
    phase_manager = _make_phase_manager()
    scheduler = GridSearchScheduler(cfg, phase_manager)

    # One run finished training
    runs_train_done = [
        RunInfo(
            run_id="r1",
            created_at=_now(),
            last_updated_at=_now(),
            has_started_training=True,
            has_completed_training=True,
            has_failed=False,
            summary={"sweep/suggestion": {"model.color": "red"}},
        ),
    ]
    jobs = scheduler.schedule(runs_train_done, available_training_slots=1)

    # Should not schedule any eval jobs
    eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]
    assert len(eval_jobs) == 0, "Should not schedule eval jobs when skip_evaluation=True"
