from __future__ import annotations

from types import SimpleNamespace

import pytest

from metta.adaptive import utils as adaptive_utils
from metta.sweep.ray import ray_run_trial as rrt


class FakeProcess:
    def __init__(self, exit_code: int = 0):
        self.returncode = exit_code
        self._poll_calls = 0

    def poll(self):
        self._poll_calls += 1
        return None if self._poll_calls == 1 else self.returncode


class DummyWandbStore:
    summaries: list[str] = []
    updates: list[tuple[str, dict]] = []

    def __init__(self, entity: str, project: str, evaluator_prefix=None):
        self.entity = entity
        self.project = project
        self.evaluator_prefix = evaluator_prefix

    def get_run_summary(self, run_id: str):
        DummyWandbStore.summaries.append(run_id)
        return {"metric/agent_step": 123, "experience/rewards": 9.5}

    def update_run_summary(self, run_id: str, summary_update: dict) -> bool:
        DummyWandbStore.updates.append((run_id, dict(summary_update)))
        return True


def _patch_common_dependencies(monkeypatch):
    reports: list[dict[str, float]] = []

    context = SimpleNamespace(get_trial_name=lambda: "demo-trial", get_trial_id=lambda: "trial-001")
    dummy_tune = SimpleNamespace(get_context=lambda: context, report=lambda payload: reports.append(payload))

    DummyWandbStore.summaries.clear()
    DummyWandbStore.updates.clear()

    monkeypatch.setattr(rrt, "tune", dummy_tune)
    monkeypatch.setattr(rrt, "WandbStore", DummyWandbStore)
    monkeypatch.setattr(rrt, "get_runtime_context", lambda: object())
    monkeypatch.setattr(rrt, "time", SimpleNamespace(sleep=lambda *_: None))

    return reports, context


def test_metta_train_fn_dispatches_and_reports(monkeypatch):
    reports, context = _patch_common_dependencies(monkeypatch)
    monkeypatch.setattr(rrt, "get_gpu_ids", lambda: [0, 1])

    created_jobs = []

    real_create_training_job = adaptive_utils.create_training_job

    def fake_create_training_job(*args, **kwargs):
        job = real_create_training_job(*args, **kwargs)
        created_jobs.append(job)
        return job

    monkeypatch.setattr(rrt, "create_training_job", fake_create_training_job)

    class DummyDispatcher:
        instances: list["DummyDispatcher"] = []

        def __init__(self, capture_output: bool, use_torchrun: bool):
            self.capture_output = capture_output
            self.use_torchrun = use_torchrun
            self.jobs = []
            self.process = FakeProcess(exit_code=0)
            DummyDispatcher.instances.append(self)

        def dispatch(self, job):
            self.jobs.append(job)
            return "pid-1"

        def get_process(self, pid):
            assert pid == "pid-1"
            return self.process

    monkeypatch.setattr(rrt, "LocalDispatcher", DummyDispatcher)

    config = {
        "sweep_config": {
            "sweep_id": "exp",
            "recipe_module": "experiments.recipes.arena_basic_easy_shaped",
            "train_entrypoint": "train",
            "stats_server_uri": "stats://uri",
            "gpus_per_trial": 2,
        },
        "params": {"trainer.batch_size": 1024},
    }

    rrt.metta_train_fn(config)

    assert DummyDispatcher.instances and DummyDispatcher.instances[0].use_torchrun is True
    assert created_jobs and created_jobs[0].run_id == context.get_trial_name()
    assert reports and set(reports[0].keys()) == {"reward", "timestep"}

    metadata_update = DummyWandbStore.updates[-1]
    assert metadata_update[0] == created_jobs[0].run_id
    assert metadata_update[1]["sweep/suggestion"] == config["params"]
    assert metadata_update[1]["sweep/assigned_gpus"] == 2
    assert metadata_update[1]["sweep/trial_id"] == context.get_trial_id()


def test_metta_train_fn_raises_on_failed_training(monkeypatch):
    reports, _ = _patch_common_dependencies(monkeypatch)
    monkeypatch.setattr(rrt, "get_gpu_ids", lambda: [])
    monkeypatch.setattr(rrt, "create_training_job", adaptive_utils.create_training_job)

    class FailingDispatcher:
        def __init__(self, capture_output: bool, use_torchrun: bool):
            self.process = FakeProcess(exit_code=9)

        def dispatch(self, job):
            return "pid-2"

        def get_process(self, pid):
            return self.process

    monkeypatch.setattr(rrt, "LocalDispatcher", FailingDispatcher)

    config = {
        "sweep_config": {
            "sweep_id": "exp",
            "recipe_module": "experiments.recipes.arena_basic_easy_shaped",
            "train_entrypoint": "train",
            "stats_server_uri": "stats://uri",
            "gpus_per_trial": 0,
        },
        "params": {},
    }

    with pytest.raises(SystemExit) as excinfo:
        rrt.metta_train_fn(config)

    assert excinfo.value.code == 9
    assert reports, "metrics should still be reported before failure"
