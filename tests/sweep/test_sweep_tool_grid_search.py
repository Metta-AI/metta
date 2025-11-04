"""Integration test for grid_search helper with SweepTool pipeline.

This avoids external services by monkeypatching store/dispatcher/controller.
Verifies that the grid-search scheduler is constructed and can emit suggestions.
"""

from typing import Any

from metta.sweep.core import CategoricalParameterConfig, grid_search
from metta.sweep.schedulers.grid_search import GridSearchScheduler


def test_sweep_tool_grid_search_builds_scheduler(monkeypatch, tmp_path) -> None:
    import metta.tools.sweep as sweep_mod

    # Dummy components to avoid external dependencies
    class DummyStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def fetch_runs(self, filters: dict, limit: int | None = None):
            return []

        def init_run(self, *args: Any, **kwargs: Any):
            return None

        def update_run_summary(self, *args: Any, **kwargs: Any):
            return True

    class DummyDispatcher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def dispatch(self, job):
            return "ok"

    captured: dict[str, Any] = {}

    class DummyController:
        def __init__(self, experiment_id, scheduler, dispatcher, store, config):
            captured["scheduler"] = scheduler
            captured["dispatcher"] = dispatcher
            captured["store"] = store
            captured["config"] = config

        def run(self, on_eval_completed=None):  # noqa: ARG002
            # Do nothing; we only care that construction succeeded
            return 0

    # Patch components in sweep tool module
    monkeypatch.setattr(sweep_mod, "WandbStore", DummyStore)
    monkeypatch.setattr(sweep_mod, "LocalDispatcher", DummyDispatcher)
    monkeypatch.setattr(sweep_mod, "AdaptiveController", DummyController)

    # Build grid search tool
    params = {
        "model": {"color": CategoricalParameterConfig(choices=["red", "blue"])},
        "trainer": {"device": ["cpu", "cuda"]},
    }
    tool = grid_search(
        name="grid.test",
        recipe="test.recipe",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        objective="test/metric",
        parameters=params,
        max_trials=3,
        num_parallel_trials=2,
    )

    # Avoid contacting external services
    tool.sweep_server_uri = ""
    # Use local dispatcher path we monkeypatched
    tool.dispatcher_type = sweep_mod.DispatcherType.LOCAL
    # Point data_dir to tmp to avoid polluting repo
    tool.system.data_dir = tmp_path

    # Invoke -> constructs scheduler and controller
    tool.invoke({})

    # Validate that a GridSearchScheduler was constructed
    sched = captured.get("scheduler")
    assert isinstance(sched, GridSearchScheduler)

    # Ask the scheduler for training jobs and validate suggestions
    jobs = sched.schedule([], available_training_slots=2)
    assert len(jobs) == 2
    for job in jobs:
        s = job.metadata.get("sweep/suggestion", {})
        assert s["model.color"] in {"red", "blue"}
        assert s["trainer.device"] in {"cpu", "cuda"}
