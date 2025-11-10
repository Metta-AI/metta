from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from metta.sweep.ray import ray_run_trial
from metta.sweep.ray.ray_controller import SweepConfig


def _run_sweep_diagnostic(sweep_config: SweepConfig, params: dict[str, Any]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    metadata_updates: list[tuple[str, dict[str, Any]]] = []
    dispatched_jobs: list[Any] = []

    class _DryRunProcess:
        def __init__(self, exit_code: int = 0) -> None:
            self.returncode = exit_code
            self._poll_calls = 0

        def poll(self) -> int | None:
            self._poll_calls += 1
            return None if self._poll_calls < 2 else self.returncode

    class _DryRunDispatcher:
        def __init__(self, capture_output: bool, use_torchrun: bool) -> None:
            self.capture_output = capture_output
            self.use_torchrun = use_torchrun
            self._process = _DryRunProcess()

        def dispatch(self, job):
            dispatched_jobs.append(job)
            return "dry-run-pid"

        def get_process(self, pid: str):
            assert pid == "dry-run-pid"
            return self._process

    class _DryRunWandbStore:
        def __init__(self, entity: str, project: str, evaluator_prefix: str | None = None) -> None:
            self.entity = entity
            self.project = project
            self.evaluator_prefix = evaluator_prefix

        def get_run_summary(self, run_id: str) -> dict[str, Any]:
            return {"metric/agent_step": 0, "experience/rewards": 0}

        def update_run_summary(self, run_id: str, summary_update: dict[str, Any]) -> bool:
            metadata_updates.append((run_id, dict(summary_update)))
            return True

    context = SimpleNamespace(
        get_trial_name=lambda: f"{sweep_config.sweep_id}.dryrun",
        get_trial_id=lambda: "dry-run-trial",
    )

    dummy_tune = SimpleNamespace(
        get_context=lambda: context,
        report=lambda payload: reports.append(dict(payload)),
    )

    dummy_time = SimpleNamespace(sleep=lambda *_: None)

    config = {
        "sweep_config": sweep_config.model_dump(),
        "params": params,
    }

    with patch.object(ray_run_trial, "LocalDispatcher", _DryRunDispatcher), patch.object(
        ray_run_trial, "WandbStore", _DryRunWandbStore
    ), patch.object(ray_run_trial, "get_gpu_ids", lambda: [0]), patch.object(
        ray_run_trial, "get_runtime_context", lambda: object()
    ), patch.object(
        ray_run_trial, "tune", dummy_tune
    ), patch.object(
        ray_run_trial, "time", dummy_time
    ), patch.object(
        ray_run_trial, "print", lambda *args, **kwargs: None
    ):
        ray_run_trial.metta_train_fn(config)

    return {
        "reports": reports,
        "metadata_updates": metadata_updates,
        "dispatched_jobs": dispatched_jobs,
    }


def _format_diagnostic_output(summary: dict[str, Any]) -> list[str]:
    lines = ["[sweep-dry-run] diagnostic summary"]
    if summary["dispatched_jobs"]:
        job = summary["dispatched_jobs"][0]
        dump = job.model_dump()
        lines.append(f"- cmd: {dump['cmd']}")
        lines.append(f"- args: {dump['args']}")
        lines.append(f"- overrides: {list(dump['overrides'].keys())}")

    if summary["reports"]:
        lines.append(f"- last metrics: {summary['reports'][-1]}")
    if summary["metadata_updates"]:
        run_id, payload = summary["metadata_updates"][-1]
        lines.append(f"- metadata update ({run_id}): {payload}")
    return lines


@pytest.mark.integration
def test_sweep_dry_run_diagnostic(capsys):
    sweep_config = SweepConfig(
        sweep_id="diag",
        recipe_module="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        stats_server_uri="stats://dry-run",
        gpus_per_trial=0,
    )
    params = {"trainer.batch_size": 256}

    summary = _run_sweep_diagnostic(sweep_config, params)
    for line in _format_diagnostic_output(summary):
        print(line)

    output = capsys.readouterr().out
    assert "[sweep-dry-run] diagnostic summary" in output
    assert summary["dispatched_jobs"], "dispatcher should receive a job definition"
    assert summary["reports"], "tune.report should have been called"
    assert summary["metadata_updates"], "metadata should be saved back to WandB"
