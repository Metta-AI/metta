from __future__ import annotations

from typing import Any, Iterable

import pytest

from experiments.notebooks.util import config_for_run


class _FakeRun:
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config


class _FakeApi:
    def __init__(self, runs: Iterable[_FakeRun]) -> None:
        self._runs = list(runs)

    def runs(self, _path: str, *, filters: dict[str, Any] | None = None) -> list[_FakeRun]:
        if not filters:
            return self._runs
        if "displayName" in filters:
            return [r for r in self._runs if r.name == filters["displayName"]]
        if "config.policy_name" in filters:
            key = filters["config.policy_name"]
            return [r for r in self._runs if r.config.get("policy_name") == key]
        return []


def test_config_for_run_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun("train-run", {"lr": 0.1})
    monkeypatch.setattr("wandb.Api", lambda: _FakeApi([run]))
    cfg = config_for_run(run_name="train-run")
    assert cfg == {"lr": 0.1}


def test_config_for_run_by_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun("other-run", {"policy_name": "my-policy", "gamma": 0.99})
    monkeypatch.setattr("wandb.Api", lambda: _FakeApi([run]))
    cfg = config_for_run(policy_name="my-policy")
    assert cfg == {"policy_name": "my-policy", "gamma": 0.99}


def test_config_for_run_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wandb.Api", lambda: _FakeApi([]))
    with pytest.raises(LookupError):
        config_for_run(run_name="missing")
