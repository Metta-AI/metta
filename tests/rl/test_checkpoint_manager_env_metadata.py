import logging
from pathlib import Path

import pytest

from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig


class _DummyAction:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyActions:
    def __init__(self, names: list[str]) -> None:
        self._actions = [_DummyAction(n) for n in names]

    def actions(self):
        return self._actions


class _DummyEnvInfo:
    def __init__(self, num_agents: int, actions: list[str]) -> None:
        self.num_agents = num_agents
        self.obs_width = 1
        self.obs_height = 1
        self.tags: list[str] = []
        self.actions = _DummyActions(actions)


def test_env_metadata_persist_and_load(tmp_path: Path) -> None:
    cfg = SystemConfig(data_dir=tmp_path, device="cpu", vectorization="serial")
    mgr = CheckpointManager(run="test-run", system_cfg=cfg, require_remote_enabled=False)
    env_info = _DummyEnvInfo(num_agents=2, actions=["noop", "move"])

    mgr._persist_env_metadata(env_info)
    meta = CheckpointManager._load_env_metadata(mgr.checkpoint_dir / "dummy.mpt")
    assert meta is not None
    assert meta["num_agents"] == 2
    assert meta["actions"] == ["noop", "move"]


def test_env_metadata_validation_warns_on_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    meta = {"num_agents": 2, "obs_width": 1, "obs_height": 1, "actions": ["noop", "move"], "tags": []}
    env_info = _DummyEnvInfo(num_agents=3, actions=["noop", "attack"])

    CheckpointManager._validate_env_metadata(meta, env_info)
    assert any("Environment metadata mismatch" in rec.message for rec in caplog.records)
