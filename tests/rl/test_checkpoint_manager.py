from __future__ import annotations

from pathlib import Path

import torch

from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from mettagrid.util.uri_resolvers.schemes import get_checkpoint_metadata, policy_spec_from_uri
from tests.helpers.fast_train_tool import DummyPolicy, DummyPolicyArchitecture


def _make_system_config(tmp_path: Path) -> SystemConfig:
    return SystemConfig(
        data_dir=tmp_path,
        local_only=True,
        device="cpu",
        vectorization="serial",
        seed=123,
    )


def test_get_latest_checkpoint_returns_highest_epoch(tmp_path: Path) -> None:
    system_cfg = _make_system_config(tmp_path)
    checkpoint_manager = CheckpointManager(run="test_run", system_cfg=system_cfg)

    for epoch in [1, 5, 10]:
        policy = DummyPolicy(epoch)
        checkpoint_manager.save_policy_checkpoint(
            state_dict=policy.state_dict(),
            architecture=DummyPolicyArchitecture(),
            epoch=epoch,
        )

    latest = checkpoint_manager.get_latest_checkpoint()
    assert latest is not None

    metadata = get_checkpoint_metadata(latest)
    assert metadata.epoch == 10


def test_trainer_state_save_and_restore(tmp_path: Path) -> None:
    system_cfg = _make_system_config(tmp_path)
    checkpoint_manager = CheckpointManager(run="test_run", system_cfg=system_cfg)

    optimizer = torch.optim.Adam([torch.tensor(1.0)])
    checkpoint_manager.save_trainer_state(
        optimizer,
        epoch=5,
        agent_step=1000,
        stopwatch_state={"elapsed_time": 123.45},
    )

    loaded = checkpoint_manager.load_trainer_state()
    assert loaded is not None
    assert loaded["epoch"] == 5
    assert loaded["agent_step"] == 1000
    assert loaded["stopwatch_state"]["elapsed_time"] == 123.45
    assert "optimizer_state" in loaded


def test_policy_spec_from_checkpoint_dir(tmp_path: Path) -> None:
    system_cfg = _make_system_config(tmp_path)
    checkpoint_manager = CheckpointManager(run="test_run", system_cfg=system_cfg)

    policy = DummyPolicy(1)
    checkpoint_manager.save_policy_checkpoint(
        state_dict=policy.state_dict(),
        architecture=DummyPolicyArchitecture(),
        epoch=1,
    )

    latest = checkpoint_manager.get_latest_checkpoint()
    assert latest is not None

    spec = policy_spec_from_uri(latest)
    assert spec.class_path
    assert spec.data_path or spec.init_kwargs
