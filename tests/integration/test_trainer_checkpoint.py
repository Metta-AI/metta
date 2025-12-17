from __future__ import annotations

from pathlib import Path

from metta.rl.checkpoint_manager import CheckpointManager
from mettagrid.util.uri_resolvers.schemes import get_checkpoint_metadata, policy_spec_from_uri
from tests.helpers.fast_train_tool import create_minimal_training_setup, run_fast_train_tool


def test_trainer_checkpoint_save_and_resume(tmp_path: Path) -> None:
    run_name = "test_checkpoint_run"
    trainer_cfg, training_env_cfg, policy_cfg, system_cfg = create_minimal_training_setup(tmp_path)

    checkpoint_manager = CheckpointManager(run=run_name, system_cfg=system_cfg)

    run_fast_train_tool(
        run_name=run_name,
        system_cfg=system_cfg,
        trainer_cfg=trainer_cfg,
        training_env_cfg=training_env_cfg,
        policy_cfg=policy_cfg,
    )

    trainer_state = checkpoint_manager.load_trainer_state()
    assert trainer_state is not None
    assert trainer_state["agent_step"] > 0
    assert trainer_state["epoch"] > 0

    latest_policy_uri = checkpoint_manager.get_latest_checkpoint()
    assert latest_policy_uri is not None
    latest_policy_meta = get_checkpoint_metadata(latest_policy_uri)
    assert latest_policy_meta.epoch == trainer_state["epoch"]

    first_run_agent_step = trainer_state["agent_step"]
    first_run_epoch = trainer_state["epoch"]

    trainer_cfg.total_timesteps = first_run_agent_step + 500
    run_fast_train_tool(
        run_name=run_name,
        system_cfg=system_cfg,
        trainer_cfg=trainer_cfg,
        training_env_cfg=training_env_cfg,
        policy_cfg=policy_cfg,
    )

    trainer_state_2 = checkpoint_manager.load_trainer_state()
    assert trainer_state_2 is not None
    assert trainer_state_2["agent_step"] > first_run_agent_step
    assert trainer_state_2["epoch"] >= first_run_epoch

    latest_policy_uri = checkpoint_manager.get_latest_checkpoint()
    assert latest_policy_uri is not None
    latest_policy_meta = get_checkpoint_metadata(latest_policy_uri)
    assert latest_policy_meta.epoch == trainer_state_2["epoch"]


def test_policy_spec_from_checkpoint(tmp_path: Path) -> None:
    run_name = "test_policy_spec"
    trainer_cfg, training_env_cfg, policy_cfg, system_cfg = create_minimal_training_setup(tmp_path)

    checkpoint_manager = run_fast_train_tool(
        run_name=run_name,
        system_cfg=system_cfg,
        trainer_cfg=trainer_cfg,
        training_env_cfg=training_env_cfg,
        policy_cfg=policy_cfg,
    )

    policy_uri = checkpoint_manager.get_latest_checkpoint()
    assert policy_uri is not None

    spec = policy_spec_from_uri(policy_uri)
    assert spec.class_path
    assert spec.data_path or spec.init_kwargs
