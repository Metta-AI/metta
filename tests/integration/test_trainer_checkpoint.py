"""
Integration tests for trainer and checkpoint functionality using the new Trainer/TrainTool flow.

This verifies that
1. Training produces checkpoints and trainer state
2. Training can resume from a checkpoint
3. Policy checkpoints are loadable and consistent
"""

import os
import shutil
import tempfile
from pathlib import Path

from metta.agent.policies.fast import FastConfig
from metta.cogworks.curriculum import Curriculum
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import TrainingEnvironmentConfig
from mettagrid.policy.checkpoint_policy import CheckpointPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import get_checkpoint_metadata
from tests.helpers.fast_train_tool import create_minimal_training_setup, run_fast_train_tool


class TestTrainerCheckpointIntegration:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_minimal_config(
        self,
    ) -> tuple[TrainerConfig, TrainingEnvironmentConfig, FastConfig, SystemConfig]:
        return create_minimal_training_setup(self.temp_dir)

    def _run_training(
        self,
        *,
        run_name: str,
        trainer_cfg: TrainerConfig,
        training_env_cfg: TrainingEnvironmentConfig,
        policy_cfg: FastConfig,
        system_cfg: SystemConfig,
    ) -> None:
        run_fast_train_tool(
            run_name=run_name,
            system_cfg=system_cfg,
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
        )

    def test_trainer_checkpoint_save_and_resume(self) -> None:
        run_name = "test_checkpoint_run"
        trainer_cfg, training_env_cfg, policy_cfg, system_cfg = self._create_minimal_config()

        expected_run_dir = system_cfg.data_dir / run_name
        checkpoint_manager = CheckpointManager(run=run_name, system_cfg=system_cfg)

        assert expected_run_dir.exists(), "expected_run_dir was not created"
        expected_checkpoint_dir = expected_run_dir / "checkpoints"
        assert expected_checkpoint_dir.exists(), "expected_checkpoint_dir was not created"

        print("Starting first training run...")
        self._run_training(
            run_name=run_name,
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state_path = expected_run_dir / "checkpoints" / "trainer_state.pt"
        assert trainer_state_path.exists(), "Trainer checkpoint was not created"

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None, "Failed to load trainer state"
        assert trainer_state["agent_step"] > 0
        assert trainer_state["epoch"] > 0

        latest_policy_uri = checkpoint_manager.get_latest_checkpoint()
        assert latest_policy_uri, "No policy files found in checkpoint directory"
        latest_policy_meta = get_checkpoint_metadata(latest_policy_uri)
        assert latest_policy_meta.epoch == trainer_state["epoch"], (
            "Trainer state epoch is not aligned with latest policy checkpoint"
        )

        first_run_agent_step = trainer_state["agent_step"]
        first_run_epoch = trainer_state["epoch"]
        print(f"First run completed: agent_step={first_run_agent_step}, epoch={first_run_epoch}")

        print("Starting second training run (resume from checkpoint)...")
        trainer_cfg.total_timesteps = first_run_agent_step + 500

        checkpoint_manager_2 = CheckpointManager(run=run_name, system_cfg=system_cfg)

        self._run_training(
            run_name=run_name,
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state_2 = checkpoint_manager_2.load_trainer_state()
        assert trainer_state_2 is not None, "Failed to load trainer state after resume"
        assert trainer_state_2["agent_step"] > first_run_agent_step
        assert trainer_state_2["epoch"] >= first_run_epoch

        latest_policy_uri = checkpoint_manager_2.get_latest_checkpoint()
        assert latest_policy_uri, "No policy checkpoints found after resume"
        latest_policy_meta = get_checkpoint_metadata(latest_policy_uri)
        assert latest_policy_meta.epoch == trainer_state_2["epoch"], (
            "Trainer state epoch is not aligned with latest policy checkpoint after resume"
        )

    def test_checkpoint_fields_are_preserved(self) -> None:
        run_name = "test_checkpoint_fields"
        trainer_cfg, training_env_cfg, policy_cfg, system_cfg = self._create_minimal_config()

        checkpoint_manager = CheckpointManager(run=run_name, system_cfg=system_cfg)

        self._run_training(
            run_name=run_name,
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None
        assert trainer_state["agent_step"] > 0
        assert trainer_state["epoch"] > 0
        assert isinstance(trainer_state.get("optimizer_state"), dict)

        policy_uri = checkpoint_manager.get_latest_checkpoint()
        assert policy_uri, "No policy checkpoints found"

    def test_policy_loading_from_checkpoint(self) -> None:
        run_name = "test_policy_loading"
        trainer_cfg, training_env_cfg, policy_cfg, system_cfg = self._create_minimal_config()

        checkpoint_manager = CheckpointManager(run=run_name, system_cfg=system_cfg)

        self._run_training(
            run_name=run_name,
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None

        policy_uri = checkpoint_manager.get_latest_checkpoint()
        assert policy_uri, "Expected at least one policy checkpoint"

        env_cfg = Curriculum(training_env_cfg.curriculum).get_task().get_env_cfg()
        env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
        assert (
            CheckpointPolicy.from_checkpoint_uri(env_info, policy_uri, device_override=system_cfg.device)
            .wrapped_policy.state_dict()
        )
