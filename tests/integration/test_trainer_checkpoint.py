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
from metta.cogworks.curriculum import env_curriculum
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import CheckpointConfig, TrainerConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_arena


class TestTrainerCheckpointIntegration:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.temp_dir, "test_run")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def teardown_method(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_minimal_config(
        self,
    ) -> tuple[TrainerConfig, TrainingEnvironmentConfig, FastConfig, SystemConfig]:
        curriculum = env_curriculum(make_arena(num_agents=6))

        trainer_cfg = TrainerConfig(
            total_timesteps=1_000,
            batch_size=512,
            minibatch_size=256,
            bptt_horizon=8,
            update_epochs=1,
            curriculum=curriculum,
            checkpoint=CheckpointConfig(
                checkpoint_interval=2,
                checkpoint_dir=self.checkpoint_dir,
                remote_prefix=None,
            ),
            evaluation=None,
        )

        training_env_cfg = TrainingEnvironmentConfig(
            curriculum=curriculum,
            num_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=32,
            vectorization="serial",
            seed=42,
        )

        policy_cfg = FastConfig()
        system_cfg = SystemConfig(
            device="cpu",
            vectorization="serial",
            data_dir=self.temp_dir,
            seed=42,
        )

        return trainer_cfg, training_env_cfg, policy_cfg, system_cfg

    def _run_training(
        self,
        *,
        run_name: str,
        trainer_cfg: TrainerConfig,
        training_env_cfg: TrainingEnvironmentConfig,
        policy_cfg: FastConfig,
        system_cfg: SystemConfig,
    ) -> None:
        tool = TrainTool(
            run=run_name,
            run_dir=self.run_dir,
            device=system_cfg.device,
            system=system_cfg.model_copy(deep=True),
            trainer=trainer_cfg.model_copy(deep=True),
            training_env=training_env_cfg.model_copy(deep=True),
            policy_architecture=policy_cfg.model_copy(deep=True),
            stats_server_uri=None,
        )
        tool.invoke({})

    def test_trainer_checkpoint_save_and_resume(self) -> None:
        trainer_cfg, training_env_cfg, policy_cfg, system_cfg = self._create_minimal_config()

        checkpoint_manager = CheckpointManager(run="test_checkpoint_run", run_dir=self.run_dir)

        print("Starting first training run...")
        self._run_training(
            run_name="test_checkpoint_run",
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state_path = Path(self.run_dir) / "test_checkpoint_run" / "checkpoints" / "trainer_state.pt"
        assert trainer_state_path.exists(), "Trainer checkpoint was not created"

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None, "Failed to load trainer state"
        assert trainer_state["agent_step"] > 0
        assert trainer_state["epoch"] > 0

        policy_files = [f for f in Path(checkpoint_manager.checkpoint_dir).glob("*.pt") if f.name != "trainer_state.pt"]
        assert policy_files, "No policy files found in checkpoint directory"

        first_run_agent_step = trainer_state["agent_step"]
        first_run_epoch = trainer_state["epoch"]
        print(f"First run completed: agent_step={first_run_agent_step}, epoch={first_run_epoch}")

        print("Starting second training run (resume from checkpoint)...")
        trainer_cfg.total_timesteps = first_run_agent_step + 500

        checkpoint_manager_2 = CheckpointManager(run="test_checkpoint_run", run_dir=self.run_dir)

        self._run_training(
            run_name="test_checkpoint_run",
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state_2 = checkpoint_manager_2.load_trainer_state()
        assert trainer_state_2 is not None, "Failed to load trainer state after resume"
        assert trainer_state_2["agent_step"] > first_run_agent_step
        assert trainer_state_2["epoch"] >= first_run_epoch

        policy_files_2 = [
            f for f in Path(checkpoint_manager_2.checkpoint_dir).glob("*.pt") if f.name != "trainer_state.pt"
        ]
        assert len(policy_files_2) >= len(policy_files)

    def test_checkpoint_fields_are_preserved(self) -> None:
        trainer_cfg, training_env_cfg, policy_cfg, system_cfg = self._create_minimal_config()
        trainer_cfg.checkpoint.checkpoint_interval = 1

        checkpoint_manager = CheckpointManager(run="test_checkpoint_fields", run_dir=self.run_dir)

        self._run_training(
            run_name="test_checkpoint_fields",
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

        policy_uris = checkpoint_manager.select_checkpoints()
        assert policy_uris, "No policy checkpoints found"

    def test_policy_loading_from_checkpoint(self) -> None:
        trainer_cfg, training_env_cfg, policy_cfg, system_cfg = self._create_minimal_config()

        checkpoint_manager = CheckpointManager(run="test_policy_loading", run_dir=self.run_dir)

        self._run_training(
            run_name="test_policy_loading",
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            policy_cfg=policy_cfg,
            system_cfg=system_cfg,
        )

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None

        policy_uris = checkpoint_manager.select_checkpoints()
        assert policy_uris, "Expected at least one policy checkpoint"

        # Load the latest policy to ensure it is valid
        policy = checkpoint_manager.load_agent()
        assert policy is not None
        assert hasattr(policy, "state_dict"), "Loaded policy should be a torch.nn.Module"
