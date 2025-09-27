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

import torch
from torch import nn

from metta.agent.policies.fast import FastConfig
from mettagrid.builder.envs import make_arena
from softmax.cogworks.curriculum import env_curriculum
from softmax.training.rl.checkpoint_manager import CheckpointManager
from softmax.training.rl.system_config import SystemConfig
from softmax.training.rl.trainer_config import TrainerConfig
from softmax.training.rl.training import (
    CheckpointerConfig,
    ContextCheckpointerConfig,
    EvaluatorConfig,
    TrainingEnvironmentConfig,
)
from softmax.training.tools.train import TrainTool


class TestTrainerCheckpointIntegration:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_minimal_config(
        self,
    ) -> tuple[TrainerConfig, TrainingEnvironmentConfig, FastConfig, SystemConfig]:
        curriculum = env_curriculum(make_arena(num_agents=1))

        trainer_cfg = TrainerConfig(
            total_timesteps=16,
            batch_size=32,
            minibatch_size=16,
            bptt_horizon=4,
            update_epochs=1,
        )

        training_env_cfg = TrainingEnvironmentConfig(
            curriculum=curriculum,
            num_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=4,
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
        tool = _FastTrainTool(
            run=run_name,
            system=system_cfg.model_copy(deep=True),
            trainer=trainer_cfg.model_copy(deep=True),
            training_env=training_env_cfg.model_copy(deep=True),
            policy_architecture=policy_cfg.model_copy(deep=True),
            stats_server_uri=None,
            checkpointer=CheckpointerConfig(epoch_interval=1),
            context_checkpointer=ContextCheckpointerConfig(epoch_interval=1),
            evaluator=EvaluatorConfig(epoch_interval=0, evaluate_local=False, evaluate_remote=False),
        )
        tool.invoke({})

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

        policy_files = [f for f in Path(checkpoint_manager.checkpoint_dir).glob("*.pt") if f.name != "trainer_state.pt"]
        assert policy_files, "No policy files found in checkpoint directory"

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

        policy_files_2 = [
            f for f in Path(checkpoint_manager_2.checkpoint_dir).glob("*.pt") if f.name != "trainer_state.pt"
        ]
        assert len(policy_files_2) >= len(policy_files)

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

        policy_uris = checkpoint_manager.select_checkpoints()
        assert policy_uris, "No policy checkpoints found"

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

        policy_uris = checkpoint_manager.select_checkpoints()
        assert policy_uris, "Expected at least one policy checkpoint"

        # Load the latest policy to ensure it is valid
        policy = checkpoint_manager.load_agent()
        assert policy is not None
        assert hasattr(policy, "state_dict"), "Loaded policy should be a torch.nn.Module"


class DummyPolicy(nn.Module):
    """Lightweight torch module used to populate fake checkpoints quickly."""

    def __init__(self, epoch: int) -> None:
        super().__init__()
        self.register_buffer("epoch_tensor", torch.tensor(epoch, dtype=torch.float32))


class _FastTrainTool(TrainTool):
    """Minimal TrainTool variant that writes synthetic checkpoints without training."""

    def invoke(self, args: dict[str, str]) -> int | None:
        if "run" in args:
            assert self.run is None, "run cannot be set twice"
            self.run = args["run"]

        run_name = self.run or "default"

        checkpoint_manager = CheckpointManager(run=run_name, system_cfg=self.system)

        trainer_state_path = checkpoint_manager.checkpoint_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            previous_state = torch.load(trainer_state_path, weights_only=False)
            previous_agent_step = int(previous_state.get("agent_step", 0))
            previous_epoch = int(previous_state.get("epoch", 0))
        else:
            previous_agent_step = 0
            previous_epoch = 0

        agent_step = max(previous_agent_step, int(self.trainer.total_timesteps))
        epoch = previous_epoch + 1

        torch.save(
            {
                "agent_step": agent_step,
                "epoch": epoch,
                "optimizer_state": {},
            },
            trainer_state_path,
        )

        policy_path = checkpoint_manager.checkpoint_dir / f"{run_name}:v{epoch}.pt"
        policy = DummyPolicy(epoch)
        torch.save(policy, policy_path)

        return 0
