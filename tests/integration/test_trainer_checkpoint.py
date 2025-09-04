"""
Integration test for trainer and policy checkpoint functionality.

This test verifies that:
1. The trainer can save checkpoints and policy states
2. Training can be resumed from a checkpoint
3. The policy is correctly loaded and training continues
4. Agent step counting and epoch tracking are preserved across restarts
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from metta.agent.agent_config import AgentConfig
from metta.cogworks.curriculum import env_curriculum
from metta.core.distributed import TorchDistributedConfig
from metta.mettagrid.builder.envs import make_arena
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from metta.rl.trainer import train
from metta.rl.trainer_config import CheckpointConfig, TrainerConfig


class TestTrainerCheckpointIntegration:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.temp_dir, "test_run")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def teardown_method(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_minimal_config(self) -> tuple[TrainerConfig, AgentConfig, SystemConfig]:
        trainer_cfg = TrainerConfig(
            total_timesteps=1000,
            batch_size=512,
            minibatch_size=256,
            bptt_horizon=8,
            update_epochs=1,
            rollout_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=32,
            curriculum=env_curriculum(make_arena(num_agents=6)),
            checkpoint=CheckpointConfig(
                checkpoint_interval=2,
                wandb_checkpoint_interval=2,
                checkpoint_dir=self.checkpoint_dir,
            ),
            evaluation=None,
        )

        agent_cfg = AgentConfig()

        system_cfg = SystemConfig(
            device="cpu",
            vectorization="serial",
            data_dir=self.temp_dir,
            seed=42,
        )

        return trainer_cfg, agent_cfg, system_cfg

    def _create_torch_dist_config(self, device: str = "cpu") -> TorchDistributedConfig:
        return TorchDistributedConfig(
            device=device,
            distributed=False,
            is_master=True,
            rank=0,
            local_rank=0,
            world_size=1,
        )

    def test_trainer_checkpoint_save_and_resume(self) -> None:
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()
        torch_dist_cfg = self._create_torch_dist_config()
        device = torch.device(system_cfg.device)

        checkpoint_manager = CheckpointManager(run="test_checkpoint_run", run_dir=self.run_dir)

        print("Starting first training run...")
        train(
            run_dir=self.run_dir,
            run="test_checkpoint_run",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            checkpoint_manager=checkpoint_manager,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        trainer_state_path = Path(self.run_dir) / "test_checkpoint_run" / "checkpoints" / "trainer_state.pt"
        assert trainer_state_path.exists(), "Trainer checkpoint was not created"

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None, "Failed to load trainer state"
        assert trainer_state["agent_step"] > 0, "Agent step should be greater than 0"
        assert trainer_state["epoch"] > 0, "Epoch should be greater than 0"

        policy_files = list(Path(checkpoint_manager.checkpoint_dir).glob("*.pt"))
        policy_files = [f for f in policy_files if f.name != "trainer_state.pt"]
        assert len(policy_files) > 0, "No policy files found in checkpoint directory"

        first_run_agent_step = trainer_state["agent_step"]
        first_run_epoch = trainer_state["epoch"]

        print(f"First run completed: agent_step={first_run_agent_step}, epoch={first_run_epoch}")

        print("Starting second training run (resume from checkpoint)...")

        trainer_cfg.total_timesteps = first_run_agent_step + 500

        checkpoint_manager_2 = CheckpointManager(run="test_checkpoint_run", run_dir=self.run_dir)

        train(
            run_dir=self.run_dir,
            run="test_checkpoint_run",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            checkpoint_manager=checkpoint_manager_2,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        trainer_state_2 = checkpoint_manager_2.load_trainer_state()
        assert trainer_state_2 is not None, "Failed to load trainer state after resume"

        assert trainer_state_2["agent_step"] > first_run_agent_step, (
            f"Agent step should have increased: {trainer_state_2['agent_step']} <= {first_run_agent_step}"
        )
        assert trainer_state_2["epoch"] >= first_run_epoch, (
            f"Epoch should have increased or stayed same: {trainer_state_2['epoch']} < {first_run_epoch}"
        )

        print(f"Second run completed: agent_step={trainer_state_2['agent_step']}, epoch={trainer_state_2['epoch']}")

        policy_files_2 = list(Path(checkpoint_manager_2.checkpoint_dir).glob("*.pt"))
        policy_files_2 = [f for f in policy_files_2 if f.name != "trainer_state.pt"]
        assert len(policy_files_2) >= len(policy_files), "Should have at least as many policy files"

        print("Checkpoint save and resume test passed!")

    def test_checkpoint_fields_are_preserved(self) -> None:
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()
        torch_dist_cfg = self._create_torch_dist_config()
        device = torch.device(system_cfg.device)

        trainer_cfg.checkpoint.checkpoint_interval = 1

        checkpoint_manager = CheckpointManager(run="test_checkpoint_fields", run_dir=self.run_dir)

        train(
            run_dir=self.run_dir,
            run="test_checkpoint_fields",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            checkpoint_manager=checkpoint_manager,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None, "Failed to load trainer state"

        assert trainer_state["agent_step"] > 0, f"Invalid agent_step: {trainer_state['agent_step']}"
        assert trainer_state["epoch"] > 0, f"Invalid epoch: {trainer_state['epoch']}"
        assert trainer_state["optimizer_state"] is not None, "Optimizer state dict is None"
        assert isinstance(trainer_state["optimizer_state"], dict), "Optimizer state dict is not a dict"
        assert len(trainer_state["optimizer_state"]) > 0, "Optimizer state dict is empty"

        if "stopwatch_state" in trainer_state and trainer_state["stopwatch_state"] is not None:
            assert isinstance(trainer_state["stopwatch_state"], dict), "Stopwatch state is not a dict"

        policy_uris = checkpoint_manager.select_checkpoints()
        assert len(policy_uris) > 0, "No policy checkpoints found"

        print("Checkpoint fields preservation test passed!")
        print(f"Checkpoint details: agent_step={trainer_state['agent_step']}, epoch={trainer_state['epoch']}")
        print(f"Policy URIs: {policy_uris[:1]}")
        print(f"Optimizer state keys: {list(trainer_state['optimizer_state'].keys())[:5]}...")

    def test_policy_loading_from_checkpoint(self) -> None:
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()
        torch_dist_cfg = self._create_torch_dist_config()
        device = torch.device(system_cfg.device)

        checkpoint_manager = CheckpointManager(run="test_policy_loading", run_dir=self.run_dir)

        train(
            run_dir=self.run_dir,
            run="test_policy_loading",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            checkpoint_manager=checkpoint_manager,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None, "Failed to load trainer state"

        policy_uris = checkpoint_manager.select_checkpoints()
        assert len(policy_uris) > 0, "No policy checkpoints found"

        policy_uri = policy_uris[0]
        assert policy_uri.startswith("file://"), f"Expected file URI, got: {policy_uri}"

        policy_file_path = policy_uri[7:]
        assert os.path.exists(policy_file_path), f"Policy file does not exist: {policy_file_path}"

        policy = checkpoint_manager.load_from_uri(policy_uri, device=device)
        assert policy is not None, "Failed to load policy from URI"

        print(f"Policy loading test passed! Policy loaded from: {policy_uri}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_with_gpu_device(self) -> None:
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()

        system_cfg.device = "cuda"
        device = torch.device(system_cfg.device)
        torch_dist_cfg = self._create_torch_dist_config(device=system_cfg.device)

        trainer_cfg.batch_size = 64
        trainer_cfg.minibatch_size = 32
        trainer_cfg.forward_pass_minibatch_target_size = 16

        checkpoint_manager = CheckpointManager(run="test_gpu_checkpoint", run_dir=self.run_dir)

        train(
            run_dir=self.run_dir,
            run="test_gpu_checkpoint",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            checkpoint_manager=checkpoint_manager,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is not None, "Failed to create checkpoint with GPU"
        assert trainer_state["agent_step"] > 0, "No training progress recorded"

        print("GPU checkpoint test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
