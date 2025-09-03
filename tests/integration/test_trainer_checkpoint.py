#!/usr/bin/env python3
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
from metta.agent.policy_store import PolicyStore
from metta.cogworks.curriculum import env_curriculum
from metta.common.wandb.wandb_context import WandbConfig
from metta.core.distributed import TorchDistributedConfig
from metta.mettagrid.config.envs import make_arena
from metta.rl.system_config import SystemConfig
from metta.rl.trainer import train
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import CheckpointConfig, TrainerConfig


class TestTrainerCheckpointIntegration:
    """Integration tests for trainer checkpoint functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.temp_dir, "test_run")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_minimal_config(self) -> tuple[TrainerConfig, AgentConfig, SystemConfig]:
        """Create minimal configuration for testing."""
        # Create minimal trainer config for fast testing
        trainer_cfg = TrainerConfig(
            total_timesteps=1000,  # Very small for fast testing
            batch_size=512,  # Increased to accommodate 30 agents (6 agents * 5 envs)
            minibatch_size=256,
            bptt_horizon=8,
            update_epochs=1,
            rollout_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=32,
            curriculum=env_curriculum(make_arena(num_agents=6)),  # Small arena
            checkpoint=CheckpointConfig(
                checkpoint_interval=2,  # Checkpoint every 2 epochs
                wandb_checkpoint_interval=2,
                checkpoint_dir=self.checkpoint_dir,
            ),
            evaluation=None,  # Disable evaluation for faster testing
        )

        agent_cfg = AgentConfig()

        system_cfg = SystemConfig(
            device="cpu",  # Use CPU for reproducible testing
            vectorization="serial",  # Simple vectorization
            data_dir=self.temp_dir,
            seed=42,
        )

        return trainer_cfg, agent_cfg, system_cfg

    def _create_torch_dist_config(self, device: str = "cpu") -> TorchDistributedConfig:
        """Create torch distributed config for single process."""
        return TorchDistributedConfig(
            device=device,
            distributed=False,
            is_master=True,
            rank=0,
            local_rank=0,
            world_size=1,
        )

    def test_trainer_checkpoint_save_and_resume(self) -> None:
        """Test that trainer can save checkpoint and resume training."""
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()
        torch_dist_cfg = self._create_torch_dist_config()
        device = torch.device(system_cfg.device)

        # Create policy store
        policy_store = PolicyStore.create(
            device=device,
            data_dir=system_cfg.data_dir,
            wandb_config=WandbConfig.Unconfigured(),
            wandb_run=None,
        )

        # First training run - should create checkpoint
        print("Starting first training run...")
        train(
            run_dir=self.run_dir,
            run="test_checkpoint_run",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            policy_store=policy_store,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        # Verify checkpoint was created
        checkpoint_path = Path(self.run_dir) / "trainer_state.pt"
        assert checkpoint_path.exists(), "Trainer checkpoint was not created"

        # Load and verify checkpoint contents
        checkpoint = TrainerCheckpoint.load(self.run_dir)
        assert checkpoint is not None, "Failed to load checkpoint"
        assert checkpoint.agent_step > 0, "Agent step should be greater than 0"
        assert checkpoint.epoch > 0, "Epoch should be greater than 0"
        assert checkpoint.policy_path is not None, "Policy path should be set"
        assert checkpoint.optimizer_state_dict is not None, "Optimizer state should be saved"

        # Verify policy was saved
        policy_files = list(Path(self.checkpoint_dir).glob("model_*.pt"))
        assert len(policy_files) > 0, "No policy files found in checkpoint directory"

        # Store values from first run for comparison
        first_run_agent_step = checkpoint.agent_step
        first_run_epoch = checkpoint.epoch

        print(f"First run completed: agent_step={first_run_agent_step}, epoch={first_run_epoch}")

        # Second training run - should resume from checkpoint
        print("Starting second training run (resume from checkpoint)...")

        # Modify total_timesteps to allow more training
        trainer_cfg.total_timesteps = first_run_agent_step + 500  # Train a bit more

        # Create new policy store for second run
        policy_store_2 = PolicyStore.create(
            device=device,
            data_dir=system_cfg.data_dir,
            wandb_config=WandbConfig.Unconfigured(),
            wandb_run=None,
        )

        train(
            run_dir=self.run_dir,
            run="test_checkpoint_run",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            policy_store=policy_store_2,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        # Verify checkpoint was updated
        checkpoint_2 = TrainerCheckpoint.load(self.run_dir)
        assert checkpoint_2 is not None, "Failed to load checkpoint after resume"

        # Verify training continued from where it left off
        assert checkpoint_2.agent_step > first_run_agent_step, (
            f"Agent step should have increased: {checkpoint_2.agent_step} <= {first_run_agent_step}"
        )
        assert checkpoint_2.epoch >= first_run_epoch, (
            f"Epoch should have increased or stayed same: {checkpoint_2.epoch} < {first_run_epoch}"
        )

        print(f"Second run completed: agent_step={checkpoint_2.agent_step}, epoch={checkpoint_2.epoch}")

        # Verify policy was updated
        policy_files_2 = list(Path(self.checkpoint_dir).glob("model_*.pt"))
        assert len(policy_files_2) >= len(policy_files), "Should have at least as many policy files"

        print("✅ Checkpoint save and resume test passed!")

    def test_checkpoint_fields_are_preserved(self) -> None:
        """Test that all checkpoint fields are properly saved and loaded."""
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()
        torch_dist_cfg = self._create_torch_dist_config()
        device = torch.device(system_cfg.device)

        # Set frequent checkpointing
        trainer_cfg.checkpoint.checkpoint_interval = 1

        policy_store = PolicyStore.create(
            device=device,
            data_dir=system_cfg.data_dir,
            wandb_config=WandbConfig.Unconfigured(),
            wandb_run=None,
        )

        # Run training
        train(
            run_dir=self.run_dir,
            run="test_checkpoint_fields",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            policy_store=policy_store,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        # Load and verify checkpoint
        checkpoint = TrainerCheckpoint.load(self.run_dir)
        assert checkpoint is not None, "Failed to load checkpoint"

        # Verify all required fields are present and valid
        assert checkpoint.agent_step > 0, f"Invalid agent_step: {checkpoint.agent_step}"
        assert checkpoint.epoch > 0, f"Invalid epoch: {checkpoint.epoch}"
        assert checkpoint.policy_path is not None, "Policy path is None"
        assert checkpoint.policy_path.startswith("file://"), f"Invalid policy path format: {checkpoint.policy_path}"
        assert checkpoint.optimizer_state_dict is not None, "Optimizer state dict is None"
        assert isinstance(checkpoint.optimizer_state_dict, dict), "Optimizer state dict is not a dict"
        assert len(checkpoint.optimizer_state_dict) > 0, "Optimizer state dict is empty"

        # Verify stopwatch state if present
        if checkpoint.stopwatch_state is not None:
            assert isinstance(checkpoint.stopwatch_state, dict), "Stopwatch state is not a dict"

        # Verify extra args if present
        if checkpoint.extra_args:
            assert isinstance(checkpoint.extra_args, dict), "Extra args is not a dict"

        print("✅ Checkpoint fields preservation test passed!")
        print(f"Checkpoint details: agent_step={checkpoint.agent_step}, epoch={checkpoint.epoch}")
        print(f"Policy path: {checkpoint.policy_path}")
        print(f"Optimizer state keys: {list(checkpoint.optimizer_state_dict.keys())[:5]}...")  # Show first 5 keys

    def test_policy_loading_from_checkpoint(self) -> None:
        """Test that policies are correctly loaded from checkpoint paths."""
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()
        torch_dist_cfg = self._create_torch_dist_config()
        device = torch.device(system_cfg.device)

        # First run to create checkpoint
        policy_store = PolicyStore.create(
            device=device,
            data_dir=system_cfg.data_dir,
            wandb_config=WandbConfig.Unconfigured(),
            wandb_run=None,
        )

        train(
            run_dir=self.run_dir,
            run="test_policy_loading",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            policy_store=policy_store,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        # Load checkpoint and verify policy path
        checkpoint = TrainerCheckpoint.load(self.run_dir)
        assert checkpoint is not None, "Failed to load checkpoint"
        assert checkpoint.policy_path is not None, "Policy path not set in checkpoint"

        # Verify the policy file exists
        if checkpoint.policy_path.startswith("file://"):
            policy_file_path = checkpoint.policy_path[7:]  # Remove "file://" prefix
            assert os.path.exists(policy_file_path), f"Policy file does not exist: {policy_file_path}"

        # Verify we can load the policy record
        policy_record = policy_store.policy_record(checkpoint.policy_path)
        assert policy_record is not None, "Failed to load policy record"
        assert policy_record.policy is not None, "Policy not loaded in record"

        print(f"✅ Policy loading test passed! Policy loaded from: {checkpoint.policy_path}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_with_gpu_device(self) -> None:
        """Test checkpoint functionality with GPU device."""
        trainer_cfg, agent_cfg, system_cfg = self._create_minimal_config()

        # Use GPU
        system_cfg.device = "cuda"
        device = torch.device(system_cfg.device)
        torch_dist_cfg = self._create_torch_dist_config(device=system_cfg.device)

        # Reduce batch sizes for GPU memory constraints
        trainer_cfg.batch_size = 64
        trainer_cfg.minibatch_size = 32
        trainer_cfg.forward_pass_minibatch_target_size = 16

        policy_store = PolicyStore.create(
            device=device,
            data_dir=system_cfg.data_dir,
            wandb_config=WandbConfig.Unconfigured(),
            wandb_run=None,
        )

        # Run training
        train(
            run_dir=self.run_dir,
            run="test_gpu_checkpoint",
            system_cfg=system_cfg,
            agent_cfg=agent_cfg,
            device=device,
            trainer_cfg=trainer_cfg,
            wandb_run=None,
            policy_store=policy_store,
            stats_client=None,
            torch_dist_cfg=torch_dist_cfg,
        )

        # Verify checkpoint was created
        checkpoint = TrainerCheckpoint.load(self.run_dir)
        assert checkpoint is not None, "Failed to create checkpoint with GPU"
        assert checkpoint.agent_step > 0, "No training progress recorded"

        print("✅ GPU checkpoint test passed!")


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v", "-s"])
