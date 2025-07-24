#!/usr/bin/env python3
"""Test script to verify checkpoint loading works with dual-policy system on GPU."""

import logging
import os
import sys

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PyTorch compatibility
os.environ["TORCH_WEIGHTS_ONLY_DEFAULT"] = "false"


def test_gpu_availability():
    """Test if CUDA is available."""
    if torch.cuda.is_available():
        logger.info("✓ CUDA is available")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        logger.error("✗ CUDA is not available")
        return False


def test_checkpoint_loading_gpu():
    """Test loading a checkpoint from WandB on GPU."""
    try:
        from metta.rl.util.policy_loader import load_policy_from_checkpoint

        # Test WandB URI
        wandb_uri = "wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2"

        logger.info(f"Testing checkpoint loading from: {wandb_uri}")

        # Try to load the policy on GPU
        device = torch.device("cuda:0")  # Use first GPU
        policy = load_policy_from_checkpoint(wandb_uri, device)

        logger.info(f"Successfully loaded policy: {type(policy)}")

        # Check if it's on the correct device
        if hasattr(policy, "parameters"):
            for param in policy.parameters():
                if param.device.type == "cuda":
                    logger.info("✓ Policy is on CUDA device")
                    break
            else:
                logger.warning("⚠ Policy parameters are not on CUDA device")

        # Check if it has the expected attributes
        if hasattr(policy, "forward"):
            logger.info("✓ Policy has forward method")
        else:
            logger.warning("⚠ Policy does not have forward method")

        if hasattr(policy, "eval"):
            logger.info("✓ Policy has eval method")
        else:
            logger.warning("⚠ Policy does not have eval method")

        return True

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False


def test_dual_policy_rollout_gpu():
    """Test dual policy rollout initialization on GPU."""
    try:
        from omegaconf import OmegaConf

        from metta.agent.policy_store import PolicyStore
        from metta.rl.trainer_config import CheckpointNPCConfig, DualPolicyConfig
        from metta.rl.util.dual_policy_rollout import DualPolicyRollout

        logger.info("Testing dual policy rollout initialization on GPU")

        # Create a minimal config
        config = DualPolicyConfig(
            enabled=True,
            policy_a_percentage=0.5,
            npc_type="checkpoint",
            checkpoint_npc=CheckpointNPCConfig(
                checkpoint_path="wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2"
            ),
        )

        # Create a minimal config for PolicyStore
        cfg = OmegaConf.create(
            {"run": "test_run", "data_dir": "./train_dir", "run_dir": "./train_dir/test_run", "device": "cuda"}
        )

        # Create a dummy wandb run
        class DummyWandbRun:
            def __init__(self):
                self.name = "test_run"
                self.id = "test_id"

        wandb_run = DummyWandbRun()

        # Create policy store
        policy_store = PolicyStore(cfg, wandb_run)

        # Initialize dual policy rollout on GPU
        device = torch.device("cuda:0")
        DualPolicyRollout(config, policy_store, num_agents=100, device=device)

        logger.info("✓ Dual policy rollout initialized successfully on GPU")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize dual policy rollout: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting GPU checkpoint loading tests...")

    # Test 1: GPU availability
    test1_passed = test_gpu_availability()

    if not test1_passed:
        logger.error("CUDA not available, skipping GPU tests")
        return 1

    # Test 2: Checkpoint loading on GPU
    test2_passed = test_checkpoint_loading_gpu()

    # Test 3: Dual policy rollout on GPU
    test3_passed = test_dual_policy_rollout_gpu()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY:")
    logger.info(f"GPU availability: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    logger.info(f"Checkpoint loading on GPU: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    logger.info(f"Dual policy rollout on GPU: {'✓ PASSED' if test3_passed else '✗ FAILED'}")

    if test1_passed and test2_passed and test3_passed:
        logger.info("🎉 All tests passed! GPU checkpoint NPC system is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
