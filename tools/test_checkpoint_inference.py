#!/usr/bin/env python3
"""Test script to verify checkpoint NPC inference works with real observations."""

import logging
import os
import sys

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PyTorch compatibility
os.environ["TORCH_WEIGHTS_ONLY_DEFAULT"] = "false"


def test_checkpoint_inference():
    """Test that checkpoint NPC inference works with real observations."""
    try:
        from omegaconf import OmegaConf

        from metta.agent.policy_store import PolicyStore
        from metta.rl.trainer_config import CheckpointNPCConfig, DualPolicyConfig
        from metta.rl.util.dual_policy_rollout import DualPolicyRollout

        logger.info("Testing checkpoint NPC inference with real observations")

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
            {"run": "test_run", "data_dir": "./train_dir", "run_dir": "./train_dir/test_run", "device": "cpu"}
        )

        # Create a dummy wandb run
        class DummyWandbRun:
            def __init__(self):
                self.name = "test_run"
                self.id = "test_id"

        wandb_run = DummyWandbRun()

        # Create policy store
        policy_store = PolicyStore(cfg, wandb_run)

        # Initialize dual policy rollout
        device = torch.device("cpu")
        dual_policy = DualPolicyRollout(config, policy_store, num_agents=100, device=device)

        # Create dummy observations in the correct format
        # Observations have shape [batch_size, num_tokens, 3] where 3 is (coord_byte, attr_index, attr_value)
        # coord_byte: first 4 bits are x, last 4 bits are y (0-10 range for 11x11 obs window)
        batch_size = 10  # Test with 10 observations
        num_tokens = 200  # Typical number of observation tokens

        # Create observations in the correct format
        observations = torch.zeros(batch_size, num_tokens, 3, device=device, dtype=torch.uint8)

        # Fill with valid token data
        for b in range(batch_size):
            for t in range(num_tokens):
                if t < 50:  # First 50 tokens are valid
                    # Create valid coordinates (0-10 range for 11x11 obs window)
                    x = int(torch.randint(0, 11, (1,)).item())
                    y = int(torch.randint(0, 11, (1,)).item())
                    # Pack coordinates into a byte: first 4 bits are x, last 4 bits are y
                    coord_byte = int((x << 4) | y)
                    attr_index = torch.randint(0, 3, (1,)).item()  # Layer index
                    attr_value = torch.randint(0, 255, (1,)).item()  # Feature value

                    observations[b, t, 0] = coord_byte
                    observations[b, t, 1] = attr_index
                    observations[b, t, 2] = attr_value
                else:
                    # Empty tokens (0xFF indicates empty)
                    observations[b, t, 0] = 0xFF
                    observations[b, t, 1] = 0xFF
                    observations[b, t, 2] = 0xFF

        logger.info(f"Created test observations with shape: {observations.shape}")

        # Test NPC action generation
        npc_actions, log_probs, values, lstm_state = dual_policy._get_npc_actions(observations)

        logger.info(f"Generated NPC actions with shape: {npc_actions.shape}")
        logger.info(f"Generated log_probs with shape: {log_probs.shape}")
        logger.info(f"Generated values with shape: {values.shape}")
        logger.info(f"LSTM state: {lstm_state is not None}")

        # Verify shapes are correct
        expected_action_shape = (batch_size, 2)  # (action_type, action_param)
        expected_log_probs_shape = (batch_size,)
        expected_values_shape = (batch_size,)

        assert npc_actions.shape == expected_action_shape, (
            f"Expected actions shape {expected_action_shape}, got {npc_actions.shape}"
        )
        assert log_probs.shape == expected_log_probs_shape, (
            f"Expected log_probs shape {expected_log_probs_shape}, got {log_probs.shape}"
        )
        assert values.shape == expected_values_shape, (
            f"Expected values shape {expected_values_shape}, got {values.shape}"
        )

        # Verify action values are reasonable (should be integers for action types/params)
        assert npc_actions.dtype == torch.int32, f"Expected actions dtype torch.int32, got {npc_actions.dtype}"

        # Check that actions are not all zeros (which would indicate dummy actions)
        if torch.all(npc_actions == 0):
            logger.warning("⚠ All actions are zero - this might indicate dummy actions are still being used")
        else:
            logger.info("✓ Generated non-zero actions - checkpoint inference is working!")

        # Log some sample actions
        logger.info(f"Sample actions (first 5): {npc_actions[:5]}")
        logger.info(f"Sample log_probs (first 5): {log_probs[:5]}")
        logger.info(f"Sample values (first 5): {values[:5]}")

        return True

    except Exception as e:
        logger.error(f"Failed to test checkpoint inference: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the checkpoint inference test."""
    logger.info("Starting checkpoint NPC inference test...")

    success = test_checkpoint_inference()

    if success:
        logger.info("🎉 Checkpoint NPC inference test passed!")
        return 0
    else:
        logger.error("❌ Checkpoint NPC inference test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
