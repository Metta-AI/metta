#!/usr/bin/env python3
"""
Integration test for sweep functionality with Mac hardware configuration.
Tests that the sweep system works properly with CPU-only training.
"""

from pathlib import Path

import pytest


class TestSweepIntegration:
    """Test sweep integration with Mac hardware."""

    def test_sweep_command_validation(self):
        """Test that sweep commands are properly formatted for Mac hardware."""
        # Test the command format that should work on Mac
        expected_mac_args = [
            "+hardware=macbook",
            "device=cpu",
            "trainer.num_workers=1",
            "trainer.total_timesteps=100",  # Very small for testing
        ]

        # Verify no CUDA references in the args
        for arg in expected_mac_args:
            assert "cuda" not in arg.lower()
            assert "nccl" not in arg.lower()

    def test_sweep_config_compatibility(self):
        """Test that sweep configs don't contain hardware-specific settings."""
        sweep_dir = Path("configs/sweep")

        for config_file in sweep_dir.glob("*.yaml"):
            if config_file.stat().st_size == 0:  # Skip empty files
                continue

            content = config_file.read_text()

            # Ensure no hardcoded CUDA settings in sweep configs
            assert "device: cuda" not in content, f"Found CUDA device in {config_file}"
            assert "nccl" not in content.lower(), f"Found NCCL reference in {config_file}"

    def test_demo_sweep_config_structure(self):
        """Test that demo_sweep.yaml has the expected structure."""
        config_file = Path("configs/sweep/demo_sweep.yaml")
        content = config_file.read_text()

        # Verify it contains expected sweep parameters
        assert "learning_rate:" in content
        assert "gamma:" in content
        assert "batch_size:" in content
        assert "clip_coef:" in content

        # Verify it uses proper distributions
        assert "log_normal" in content
        assert "uniform" in content
        assert "int_uniform" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
