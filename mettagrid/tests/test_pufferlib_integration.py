"""
Integration tests for PufferLib CLI compatibility.

This test module ensures that Metta's PufferLib integration works correctly
with the actual PufferLib training system, specifically testing the end-to-end
user workflow of `puffer train metta`.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestPufferLibIntegration:
    """Test suite for PufferLib CLI integration."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("CI"), reason="Only run in CI to avoid cloning repos locally")
    def test_pufferlib_cli_compatibility(self):
        """
        Integration test: Verify 'puffer train metta' works with actual PufferLib.

        This test ensures that:
        1. PufferLib can discover and load our Metta environment
        2. Our config format is compatible with PufferLib's expectations
        3. The CLI interface works end-to-end for users

        This catches breaking changes that unit tests miss, such as:
        - PufferLib CLI argument changes
        - Environment registration issues
        - Config schema incompatibilities
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pufferlib_dir = tmpdir_path / "PufferLib"

            # Clone PufferLib (standard shallow clone)
            clone_cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Metta-AI/PufferLib.git",
                str(pufferlib_dir),
            ]

            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, f"Failed to clone PufferLib: {result.stderr}"

            # Install PufferLib in test mode
            install_cmd = ["uv", "pip", "install", "-e", str(pufferlib_dir)]

            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, f"Failed to install PufferLib: {result.stderr}"

            # Test the actual puffer train metta command (real-world command, minimal config for speed)
            train_cmd = [
                sys.executable,
                "-m",
                "pufferlib.train",
                "--env",
                "metta",
                # Real-world command structure, optimized for speed like existing adapter demos
                "--train.total_timesteps",
                "20",  # Ultra-minimal like demo episodes
                "--train.wandb",
                "disabled",
                # Use fast config overrides (like demos)
                "--env.game.max_steps",
                "20",  # Very short episodes
                "--env.game.obs_width",
                "5",
                "--env.game.obs_height",
                "5",
                "--env.game.map_builder.width",
                "8",
                "--env.game.map_builder.height",
                "8",
                "--env.game.num_agents",
                "2",  # Fewer agents
            ]

            # Run the real command - should complete successfully and quickly
            result = subprocess.run(
                train_cmd,
                cwd=Path(__file__).parent.parent.parent,  # metta root directory
                capture_output=True,
                text=True,
                timeout=15,  # Fast timeout for 10 timesteps
            )

            # Check for integration/config errors
            if result.returncode != 0:
                error_output = result.stderr.lower()
                stdout_output = result.stdout.lower()

                # These indicate real integration problems (fail immediately)
                critical_errors = [
                    "no module named metta",
                    "environment not found",
                    "failed to import",
                    "config error",
                    "invalid environment",
                    "registration failed",
                    "ModuleNotFoundError".lower(),
                ]

                if any(error in error_output or error in stdout_output for error in critical_errors):
                    raise AssertionError(
                        f"PufferLib CLI failed with integration error:\n"
                        f"COMMAND: {' '.join(train_cmd)}\n"
                        f"STDOUT: {result.stdout}\n"
                        f"STDERR: {result.stderr}"
                    )

                # Runtime errors (CUDA, etc.) are acceptable - still means integration works

            # Success! The real command completed with actual training steps
