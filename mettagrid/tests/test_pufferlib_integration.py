"""
Tests for PufferLib integration with MettaGrid.

This module tests the MettaGridPufferEnv with PufferLib's CLI interface.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_puffer_cli_compatibility():
    """Test PufferLib CLI compatibility with Metta environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pufferlib_dir = tmpdir_path / "PufferLib"

        # Clone PufferLib
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

        # Install PufferLib
        install_cmd = ["uv", "pip", "install", "-e", str(pufferlib_dir)]

        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Failed to install PufferLib: {result.stderr}"

        # Test the actual puffer train metta command
        train_cmd = [
            sys.executable,
            "-m",
            "pufferlib.train",
            "--env",
            "metta",
            "--train.total_timesteps",
            "10",
            "--train.wandb",
            "disabled",
            "--env.game.max_steps",
            "10",
            "--env.game.obs_width",
            "5",
            "--env.game.obs_height",
            "5",
            "--env.game.map_builder.width",
            "8",
            "--env.game.map_builder.height",
            "8",
            "--env.game.num_agents",
            "2",
        ]

        # Run the command
        result = subprocess.run(
            train_cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check for integration errors
        if result.returncode != 0:
            error_output = result.stderr.lower()
            stdout_output = result.stdout.lower()

            critical_errors = [
                "no module named metta",
                "environment not found",
                "failed to import",
                "config error",
                "invalid environment",
                "registration failed",
                "modulenotfounderror",
            ]

            if any(error in error_output or error in stdout_output for error in critical_errors):
                raise AssertionError(
                    f"PufferLib CLI failed with integration error:\n"
                    f"COMMAND: {' '.join(train_cmd)}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
