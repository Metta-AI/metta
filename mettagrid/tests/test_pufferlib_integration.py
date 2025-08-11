"""
Tests for PufferLib integration with MettaGrid.

This module tests PufferLib's ability to load and run Metta environments
through their CLI interface using their MettaPuff wrapper.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
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

        # Run PufferLib in an ISOLATED env so build deps (Cython) don't pollute the main venv.
        # We use uvx with --from <local path>, then run `python -m pufferlib.train` inside that env.
        # Important: use "python" (from the uvx env), not sys.executable.
        train_cmd = [
            "uvx",
            "--from",
            str(pufferlib_dir),
            "python",
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

        project_root = Path(__file__).parent.parent.parent
        env = os.environ.copy()
        # Ensure the isolated interpreter can import the local repository as `metta`
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            train_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=180,  # allow a bit more time for first-time uvx env creation
            env=env,
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
                    "PufferLib CLI failed with integration error:\n"
                    f"COMMAND: {' '.join(train_cmd)}\n\n"
                    f"STDOUT:\n{result.stdout}\n\n"
                    f"STDERR:\n{result.stderr}"
                )

        # Non-critical failures (e.g., training exit codes) should still be reported
        assert result.returncode == 0, (
            "PufferLib CLI invocation failed.\n"
            f"COMMAND: {' '.join(train_cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
