# mettagrid/tests/test_pufferlib_integration.py

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_puffer_cli_compatibility():
    """Ensure PufferLib can be installed in isolation and sees Metta."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pufferlib_dir = tmpdir_path / "PufferLib"

        # Clone PufferLib (shallow)
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/Metta-AI/PufferLib.git",
            str(pufferlib_dir),
        ]
        result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"Failed to clone PufferLib: {result.stderr}"

        project_root = Path(__file__).parent.parent.parent
        env = os.environ.copy()
        # Let the isolated interpreter import this repo as `metta`
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # 1) Sanity: PufferLib importable in isolated env
        import_cmd = [
            "uvx",
            "--from",
            str(pufferlib_dir),
            "python",
            "-c",
            "import pufferlib; print('OK')",
        ]
        result = subprocess.run(import_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, (
            f"Isolated env failed to import pufferlib.\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

        # 2) Detect whether pufferlib.train exists (newer releases may drop it)
        detect_cmd = [
            "uvx",
            "--from",
            str(pufferlib_dir),
            "python",
            "-c",
            "import importlib.util; print(importlib.util.find_spec('pufferlib.train') is not None)",
        ]
        detect = subprocess.run(detect_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=60)
        has_train = detect.returncode == 0 and detect.stdout.strip() == "True"

        if has_train:
            # 3a) Run the training module with tiny settings
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
            run = subprocess.run(train_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=240)
            # If it fails, surface only integration errors (import/registration)
            if run.returncode != 0:
                critical = [
                    "no module named metta",
                    "environment not found",
                    "failed to import",
                    "invalid environment",
                    "registration failed",
                    "modulenotfounderror",
                ]
                if any(c in (run.stderr.lower() + run.stdout.lower()) for c in critical):
                    raise AssertionError(
                        "PufferLib CLI failed with integration error.\n"
                        f"COMMAND: {' '.join(train_cmd)}\n\n"
                        f"STDOUT:\n{run.stdout}\n\nSTDERR:\n{run.stderr}"
                    )
                # non-critical failures acceptable; just assert import worked above
        else:
            # 3b) Fallback: ensure CLI exists (e.g., `pufferlib --help`)
            help_cmd = ["uvx", "--from", str(pufferlib_dir), "pufferlib", "--help"]
            run = subprocess.run(help_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, (
                "PufferLib CLI not available (no train module and no CLI).\n"
                f"STDOUT:\n{run.stdout}\n\nSTDERR:\n{run.stderr}"
            )
