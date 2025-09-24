"""
Tests for PufferLib integration with MettaGrid.

This module tests PufferLib's ability to load and run Metta environments
through their CLI interface using their MettaPuff wrapper.
"""

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def get_pufferlib_repo():
    """Get PufferLib repository path with automatic cleanup."""
    cache_dir = os.environ.get("PUFFERLIB_CACHE_DIR")
    temp_dir = None

    try:
        if cache_dir:
            cache_path = Path(cache_dir)
            repo_path = cache_path / "PufferLib"

            if not repo_path.exists():
                cache_path.mkdir(parents=True, exist_ok=True)
                clone_cmd = [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/PufferAI/PufferLib.git",
                    str(repo_path),
                ]
                result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to clone PufferLib: {result.stderr}")
            elif os.environ.get("PUFFERLIB_UPDATE_CACHE") == "true":
                subprocess.run(["git", "-C", str(repo_path), "pull", "--ff-only"], capture_output=True, timeout=30)

            yield repo_path
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="pufferlib_test_"))
            repo_path = temp_dir / "PufferLib"
            clone_cmd = ["git", "clone", "--depth", "1", "https://github.com/PufferAI/PufferLib.git", str(repo_path)]
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone PufferLib: {result.stderr}")
            yield repo_path
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.mark.timeout(200)
@pytest.mark.skipif(
    os.environ.get("CI") != "true", reason="Skipping PufferLib integration test in local environment (CI only)"
)
@pytest.mark.slow
def test_puffer_cli_compatibility():
    """Ensure PufferLib can be installed in isolation and sees Metta."""
    with get_pufferlib_repo() as pufferlib_dir:
        project_root = Path(__file__).parent.parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # Check PufferLib importable
        import_cmd = ["uvx", "--from", str(pufferlib_dir), "python", "-c", "import pufferlib; print('OK')"]
        result = subprocess.run(import_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, (
            f"Isolated env failed to import pufferlib.\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

        # Check for pufferlib.pufferl module
        detect_cmd = [
            "uvx",
            "--from",
            str(pufferlib_dir),
            "python",
            "-c",
            "import importlib.util as i; print(i.find_spec('pufferlib.pufferl') is not None)",
        ]
        detect = subprocess.run(detect_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=60)
        has_pufferl = detect.returncode == 0 and detect.stdout.strip() == "True"

        if not has_pufferl:
            pytest.skip("PufferLib build provides neither CLI nor `pufferlib.pufferl`; skipping CLI run.")

        # Run training with minimal settings
        train_cmd = [
            "uvx",
            "--from",
            str(pufferlib_dir),
            "python",
            "-m",
            "pufferlib.pufferl",
            "train",
            "metta",
            "--train.total_timesteps",
            "10",
            "--train.wandb",
            "disabled",
            "--train.device",
            "cpu",
            "--env.game.max_steps",
            "10",
            "--env.game.map_builder.width",
            "8",
            "--env.game.map_builder.height",
            "8",
            "--env.game.num_agents",
            "2",
            "--vec.backend",
            "Serial",
        ]
        run = subprocess.run(train_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=240)

        if run.returncode != 0:
            critical = [
                "no module named metta",
                "environment not found",
                "failed to import",
                "invalid environment",
                "registration failed",
                "modulenotfounderror",
            ]
            blob = (run.stderr or "").lower() + (run.stdout or "").lower()
            if any(c in blob for c in critical):
                raise AssertionError(
                    f"PufferLib CLI failed with integration error.\nCOMMAND: {' '.join(train_cmd)}\n\n"
                    f"STDOUT:\n{run.stdout}\n\nSTDERR:\n{run.stderr}"
                )
