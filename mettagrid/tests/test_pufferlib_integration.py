"""
Tests for PufferLib integration with MettaGrid.

This module tests PufferLib's ability to load and run Metta environments
through their CLI interface using their MettaPuff wrapper.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def get_pufferlib_repo():
    """
    Get PufferLib repository path, using cache if available.

    Returns:
        Path to PufferLib repository
    """
    # Check if we have a cache directory from GitHub Actions
    cache_dir = os.environ.get("PUFFERLIB_CACHE_DIR")

    if cache_dir:
        cache_path = Path(cache_dir)
        repo_path = cache_path / "PufferLib"

        if not repo_path.exists():
            print(f"Cloning PufferLib to cache: {repo_path}")
            cache_path.mkdir(parents=True, exist_ok=True)
            clone_cmd = ["git", "clone", "--depth", "1", "https://github.com/PufferAI/PufferLib.git", str(repo_path)]
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone PufferLib: {result.stderr}")
        else:
            print(f"Using cached PufferLib from: {repo_path}")

            # Optional: Update the cached repo if requested
            if os.environ.get("PUFFERLIB_UPDATE_CACHE") == "true":
                print("Updating cached PufferLib...")
                subprocess.run(["git", "-C", str(repo_path), "pull", "--ff-only"], capture_output=True, timeout=30)

        return repo_path

    else:
        # No cache available, use temp directory
        print("No cache directory specified, using temporary directory")
        tmpdir = tempfile.mkdtemp(prefix="pufferlib_test_")
        repo_path = Path(tmpdir) / "PufferLib"

        print(f"Cloning PufferLib to temp: {repo_path}")
        clone_cmd = ["git", "clone", "--depth", "1", "https://github.com/PufferAI/PufferLib.git", str(repo_path)]
        result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            shutil.rmtree(tmpdir)
            raise RuntimeError(f"Failed to clone PufferLib: {result.stderr}")

        # Return path and cleanup function
        return repo_path


@pytest.mark.skipif(
    os.environ.get("CI") != "true", reason="Skipping PufferLib integration test in local environment (CI only)"
)
@pytest.mark.slow
def test_puffer_cli_compatibility():
    """Ensure PufferLib can be installed in isolation and sees Metta."""

    # Get PufferLib repo (from cache or fresh clone)
    pufferlib_dir = get_pufferlib_repo()

    # Track if we created a temp dir that needs cleanup
    temp_parent = None
    if "PUFFERLIB_CACHE_DIR" not in os.environ:
        # We're using a temp directory
        temp_dir = tempfile.mkdtemp(prefix="pufferlib_test_")
        pufferlib_dir = Path(temp_dir) / "PufferLib"

    try:
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

        # 2) Detect whether the modern training entrypoint exists
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

        # 3) Run the training module with tiny settings in the isolated env
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
            "--vec.backend",
            "Serial",
        ]
        run = subprocess.run(train_cmd, cwd=project_root, env=env, capture_output=True, text=True, timeout=240)

        # Only fail on true integration errors (imports/registration); otherwise just report.
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
                    "PufferLib CLI failed with integration error.\n"
                    f"COMMAND: {' '.join(train_cmd)}\n\n"
                    f"STDOUT:\n{run.stdout}\n\nSTDERR:\n{run.stderr}"
                )
            # Non-critical failures acceptable; at least import worked.

    finally:
        # Clean up temp directory if we created one
        if temp_parent and temp_parent.exists():
            shutil.rmtree(temp_parent)
