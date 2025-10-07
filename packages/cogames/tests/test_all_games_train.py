"""Test that 'cogames train' works for all games."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from cogames.game import get_all_missions


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp(prefix="cogames_train_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.mark.parametrize("mission_name", get_all_missions())
@pytest.mark.timeout(60)
def test_mission_train(mission_name, temp_dir):
    """Test that 'cogames train' works for small games with minimal steps."""
    checkpoint_dir = temp_dir / mission_name / "train"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "train",
            mission_name,
            "--steps=200",
            f"--checkpoints={checkpoint_dir}",
            "--batch-size=2",
            "--minibatch-size=2",
            "--num-workers=1",
            "--parallel-envs=2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Some games may have configuration issues - that's okay for this smoke test
    # The important thing is that the train command can be invoked without crashing
    if result.returncode != 0:
        pytest.fail(f"Training crashed for mission {mission_name}: {result.stderr}")
