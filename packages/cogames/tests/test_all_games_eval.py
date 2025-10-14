"""Test that 'cogames eval' works for all games."""

import subprocess

import pytest

from cogames.cli.mission import get_all_missions


@pytest.mark.parametrize("mission_name", get_all_missions())
@pytest.mark.timeout(60)
def test_mission_eval(mission_name):
    """Test that 'cogames eval' works for small games with random policy."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "eval",
            "-m",
            mission_name,
            "-p",
            "cogames.policy.random.RandomPolicy::2",
            "-p",
            "cogames.policy.random.RandomPolicy::5",
            "--episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Eval failed for mission {mission_name}: {result.stderr}")

    assert "Episode 1" in result.stdout or "episode" in result.stdout.lower()


@pytest.mark.parametrize("mission_name", [get_all_missions()[0]])
@pytest.mark.timeout(60)
def test_alternate_eval_format(mission_name):
    """Test that 'cogames eval' works for small games with random policy with alternate cli format."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "eval",
            "-m",
            mission_name,
            "-p",
            "random",
            "--episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Eval failed for mission {mission_name}: {result.stderr}")

    assert "Episode 1" in result.stdout or "episode" in result.stdout.lower()
