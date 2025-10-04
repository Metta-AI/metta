"""Test that 'cogames eval' works for all games."""

import subprocess

import pytest

from cogames.game import get_all_games


@pytest.mark.parametrize("game_name", get_all_games())
@pytest.mark.timeout(60)
def test_game_eval(game_name):
    """Test that 'cogames eval' works for small games with random policy."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "eval",
            game_name,
            "cogames.policy.random.RandomPolicy",
            "--episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Eval failed for game {game_name}: {result.stderr}")

    assert "Episode 1" in result.stdout or "episode" in result.stdout.lower()
