"""Test that 'cogames play' works for all games."""

import subprocess

import pytest

from cogames.game import get_all_games


@pytest.mark.parametrize("game_name", get_all_games())
@pytest.mark.timeout(60)
def test_game_play_non_interactive(game_name):
    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "play",
            game_name,
            "--steps",
            "10",
            "--render",
            "none",  # Use 'none' for headless testing without terminal requirements
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Play failed for game {game_name}: {result.stderr}")
