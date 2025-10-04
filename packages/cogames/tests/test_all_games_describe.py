"""Test that 'cogames games' describe command works for all games."""

import subprocess

import pytest

from cogames.game import get_all_games


@pytest.mark.parametrize("game_name", get_all_games())
@pytest.mark.timeout(60)
def test_game_describe(game_name):
    """Test that 'cogames games <game>' works for all games."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games", game_name],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Failed to describe game {game_name}: {result.stderr}"
    assert "Game Configuration:" in result.stdout
    assert "Available Actions:" in result.stdout
