"""Test that 'cogames games' describe command works for all games."""

import subprocess

import pytest

import cogames.cli.mission


@pytest.mark.parametrize("mission_name", cogames.cli.mission.get_all_missions())
@pytest.mark.timeout(60)
def test_mission_describe(mission_name):
    """Test that 'cogames mission -m <mission_name>' works for all games."""
    result = subprocess.run(
        ["uv", "run", "cogames", "missions", "-m", mission_name],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Failed to describe mission {mission_name}: {result.stderr}"
    assert "Mission Configuration:" in result.stdout
    assert "Available Actions:" in result.stdout
