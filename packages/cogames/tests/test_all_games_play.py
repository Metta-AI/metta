"""Test that 'cogames play' works for all games."""

import subprocess

import pytest

import cogames.cli.mission


@pytest.mark.parametrize("mission_name", cogames.cli.mission.get_all_missions())
@pytest.mark.timeout(60)
def test_mission_play_non_interactive(mission_name):
    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "play",
            "-m",
            mission_name,
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
        pytest.fail(f"Play failed for mission {mission_name}: {result.stderr}")
