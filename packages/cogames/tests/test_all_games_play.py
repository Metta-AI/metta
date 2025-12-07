"""Test that 'cogames play' works for all games."""

import pytest
from typer.testing import CliRunner

from cogames.cli.mission import get_all_missions
from cogames.main import app

runner = CliRunner()


# Skip navigation missions - they timeout in CI due to heavy initialization
_NON_NAVIGATION_MISSIONS = [m for m in get_all_missions() if "navigation" not in m]


@pytest.mark.parametrize("mission_name", _NON_NAVIGATION_MISSIONS)
@pytest.mark.timeout(60)
def test_mission_play_non_interactive(mission_name):
    result = runner.invoke(
        app,
        [
            "play",
            "-m",
            mission_name,
            "--steps",
            "10",
            "--render",
            "none",  # Use 'none' for headless testing without terminal requirements
        ],
    )

    if result.exit_code != 0:
        pytest.fail(f"Play failed for mission {mission_name}: {result.output}")
