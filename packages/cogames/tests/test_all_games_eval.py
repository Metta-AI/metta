"""Test that 'cogames run' works for all games."""

import pytest
from typer.testing import CliRunner

from cogames.cli.mission import get_all_missions
from cogames.main import app

runner = CliRunner()


@pytest.mark.parametrize("mission_name", get_all_missions())
@pytest.mark.timeout(60)
def test_mission_run(mission_name: str):
    """Test that 'cogames run' works for small games with random policy."""
    if "navigation" in mission_name:
        pytest.skip("Navigation missions require S3 access (AWS credentials not available)")

    result = runner.invoke(
        app,
        [
            "run",
            "-m",
            mission_name,
            "-p",
            "class=random,proportion=2",
            "-p",
            "class=random,proportion=5",
            "--episodes",
            "1",
        ],
    )

    if result.exit_code != 0:
        pytest.fail(f"Run failed for mission {mission_name}: {result.output}")

    assert "Episode 1" in result.output or "episode" in result.output.lower()


@pytest.mark.parametrize("mission_name", [get_all_missions()[0]])
@pytest.mark.timeout(60)
def test_alternate_run_format(mission_name):
    """Test that 'cogames run' works for small games with random policy using class= format."""
    result = runner.invoke(
        app,
        [
            "run",
            "-m",
            mission_name,
            "-p",
            "class=random",
            "--episodes",
            "1",
        ],
    )

    if result.exit_code != 0:
        pytest.fail(f"Run failed for mission {mission_name}: {result.output}")

    assert "Episode 1" in result.output or "episode" in result.output.lower()
