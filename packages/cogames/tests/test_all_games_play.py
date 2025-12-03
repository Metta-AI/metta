"""Test that 'cogames play' works for all games."""

import pytest
from typer.testing import CliRunner

from cogames.cli.mission import get_all_missions
from cogames.main import app

runner = CliRunner()


def has_aws_credentials() -> bool:
    """Check if AWS credentials are available."""
    try:
        import boto3

        boto3.client("s3").list_buckets()
        return True
    except Exception:
        return False


@pytest.mark.parametrize("mission_name", get_all_missions())
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
