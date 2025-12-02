"""Test that 'cogames play' works for all games."""

import subprocess

import pytest

from cogames.cli.mission import get_all_missions


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
    # Skip navigation missions in CI without AWS credentials
    if "navigation" in mission_name and not has_aws_credentials():
        pytest.skip("Navigation missions require S3 access (AWS credentials not available)")

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
