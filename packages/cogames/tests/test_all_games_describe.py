"""Test that 'cogames games' describe command works for all games."""

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
def test_mission_describe(mission_name):
    """Test that 'cogames mission -m <mission_name>' works for all games."""
    # Skip navigation missions in CI without AWS credentials
    if "navigation" in mission_name and not has_aws_credentials():
        pytest.skip("Navigation missions require S3 access (AWS credentials not available)")

    result = subprocess.run(
        ["uv", "run", "cogames", "missions", "-m", mission_name],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Failed to describe mission {mission_name}: {result.stderr}"
    assert "Mission Configuration:" in result.stdout
    assert "Available Actions:" in result.stdout
