"""Test that 'cogames eval' works for all games."""

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
def test_mission_eval(mission_name):
    """Test that 'cogames eval' works for small games with random policy."""
    # Skip navigation missions in CI without AWS credentials
    if "navigation" in mission_name and not has_aws_credentials():
        pytest.skip("Navigation missions require S3 access (AWS credentials not available)")

    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "eval",
            "-m",
            mission_name,
            "-p",
            "class=random,proportion=2",
            "-p",
            "class=random,proportion=5",
            "--episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Eval failed for mission {mission_name}: {result.stderr}")

    assert "Episode 1" in result.stdout or "episode" in result.stdout.lower()


@pytest.mark.parametrize("mission_name", [get_all_missions()[0]])
@pytest.mark.timeout(60)
def test_alternate_eval_format(mission_name):
    """Test that 'cogames eval' works for small games with random policy using class= format."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "cogames",
            "eval",
            "-m",
            mission_name,
            "-p",
            "class=random",
            "--episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(f"Eval failed for mission {mission_name}: {result.stderr}")

    assert "Episode 1" in result.stdout or "episode" in result.stdout.lower()
