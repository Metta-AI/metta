"""Tests for cogames CLI commands."""

import subprocess
import tempfile
from pathlib import Path

cogames_root = Path(__file__).parent.parent


def test_missions_list_command():
    """Test that 'cogames missions' lists all available missions."""
    result = subprocess.run(
        ["uv", "run", "cogames", "missions"],
        cwd=cogames_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected content
    output = result.stdout
    assert "training_facility_1" in output
    assert "Agents" in output
    assert "Map Size" in output


def test_missions_describe_command():
    """Test that 'cogames missions <mission_name>' describes a specific mission."""
    result = subprocess.run(
        ["uv", "run", "cogames", "missions", "training_facility_1"],
        cwd=cogames_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected game details
    output = result.stdout
    assert "training_facility_1" in output
    assert "Mission Configuration:" in output
    assert "Number of agents:" in output
    assert "Available Actions:" in output


def test_missions_nonexistent_mission():
    """Test that describing a nonexistent game returns an error."""
    result = subprocess.run(
        ["uv", "run", "cogames", "missions", "nonexistent_mission"],
        cwd=cogames_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1, "Command should fail for nonexistent mission"
    assert "Error:" in result.stdout or "Error:" in result.stderr


def test_missions_help_command():
    """Test that 'cogames --help' shows help text."""
    result = subprocess.run(
        ["uv", "run", "cogames", "--help"],
        cwd=cogames_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that help text contains expected commands
    output = result.stdout
    assert "games" in output
    assert "play" in output
    assert "train" in output


def test_make_mission_command():
    """Test that 'cogames make-mission' creates a new mission configuration."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        # Run make-game and write to temp file
        result = subprocess.run(
            [
                "uv",
                "run",
                "cogames",
                "make-mission",
                "random",
                "--width",
                "100",
                "--height",
                "100",
                "--output",
                str(tmp_path),
            ],
            cwd=cogames_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"make-mission failed: {result.stderr}"

        # Run games command with the generated file
        result = subprocess.run(
            ["uv", "run", "cogames", "missions", str(tmp_path)],
            cwd=cogames_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"missions failed: {result.stderr}"

        assert tmp_path.exists()
    finally:
        tmp_path.unlink(missing_ok=True)
