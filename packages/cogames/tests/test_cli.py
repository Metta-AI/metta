"""Tests for cogames CLI commands."""

import subprocess
import tempfile
from pathlib import Path

cogames_root = Path(__file__).parent.parent


def test_games_list_command():
    """Test that 'cogames games' lists all available games."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games"],
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


def test_games_describe_command():
    """Test that 'cogames games <game_name>' describes a specific game."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games", "training_facility_1"],
        cwd=cogames_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected game details
    output = result.stdout
    assert "training_facility_1" in output
    assert "Game Configuration:" in output
    assert "Number of agents:" in output
    assert "Available Actions:" in output


def test_games_nonexistent_game():
    """Test that describing a nonexistent game returns an error."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games", "nonexistent_game"],
        cwd=cogames_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1, "Command should fail for nonexistent game"
    assert "Error:" in result.stdout or "Error:" in result.stderr


def test_games_help_command():
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


def test_make_game_command():
    """Test that 'cogames make-game' creates a new game configuration."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        # Run make-game and write to temp file
        result = subprocess.run(
            [
                "uv",
                "run",
                "cogames",
                "make-game",
                "assembler_1_simple",
                "--width",
                "100",
                "--output",
                str(tmp_path),
            ],
            cwd=cogames_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"make-game failed: {result.stderr}"

        # Run games command with the generated file
        result = subprocess.run(
            ["uv", "run", "cogames", "games", str(tmp_path)],
            cwd=cogames_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"games failed: {result.stderr}"

        assert tmp_path.exists()
    finally:
        tmp_path.unlink(missing_ok=True)
