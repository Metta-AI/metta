"""Tests for cogames CLI commands."""

import subprocess
from pathlib import Path


def test_games_list_command():
    """Test that 'cogames games' lists all available games."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected content
    output = result.stdout
    assert "Available Games" in output
    assert "assembler_1_simple" in output
    assert "Agents" in output
    assert "Map Size" in output


def test_games_describe_command():
    """Test that 'cogames games <game_name>' describes a specific game."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games", "assembler_1_simple"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that the output contains expected game details
    output = result.stdout
    assert "assembler_1_simple" in output
    assert "Game Configuration:" in output
    assert "Number of agents:" in output
    assert "Available Actions:" in output


def test_games_nonexistent_game():
    """Test that describing a nonexistent game returns an error."""
    result = subprocess.run(
        ["uv", "run", "cogames", "games", "nonexistent_game"],
        cwd=Path(__file__).parent.parent,
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
        cwd=Path(__file__).parent.parent,
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
