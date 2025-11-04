"""Tests for launch.py script.

These tests verify that launch.py can parse arguments and validate configurations
without actually launching jobs.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def get_launch_script_path() -> Path:
    """Get path to launch.py script."""
    # Assumes we're in repo root when running tests
    return Path("devops/skypilot/launch.py")


def test_launch_dry_run_success():
    """Test that launch.py --dry-run completes successfully without Pydantic warnings.

    This validates:
    1. No Pydantic warnings from SkyPilot dependencies
    2. Argument parsing works
    3. Configuration validation works
    4. Task creation works
    5. Script gets far enough to display job summary
    """
    launch_script = get_launch_script_path()

    # Run with --dry-run and --skip-git-check to avoid git state requirements
    result = subprocess.run(
        [
            sys.executable,
            str(launch_script),
            "arena.train",
            "run=test_dry_run",
            "--dry-run",
            "--skip-git-check",
        ],
        capture_output=True,
        text=True,
    )

    # Should exit cleanly with code 0
    assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}\nstderr: {result.stderr}"

    # Should not have Pydantic warnings
    assert "UnsupportedFieldAttributeWarning" not in result.stderr, (
        f"Found Pydantic warnings in stderr:\n{result.stderr}"
    )
    assert "pydantic._internal._generate_schema" not in result.stderr, (
        f"Found Pydantic warnings in stderr:\n{result.stderr}"
    )

    # Should have gotten far enough to show "Dry run" message
    assert "Dry run" in result.stdout or "Dry run" in result.stderr, (
        f"Expected 'Dry run' message not found.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_launch_two_token_syntax():
    """Test that launch.py supports two-token syntax like 'train arena'."""
    launch_script = get_launch_script_path()

    # Test two-token syntax
    result = subprocess.run(
        [
            sys.executable,
            str(launch_script),
            "train",
            "arena",
            "run=test_two_token",
            "--dry-run",
            "--skip-git-check",
        ],
        capture_output=True,
        text=True,
    )

    # Should exit cleanly
    assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}\nstderr: {result.stderr}"

    # Should have 'Dry run' message
    output = result.stdout + result.stderr
    assert "Dry run" in output, f"Expected 'Dry run' message in output:\n{output}"


def test_launch_invalid_run_name():
    """Test that launch.py rejects invalid run names (Sky cluster naming rules).

    Sky requires run names to:
    - Start with a letter
    - Contain only letters, numbers, dashes, underscores, or dots
    - End with a letter or number
    """
    launch_script = get_launch_script_path()

    # Run name starting with number should fail
    result = subprocess.run(
        [
            sys.executable,
            str(launch_script),
            "arena.train",
            "run=123invalid",
            "--dry-run",
            "--skip-git-check",
        ],
        capture_output=True,
        text=True,
    )

    # Should fail validation
    assert result.returncode != 0, "Expected non-zero exit code for invalid run name"

    # Should mention the validation error
    output = result.stdout + result.stderr
    assert "Invalid run name" in output or "must" in output.lower(), (
        f"Expected validation error message not found in output:\n{output}"
    )


def test_launch_invalid_module_path():
    """Test that launch.py validates module paths."""
    launch_script = get_launch_script_path()

    # Use a module path that doesn't exist
    result = subprocess.run(
        [
            sys.executable,
            str(launch_script),
            "nonexistent.module.path",
            "run=test",
            "--dry-run",
            "--skip-git-check",
        ],
        capture_output=True,
        text=True,
    )

    # Should fail validation
    assert result.returncode != 0, "Expected non-zero exit code for invalid module path"


def test_launch_dump_config():
    """Test that launch.py can dump configuration without launching."""
    launch_script = get_launch_script_path()

    # Test YAML dump
    result = subprocess.run(
        [
            sys.executable,
            str(launch_script),
            "arena.train",
            "run=test_dump",
            "--dump-config=yaml",
            "--skip-git-check",
        ],
        capture_output=True,
        text=True,
    )

    # Should not fail (exit code 0 or None since it's just dumping config)
    # Note: --dump-config doesn't explicitly call sys.exit(), so check for clean execution
    assert "Traceback" not in result.stderr, f"Unexpected error:\n{result.stderr}"

    # Should contain YAML-like content
    output = result.stdout
    assert "name:" in output or "resources:" in output, f"Expected YAML config output not found:\n{output}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
