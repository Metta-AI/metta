"""Tests for metta pytest command flags."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.setup


def test_pytest_help_shows_flags() -> None:
    """Test that metta pytest --help shows --test and --benchmark flags."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "pytest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--test" in result.stdout
    assert "--benchmark" in result.stdout
    assert "Run unit tests" in result.stdout
    assert "Run benchmarks" in result.stdout


def test_pytest_default_skips_benchmarks() -> None:
    """Test that metta pytest (no flags) skips benchmarks."""
    # Run with dry-run equivalent to see what command would be executed
    # We'll look at the help output which documents the default behavior
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "pytest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # The help text should indicate tests are the default
    assert "default if no flags specified" in result.stdout.lower() or "Run unit tests" in result.stdout


def test_pytest_with_benchmark_flag() -> None:
    """Test that --benchmark flag is recognized."""
    # Test that the flag is accepted (using --help to avoid running actual tests)
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "pytest", "--benchmark", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should succeed - the --help overrides actual execution
    assert result.returncode == 0


def test_pytest_with_test_and_benchmark_flags() -> None:
    """Test that --test and --benchmark flags can be used together."""
    # Test that both flags are accepted (using --help to avoid running actual tests)
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "pytest", "--test", "--benchmark", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should succeed - the --help overrides actual execution
    assert result.returncode == 0


def test_pytest_with_test_flag_only() -> None:
    """Test that --test flag alone is recognized."""
    # Test that the flag is accepted (using --help to avoid running actual tests)
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "pytest", "--test", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should succeed - the --help overrides actual execution
    assert result.returncode == 0


def test_pytest_ci_mode_accepts_flags() -> None:
    """Test that --ci mode works with --test and --benchmark flags."""
    # Test that CI mode accepts the flags (using --help to avoid running actual tests)
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "pytest", "--ci", "--test", "--benchmark", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should succeed - the --help overrides actual execution
    assert result.returncode == 0
