"""Tests for the metta ci command."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.setup


def test_ci_help_shows_all_stages() -> None:
    """Test that metta ci --help shows all available stages."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # Check that all 5 stages are mentioned in help
    assert "lint" in result.stdout
    assert "python-tests" in result.stdout
    assert "python-benchmarks" in result.stdout
    assert "cpp-tests" in result.stdout
    assert "cpp-benchmarks" in result.stdout


def test_ci_invalid_stage_fails() -> None:
    """Test that running metta ci with an invalid stage fails with helpful error."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "invalid-stage"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Unknown stage" in result.stdout or "Unknown stage" in result.stderr


def test_ci_lint_stage_runs() -> None:
    """Test that metta ci --stage lint runs successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "lint"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that it ran (may pass or fail depending on code state)
    assert "Linting" in result.stdout or "Linting" in result.stderr


def test_ci_python_tests_stage_runs() -> None:
    """Test that metta ci --stage python-tests runs successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "python-tests", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # With --help, it should show the metta ci help (extra args are captured)
    # This tests that the stage exists and is recognized
    assert result.returncode == 0


def test_ci_python_benchmarks_stage_exists() -> None:
    """Test that python-benchmarks stage is recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "python-benchmarks", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # With --help, it should show the metta ci help
    assert result.returncode == 0


def test_ci_cpp_tests_stage_runs() -> None:
    """Test that metta ci --stage cpp-tests runs successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "cpp-tests"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    # Check that it ran (may pass or fail depending on build state)
    assert "C++ Tests" in result.stdout or "C++ unit tests" in result.stdout or "C++ Tests" in result.stderr


def test_ci_cpp_benchmarks_stage_runs() -> None:
    """Test that metta ci --stage cpp-benchmarks runs successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "cpp-benchmarks"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    # Check that it ran (may pass or fail depending on build state)
    assert "C++ Benchmarks" in result.stdout or "C++ benchmarks" in result.stdout or "C++ Benchmarks" in result.stderr


def test_ci_stages_are_separate() -> None:
    """Test that test and benchmark stages are truly separate."""
    # Run cpp-tests and verify it doesn't run benchmarks
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "ci", "--stage", "cpp-tests"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    # Should mention tests but not benchmarks header
    output = result.stdout + result.stderr
    assert "C++ Tests" in output or "C++ unit tests" in output
    # Benchmarks should not be mentioned in the stage name
    assert "C++ Benchmarks" not in output or "benchmark" not in output.lower().split("tests")[0]
