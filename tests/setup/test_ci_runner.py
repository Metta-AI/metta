"""Tests for the metta ci command."""

from __future__ import annotations

import subprocess
import sys

import pytest
import typer

from metta.setup.tools.ci_runner import ALLOWED_SKIP_PACKAGES, _normalize_python_stage_args

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
    # Check that all 4 stages are mentioned in help
    assert "lint" in result.stdout
    assert "python-tests-and-benchmarks" in result.stdout
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


def test_ci_requires_stage_for_extra_args() -> None:
    """Extra args without --stage should fail fast."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "metta.setup.metta_cli",
            "ci",
            "--",
            "--skip-package",
            "tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "Extra arguments require specifying a --stage" in combined


def test_ci_non_python_stage_rejects_extra_args() -> None:
    """Stages that do not accept extra args should error."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "metta.setup.metta_cli",
            "ci",
            "--stage",
            "lint",
            "--",
            "--skip-package",
            "tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "does not accept extra arguments" in combined


def test_normalize_python_stage_args_allows_known_package() -> None:
    package = next(iter(ALLOWED_SKIP_PACKAGES))
    result = _normalize_python_stage_args(["--skip-package", package])
    assert result == ["--skip-package", package]


def test_normalize_python_stage_args_rejects_unknown_package() -> None:
    with pytest.raises(typer.Exit):
        _normalize_python_stage_args(["--skip-package", "does-not-exist"])
