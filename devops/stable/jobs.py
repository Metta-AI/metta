"""Defines the actual jobs that are run when validating a release."""

from __future__ import annotations

import os

from metta.jobs.job_config import JobConfig
from recipes.validation.ci_suite import get_ci_jobs
from recipes.validation.stable_suite import get_stable_jobs


def ci_job(name: str, cmd: list[str], timeout_s: int = 1800) -> JobConfig:
    """Create a CI job that runs a shell command."""
    return JobConfig(name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": cmd})


def get_all_jobs(version: str) -> list[JobConfig]:
    """Define all release validation jobs.

    Combines CI smoke tests and stable-specific validation tests.

    Args:
        version: Version prefix for job names (e.g., "v0.1.0")

    Returns:
        List of fully configured job configs
    """
    # Basic CI checks (python/cpp tests)
    python_ci = ci_job("python_ci", ["metta", "pytest", "--ci"])
    cpp_ci = ci_job("cpp_ci", ["metta", "cpptest", "--test"])
    cpp_benchmark = ci_job("cpp_benchmark", ["metta", "cpptest", "--benchmark"])

    # Recipe CI smoke tests
    user = os.environ.get("USER", "unknown")
    ci_recipe_jobs, _group = get_ci_jobs(f"{user}.stable.{version}")

    # Stable-specific long-running tests
    stable_recipe_jobs = get_stable_jobs(f"{user}.stable.{version}")

    return [python_ci, cpp_ci, cpp_benchmark] + ci_recipe_jobs + stable_recipe_jobs
