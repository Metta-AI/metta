"""Defines the actual jobs that are run when validating a release."""

from __future__ import annotations

import os

from metta.jobs.job_config import JobConfig
from recipes.validation.ci_suite import get_ci_jobs
from recipes.validation.stable_suite import get_stable_jobs


def ci_job(name: str, cmd: list[str], timeout_s: int = 1800) -> JobConfig:
    """Create a CI job that runs a shell command."""
    import shlex

    cmd_string = shlex.join(cmd)
    return JobConfig(name=name, cmd=cmd_string, timeout_s=timeout_s)


def get_all_jobs(version: str) -> list[JobConfig]:
    """Define all release validation jobs.

    Combines CI smoke tests and stable-specific validation tests.
    All jobs use consistent naming: {user}.stable.{version}.{job_name}

    Args:
        version: Version string (e.g., "2025.11.10-142732")

    Returns:
        List of fully configured job configs with names like:
            "jack.stable.2025.11.10-142732.python_ci"
            "jack.stable.2025.11.10-142732.arena_train"
    """
    # Use consistent prefix across all jobs: user.stable.version
    user = os.environ.get("USER", "unknown")
    prefix = f"{user}.stable.{version}"

    # Basic CI checks (python/cpp tests)
    python_ci = ci_job(f"{prefix}.python_ci", ["metta", "pytest", "--ci"])
    cpp_ci = ci_job(f"{prefix}.cpp_ci", ["metta", "cpptest", "--test"])
    cpp_benchmark = ci_job(f"{prefix}.cpp_benchmark", ["metta", "cpptest", "--benchmark"])

    # Recipe CI smoke tests
    ci_recipe_jobs, _group = get_ci_jobs(prefix=prefix)

    # Stable-specific long-running tests
    stable_recipe_jobs = get_stable_jobs(prefix=prefix)

    return [python_ci, cpp_ci, cpp_benchmark] + ci_recipe_jobs + stable_recipe_jobs
