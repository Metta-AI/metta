"""Defines the actual jobs that are run when validating a release."""

from __future__ import annotations

import os

from metta.jobs.job_config import JobConfig
from recipes.validation.ci_suite import get_ci_jobs
from recipes.validation.stable_suite import get_stable_jobs


def get_all_jobs(version: str) -> list[JobConfig]:
    """Define all release validation jobs.

    Combines CI smoke tests and stable-specific validation tests.
    All jobs use consistent naming: {user}.stable.{version}.{job_name}

    Note: Basic CI checks (pytest, cpptest) should be run separately via
    `metta ci` before release validation. This function only includes
    recipe-based validation jobs.

    Args:
        version: Version string (e.g., "2025.11.10-142732")

    Returns:
        List of fully configured job configs with names like:
            "jack.stable.2025.11.10-142732.arena_train"
    """
    # Use consistent prefix across all jobs: user.stable.version
    user = os.environ.get("USER", "unknown")
    prefix = f"{user}.stable.{version}"

    # Recipe CI smoke tests
    ci_recipe_jobs, _group = get_ci_jobs(prefix=prefix)

    # Stable-specific long-running tests
    stable_recipe_jobs = get_stable_jobs(prefix=prefix)

    return ci_recipe_jobs + stable_recipe_jobs
