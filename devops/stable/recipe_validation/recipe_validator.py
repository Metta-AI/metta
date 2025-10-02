#!/usr/bin/env -S uv run
"""Recipe validation framework for local and remote validation.

This module provides infrastructure for validating recipes/tools either locally
or remotely (via SkyPilot), organized by tool.
"""

import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from devops.skypilot.utils.testing_helpers import (
    SkyPilotTestLauncher,
    TestCondition,
)
from metta.common.util.text_styles import green, red


class ValidationLocation(StrEnum):
    """Where to run the validation."""

    LOCAL = "local"
    REMOTE = "remote"


class ValidationStatus(StrEnum):
    """Validation execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RecipeValidation:
    """Configuration for validating a single recipe."""

    name: str
    module: str
    description: str
    location: ValidationLocation
    condition: TestCondition


@dataclass
class ToolValidation:
    """Configuration for validating a tool across multiple recipes."""

    name: str
    description: str
    recipes: list[RecipeValidation]


class RecipeValidator:
    """Manages validation of recipes/tools locally and remotely."""

    def __init__(
        self, base_name: str = "recipe_validation", skip_git_check: bool = False, repo_root: Optional[str] = None
    ):
        self.base_name = base_name
        self.skip_git_check = skip_git_check
        self.repo_root = repo_root or self._get_repo_root()
        self.remote_launcher: Optional[SkyPilotTestLauncher] = None
        self.local_results: list[dict] = []

    @staticmethod
    def _get_repo_root() -> str:
        """Get repository root dynamically via git."""
        try:
            result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, text=True, capture_output=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Not in a git repository") from e

    def _ensure_remote_launcher(self) -> SkyPilotTestLauncher:
        """Ensure remote launcher is initialized."""
        if self.remote_launcher is None:
            self.remote_launcher = SkyPilotTestLauncher(
                base_name=self.base_name,
                skip_git_check=self.skip_git_check,
            )
        return self.remote_launcher

    def validate_tool(self, tool: ToolValidation) -> dict[str, ValidationStatus]:
        """Run all validations for a tool.

        Returns:
            Dictionary mapping recipe names to validation status
        """
        results = {}

        for recipe in tool.recipes:
            if recipe.location == ValidationLocation.LOCAL:
                results[recipe.name] = self._validate_recipe_local(recipe)
            else:
                results[recipe.name] = self._validate_recipe_remote(recipe)

        return results

    def _validate_recipe_local(self, recipe: RecipeValidation) -> ValidationStatus:
        """Run a recipe validation locally."""
        print(f"\n  Running local validation: {recipe.name}")

        try:
            run_name = f"local_val_{recipe.name}_{int(time.time())}"
            cmd = [
                "uv",
                "run",
                "./tools/run.py",
                recipe.module,
                f"run={run_name}",
                *recipe.condition.extra_args,
            ]

            print(f"  Command: {' '.join(cmd)}")
            start_time = time.time()

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=self.repo_root)
            duration = time.time() - start_time

            # Store result
            self.local_results.append(
                {
                    "recipe": recipe.name,
                    "duration": duration,
                    "exit_code": result.returncode,
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }
            )

            if result.returncode != 0:
                print(f"  {red('✗ Failed')} - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[-200:]}")
                return ValidationStatus.FAILED

            print(f"  {green('✓ Passed')} - Duration: {duration:.1f}s")
            return ValidationStatus.PASSED

        except subprocess.TimeoutExpired:
            print(f"  {red('✗ Timeout')} - Exceeded 5 minutes")
            return ValidationStatus.FAILED
        except Exception as e:
            print(f"  {red('✗ Error')}: {str(e)}")
            return ValidationStatus.FAILED

    def _validate_recipe_remote(self, recipe: RecipeValidation) -> ValidationStatus:
        """Launch a recipe validation remotely via SkyPilot."""
        launcher = self._ensure_remote_launcher()

        # Generate run name
        run_name = launcher.generate_run_name(recipe.name)

        # Build config for tracking
        validation_config = {
            "recipe": recipe.name,
            "description": recipe.description,
            "location": recipe.location,
            "condition": recipe.condition.name,
        }

        # Launch the job
        job = launcher.launch_job(
            module=recipe.module,
            run_name=run_name,
            base_args=["--no-spot", "--gpus=4", "--nodes", "1"],
            extra_args=recipe.condition.extra_args,
            test_config=validation_config,
            enable_ci_tests=recipe.condition.ci,
        )

        # Return status based on launch success
        return ValidationStatus.RUNNING if job.success else ValidationStatus.FAILED


# Define tools and their associated recipe validations
TOOL_VALIDATIONS = {
    "train": ToolValidation(
        name="Training Tool",
        description="Validate training workflow",
        recipes=[
            # Local validation - fast feedback
            RecipeValidation(
                name="arena_local",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="Quick local training check",
                location=ValidationLocation.LOCAL,
                condition=TestCondition(
                    name="Local Quick",
                    extra_args=["trainer.total_timesteps=1000"],
                    description="1k timesteps local",
                    ci=True,
                ),
            ),
            # Remote validation - full cluster test
            RecipeValidation(
                name="arena_remote",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="Remote cluster training",
                location=ValidationLocation.REMOTE,
                condition=TestCondition(
                    name="Remote Standard",
                    extra_args=["trainer.total_timesteps=50000"],
                    description="50k timesteps remote",
                    ci=False,
                ),
            ),
        ],
    ),
    "cluster": ToolValidation(
        name="Cluster Configuration",
        description="Validate cluster exit conditions",
        recipes=[
            RecipeValidation(
                name="runtime_timeout",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="Verify timeout handling",
                location=ValidationLocation.REMOTE,
                condition=TestCondition(
                    name="Runtime Timeout",
                    extra_args=["-t", "0.03"],  # ~2 minutes
                    description="Exit via timeout",
                    ci=True,
                ),
            ),
        ],
    ),
}
