#!/usr/bin/env -S uv run
"""Recipe validation framework for local and remote validation.

This module provides infrastructure for validating recipes/tools either locally
or remotely (via SkyPilot), organized by tool.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional

from devops.skypilot.utils.testing_helpers import (
    SkyPilotTestLauncher,
    TestCondition,
)


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
    validator: Optional[Callable] = None  # Function to validate success


@dataclass
class ToolValidation:
    """Configuration for validating a tool across multiple recipes."""

    name: str
    description: str
    recipes: list[RecipeValidation]


class RecipeValidator:
    """Manages validation of recipes/tools locally and remotely."""

    def __init__(self, base_name: str = "recipe_validation", skip_git_check: bool = False):
        self.base_name = base_name
        self.skip_git_check = skip_git_check
        self.remote_launcher: Optional[SkyPilotTestLauncher] = None

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
        # TODO: Implement local validation
        # This would run the recipe with limited timesteps and validate results
        return ValidationStatus.PENDING

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


# Define standard validation conditions
STANDARD_CONDITIONS = {
    "quick_validation": TestCondition(
        name="Quick Validation",
        extra_args=["trainer.total_timesteps=50000"],
        description="Quick validation with 50k timesteps",
        ci=False,
    ),
    "normal_completion": TestCondition(
        name="Normal Completion",
        extra_args=["trainer.total_timesteps=50000"],
        description="Exit normally after training completes",
        ci=False,
    ),
    "heartbeat_timeout": TestCondition(
        name="Heartbeat Timeout",
        extra_args=["-hb", "1"],
        description="Exit based on missing heartbeats (1 second timeout)",
        ci=False,
    ),
    "runtime_timeout": TestCondition(
        name="Runtime Timeout",
        extra_args=["-t", "0.03"],
        description="Exit based on timeout (0.03 hours = 1.8 minutes)",
        ci=True,
    ),
}


# Define tools and their associated recipe validations
TOOL_VALIDATIONS = {
    "train": ToolValidation(
        name="Training Tool",
        description="Validate training workflow across multiple recipes",
        recipes=[
            RecipeValidation(
                name="arena_basic_easy_shaped",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="Basic arena with easy shaping",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
            RecipeValidation(
                name="arena",
                module="experiments.recipes.arena.train",
                description="Standard arena recipe",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
            RecipeValidation(
                name="cvc_arena",
                module="experiments.recipes.cvc_arena.train",
                description="Cogs vs Clips recipe",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
            RecipeValidation(
                name="navigation",
                module="experiments.recipes.navigation.train",
                description="Navigation task",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
            RecipeValidation(
                name="navigation_sequence",
                module="experiments.recipes.navigation_sequence.train",
                description="Sequential navigation task",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
            RecipeValidation(
                name="object_use",
                module="experiments.recipes.object_use.train",
                description="Object Use task",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
            RecipeValidation(
                name="icl_ordered_chains",
                module="experiments.recipes.in_context_learning.ordered_chains.train",
                description="In-context learning resource chain",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["quick_validation"],
            ),
        ],
    ),
    "cluster": ToolValidation(
        name="Cluster Configuration",
        description="Validate cluster configurations and exit conditions",
        recipes=[
            RecipeValidation(
                name="1n_normal",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="1 node, normal completion",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["normal_completion"],
            ),
            RecipeValidation(
                name="1n_runtime_timeout",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="1 node, runtime timeout",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["runtime_timeout"],
            ),
            RecipeValidation(
                name="2n_runtime_timeout",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="2 nodes, runtime timeout",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["runtime_timeout"],
            ),
            RecipeValidation(
                name="4n_runtime_timeout",
                module="experiments.recipes.arena_basic_easy_shaped.train",
                description="4 nodes, runtime timeout",
                location=ValidationLocation.REMOTE,
                condition=STANDARD_CONDITIONS["runtime_timeout"],
            ),
        ],
    ),
}
