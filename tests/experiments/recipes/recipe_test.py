#!/usr/bin/env -S uv run
"""
Test script for validating recipes.

Usage:
    recipe_test.py launch              # Launch test jobs
    recipe_test.py check               # Check test results
    recipe_test.py check -l            # Check with detailed logs
"""

import sys

from devops.skypilot.utils.testing_helpers import (
    BaseTestRunner,
    SkyPilotTestLauncher,
    TestCondition,
)
from metta.common.util.text_styles import bold, cyan, yellow

# Recipe configurations
RECIPES = {
    "in_context_learning/ordered_chains": {
        "module": "experiments.recipes.in_context_learning.ordered_chains.train",
        "description": "In-context learning resource chain",
    },
    "arena_basic_easy_shaped": {
        "module": "experiments.recipes.arena_basic_easy_shaped.train",
        "description": "Basic arena with easy shaping",
    },
    "arena": {
        "module": "experiments.recipes.arena.train",
        "description": "Standard arena recipe",
    },
    "cvc_arena": {
        "module": "experiments.recipes.cvc_arena.train",
        "description": "Cogs vs Clips recipe",
    },
    "navigation_sequence": {
        "module": "experiments.recipes.navigation_sequence.train",
        "description": "Sequential navigation task",
    },
    "navigation": {
        "module": "experiments.recipes.navigation.train",
        "description": "Navigation task",
    },
    "object_use": {
        "module": "experiments.recipes.object_use.train",
        "description": "Object Use task",
    },
}

# Test condition - normal completion with short training
TEST_CONDITION = TestCondition(
    name="Normal Completion",
    extra_args=["trainer.total_timesteps=50000"],
    description="Exit normally after 50k timesteps",
    ci=False,
)

# Base configuration
BASE_ARGS = ["--no-spot", "--gpus=4", "--nodes", "1"]


class RecipeTestRunner(BaseTestRunner):
    """Test runner for recipe validation tests."""

    def __init__(self):
        super().__init__(
            prog_name="recipe_test.py",
            description="Recipe test launcher and checker",
            default_output_file="recipe_test_jobs.json",
            default_base_name="recipe_test",
            test_type="Recipe Test",
        )

    def launch_tests(self, args):
        """Launch recipe test jobs."""
        # Create launcher
        launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

        # Check git state
        if not launcher.check_git_state():
            sys.exit(1)

        # Show test configuration
        print(f"\n{bold('=== Recipe Test Configuration ===')}")
        print(f"{cyan('Recipes to test:')}")
        for recipe_key, recipe in RECIPES.items():
            print(f"  • {yellow(recipe_key)}: {recipe['description']}")
        print(f"\n{cyan('Test condition:')} {TEST_CONDITION.description}")
        print(f"{cyan('Nodes:')} 1")
        print(f"{cyan('CI tests:')} Disabled")
        print(f"\n{cyan('Total jobs to launch:')} {len(RECIPES)}")
        print(f"{cyan('Output file:')} {args.output_file}")

        # Launch jobs
        for recipe_key, recipe in RECIPES.items():
            # Generate run name
            run_name = launcher.generate_run_name(recipe_key)

            # Test config for tracking
            test_config = {
                "recipe": recipe_key,
                "description": recipe["description"],
                "timesteps": 50000,
                "nodes": 1,
                "ci_tests_enabled": False,
            }

            # Launch the job
            launcher.launch_job(
                module=recipe["module"],
                run_name=run_name,
                base_args=BASE_ARGS,
                extra_args=TEST_CONDITION.extra_args,
                test_config=test_config,
                enable_ci_tests=False,
            )

        # Save results
        output_path = launcher.save_results(args.output_file)
        print(f"{cyan('Results saved to:')} {output_path.absolute()}")

        # Print summary
        launcher.print_summary()

        # Exit with error if any launches failed
        if launcher.failed_launches:
            sys.exit(1)


def main():
    """Main entry point."""
    runner = RecipeTestRunner()
    runner.run()


if __name__ == "__main__":
    main()
