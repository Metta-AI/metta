#!/usr/bin/env -S uv run
"""
Test script for validating recipes.

Usage:
    recipe_test.py launch              # Launch test jobs
    recipe_test.py launch --staging    # Launch staging jobs (long arena_basic run)
    recipe_test.py check               # Check test results
    recipe_test.py check -l            # Check with detailed logs
"""

import sys
from argparse import ArgumentParser

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

# Override config for long arena_basic test (for stable qualification)
ARENA_BASIC_STAGING_CONDITION = TestCondition(
    name="Extended Arena Basic Run",
    extra_args=["trainer.total_timesteps=2000000000"],
    description="Extended run for release qualification (~2B timesteps on 4x4 GPUs cluster)",
    ci=False,
)

ARENA_BASIC_STAGING_ARGS = ["--no-spot", "--gpus=4", "--nodes", "4"]  # 4 nodes, 4 GPUs each (16 total)


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

    def add_custom_launch_args(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--staging",
            action="store_true",
            help="Run extended staging tests (arena_basic_easy_shaped long run on larger cluster).",
        )

    def launch_tests(self, args):
        launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

        if not launcher.check_git_state():
            sys.exit(1)

        print(f"\n{bold('=== Recipe Test Configuration ===')}")
        print(f"{cyan('Recipes to test:')}")
        for recipe_key, recipe in RECIPES.items():
            print(f"  • {yellow(recipe_key)}: {recipe['description']}")
        print(f"\n{cyan('Test condition:')} {TEST_CONDITION.description}")
        print(f"{cyan('Nodes:')} 1")
        print(f"{cyan('CI tests:')} Disabled")
        print(f"\n{cyan('Total jobs to launch:')} {len(RECIPES)}")
        print(f"{cyan('Output file:')} {args.output_file}")

        if args.staging:
            print(cyan("Staging mode enabled: arena_basic_easy_shaped will run long test"))

        for recipe_key, recipe in RECIPES.items():
            run_name = launcher.generate_run_name(recipe_key)

            # Defaults
            test_condition = TEST_CONDITION
            base_args = BASE_ARGS

            # Override for staging mode
            if args.staging and recipe_key == "arena_basic_easy_shaped":
                test_condition = ARENA_BASIC_STAGING_CONDITION
                base_args = ARENA_BASIC_STAGING_ARGS

            test_config = {
                "recipe": recipe_key,
                "description": recipe["description"],
                "timesteps": (2000000000 if args.staging and recipe_key == "arena_basic_easy_shaped" else 50000),
                "nodes": 4 if args.staging and recipe_key == "arena_basic_easy_shaped" else 1,
                "gpus_per_node": 4,  # explicit
                "ci_tests_enabled": False,
                "staging_mode": args.staging,
            }

            launcher.launch_job(
                module=recipe["module"],
                run_name=run_name,
                base_args=base_args,
                extra_args=test_condition.extra_args,
                test_config=test_config,
                enable_ci_tests=False,
            )

        output_path = launcher.save_results(args.output_file)
        print(f"{cyan('Results saved to:')} {output_path.absolute()}")
        launcher.print_summary()

        if launcher.failed_launches:
            sys.exit(1)


def main():
    """Main entry point."""
    runner = RecipeTestRunner()
    runner.run()


if __name__ == "__main__":
    main()
