#!/usr/bin/env -S uv run
"""Test multiple recipes in parallel.

Usage:
    # Test all recipes
    ./devops/test_recipes.py

    # Test specific recipes
    ./devops/test_recipes.py arena navigation

    # Custom config
    ./devops/test_recipes.py --timesteps 100000 --gpus 2
"""

from __future__ import annotations

import argparse

from devops.job_dispatcher import JobDispatcher
from devops.job_runner import RemoteJob

# Available recipes
RECIPES = {
    "arena": {
        "module": "experiments.recipes.arena.train",
        "description": "Standard arena recipe",
    },
    "arena_basic_easy_shaped": {
        "module": "experiments.recipes.arena_basic_easy_shaped.train",
        "description": "Basic arena with easy shaping",
    },
    "navigation": {
        "module": "experiments.recipes.navigation.train",
        "description": "Navigation task",
    },
    "navigation_sequence": {
        "module": "experiments.recipes.navigation_sequence.train",
        "description": "Sequential navigation task",
    },
    "in_context_learning/ordered_chains": {
        "module": "experiments.recipes.in_context_learning.ordered_chains.train",
        "description": "In-context learning resource chain",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Test multiple recipes in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "recipes",
        nargs="*",
        help="Recipes to test (default: all)",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total timesteps to train (default: 50000)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="GPUs per job (default: 4)",
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Nodes per job (default: 1)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600)",
    )

    parser.add_argument(
        "--name",
        default="recipe_test",
        help="Name for this test run",
    )

    args = parser.parse_args()

    # Determine which recipes to test
    if args.recipes:
        recipes_to_test = {k: v for k, v in RECIPES.items() if k in args.recipes}
        if not recipes_to_test:
            print(f"Error: Unknown recipes {args.recipes}")
            print(f"Available: {', '.join(RECIPES.keys())}")
            return 1
    else:
        recipes_to_test = RECIPES

    # Show what we'll test
    print("=" * 80)
    print(f"Recipe Test Runner: {args.name}")
    print("=" * 80)
    print(f"Recipes:   {len(recipes_to_test)}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Config:    {args.nodes} nodes × {args.gpus} GPUs")
    print(f"Timeout:   {args.timeout}s")
    print()

    for name, info in recipes_to_test.items():
        print(f"  • {name}: {info['description']}")

    print()

    # Create dispatcher
    dispatcher = JobDispatcher(name=args.name)

    # Create jobs
    base_args = ["--no-spot", f"--gpus={args.gpus}", "--nodes", str(args.nodes)]

    for recipe_name, recipe_info in recipes_to_test.items():
        job = RemoteJob(
            name=recipe_name.replace("/", "_"),
            module=recipe_info["module"],
            args=[f"trainer.total_timesteps={args.timesteps}"],
            base_args=base_args,
            timeout_s=args.timeout,
        )
        dispatcher.add_job(job)

    # Run all jobs
    print("Submitting jobs...")
    dispatcher.run_all()

    # Wait for completion
    print()
    results = dispatcher.wait_all(timeout_s=args.timeout)

    # Print summary
    dispatcher.print_summary()

    # Exit with error if any failed
    failed = sum(1 for r in results.values() if not r.success)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    exit(main())
