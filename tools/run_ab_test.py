#!/usr/bin/env python3
"""
CLI tool for running A/B test experiments.

Usage:
    python tools/run_ab_test.py path/to/experiment.py
    python tools/run_ab_test.py path/to/experiment.py --dry-run
    python tools/run_ab_test.py path/to/experiment.py --output-dir custom/output
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from metta.ab_test.runner import run_ab_test


def load_experiment_from_file(file_path: str) -> Any:
    """Load an experiment from a Python file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("experiment_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for experiment creation functions
    experiment = None

    # Try common function names
    for func_name in ["create_experiment", "get_experiment", "experiment"]:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if callable(func):
                experiment = func()
                break

    # If no function found, look for a variable named 'experiment'
    if experiment is None and hasattr(module, "experiment"):
        experiment = module.experiment

    if experiment is None:
        raise ValueError(
            f"No experiment found in {file_path}. "
            f"Expected a function named 'create_experiment', 'get_experiment', or 'experiment', "
            f"or a variable named 'experiment'."
        )

    return experiment


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run A/B test experiments from Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/run_ab_test.py metta/ab_test/examples/curriculum_comparison.py
  python tools/run_ab_test.py my_experiment.py --dry-run
  python tools/run_ab_test.py my_experiment.py --output-dir results/exp1
        """,
    )

    parser.add_argument("experiment_file", help="Path to Python file containing experiment definition")

    parser.add_argument("--dry-run", action="store_true", help="Show experiment configuration without running")

    parser.add_argument(
        "--output-dir",
        default="ab_test_results",
        help="Output directory for experiment results (default: ab_test_results)",
    )

    parser.add_argument("--parallel", action="store_true", help="Run variants in parallel (default: sequential)")

    parser.add_argument("--max-parallel", type=int, default=4, help="Maximum number of parallel runs (default: 4)")

    parser.add_argument("--no-retry", action="store_true", help="Don't retry failed runs")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        # Load experiment
        print(f"Loading experiment from: {args.experiment_file}")
        experiment = load_experiment_from_file(args.experiment_file)

        # Display experiment info
        print(f"\nExperiment: {experiment.name}")
        print(f"Description: {experiment.description}")
        print(f"Date: {experiment.date}")
        print(f"Variants: {list(experiment.variants.keys())}")
        print(f"Runs per variant: {experiment.runs_per_variant}")
        print(f"WandB Project: {experiment.wandb_project}")
        print(f"WandB Entity: {experiment.wandb_entity}")

        for name, variant in experiment.variants.items():
            print(f"\nVariant '{name}':")
            print(f"  Description: {variant.description}")
            print(f"  Overrides: {variant.overrides}")
            if variant.tags:
                print(f"  Tags: {variant.tags}")

        if args.dry_run:
            print("\nDRY RUN - No experiments will be executed")
            return 0

        # Run experiment
        print("\nStarting A/B test experiment...")
        results = run_ab_test(
            experiment,
            output_dir=args.output_dir,
            parallel_runs=args.parallel,
            max_parallel_runs=args.max_parallel,
            retry_failed_runs=not args.no_retry,
        )

        # Display results summary
        print("\nExperiment completed!")
        print(f"Results saved to: {args.output_dir}/{experiment.name}")

        for variant_name, runs in results.items():
            successful = sum(1 for r in runs if r["success"])
            total = len(runs)
            print(f"  {variant_name}: {successful}/{total} runs successful")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
