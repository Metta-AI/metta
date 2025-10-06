#!/usr/bin/env python3
"""Benchmark script to test 4 transformer variants across 5 difficulty levels.

This script trains and evaluates 20 combinations (4 architectures × 5 levels):
- ViT (Default perceiver-based architecture)
- ViT + Sliding Transformer
- Transformer (standard)
- Fast (fast LSTM-based baseline)

Each architecture is tested on all 5 difficulty levels:
1. Basic - Maximum reward shaping
2. Easy - Moderate reward shaping
3. Medium - Combat enabled with low rewards
4. Hard - Sparse rewards
5. Expert - No intermediate rewards with curriculum

Usage:
    uv run experiments/recipes/benchmark_architectures/run_benchmark.py

Options:
    --max-timesteps: Total timesteps per training run (default: 1000000)
    --skip-training: Skip training and only evaluate existing checkpoints
    --architectures: Comma-separated list of architectures to test (default: all)
    --levels: Comma-separated list of levels to test (default: all)
    --output-dir: Directory for results (default: ./train_dir/benchmark)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from metta.agent.policies.fast import FastConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig


ARCHITECTURES = {
    "vit": ViTDefaultConfig(),
    "vit_sliding": ViTSlidingTransConfig(),
    "transformer": TransformerPolicyConfig(),
    "fast": FastConfig(),
}

LEVELS = {
    "level_1_basic": "Level 1 - Basic (max shaping)",
    "level_2_easy": "Level 2 - Easy (moderate shaping)",
    "level_3_medium": "Level 3 - Medium (combat + low rewards)",
    "level_4_hard": "Level 4 - Hard (sparse rewards)",
    "level_5_expert": "Level 5 - Expert (no intermediate rewards)",
}


class BenchmarkRunner:
    """Manages benchmark execution across architectures and difficulty levels."""

    def __init__(
        self,
        max_timesteps: int = 1000000,
        output_dir: str = "./train_dir/benchmark",
        skip_training: bool = False,
    ):
        self.max_timesteps = max_timesteps
        self.output_dir = Path(output_dir)
        self.skip_training = skip_training
        self.results: dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "max_timesteps": max_timesteps,
            "runs": [],
        }

    def run_training(
        self, architecture_name: str, level: str, run_name: str
    ) -> dict[str, Any]:
        """Run training for a specific architecture and level combination.

        Args:
            architecture_name: Name of the architecture (e.g., 'vit', 'transformer')
            level: Difficulty level (e.g., 'level_1_basic')
            run_name: Unique name for this run

        Returns:
            Dictionary containing run results and metadata
        """
        recipe_path = f"experiments.recipes.benchmark_architectures.{level}.train"

        print(f"\n{'='*80}")
        print(f"Training: {architecture_name} on {LEVELS[level]}")
        print(f"Run name: {run_name}")
        print(f"Recipe: {recipe_path}")
        print(f"Max timesteps: {self.max_timesteps}")
        print(f"{'='*80}\n")

        start_time = time.time()

        cmd = [
            "uv",
            "run",
            "./tools/run.py",
            recipe_path,
            f"run={run_name}",
            f"trainer.total_timesteps={self.max_timesteps}",
        ]

        result = {
            "architecture": architecture_name,
            "level": level,
            "run_name": run_name,
            "command": " ".join(cmd),
            "start_time": datetime.now().isoformat(),
        }

        try:
            subprocess.run(cmd, check=True)
            result["status"] = "success"
            result["duration_seconds"] = time.time() - start_time
            print(f"\n✓ Training completed in {result['duration_seconds']:.1f}s\n")
        except subprocess.CalledProcessError as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["duration_seconds"] = time.time() - start_time
            print(f"\n✗ Training failed: {e}\n")

        return result

    def run_evaluation(
        self, architecture_name: str, level: str, run_name: str
    ) -> dict[str, Any]:
        """Run evaluation for a trained model.

        Args:
            architecture_name: Name of the architecture
            level: Difficulty level
            run_name: Unique name for this run

        Returns:
            Dictionary containing evaluation results
        """
        recipe_path = f"experiments.recipes.benchmark_architectures.{level}.evaluate"
        policy_uri = f"file://./train_dir/{run_name}/checkpoints"

        print(f"\n{'='*80}")
        print(f"Evaluating: {architecture_name} on {LEVELS[level]}")
        print(f"Policy URI: {policy_uri}")
        print(f"{'='*80}\n")

        start_time = time.time()

        cmd = [
            "uv",
            "run",
            "./tools/run.py",
            recipe_path,
            f"policy_uri={policy_uri}",
        ]

        result = {
            "architecture": architecture_name,
            "level": level,
            "run_name": run_name,
            "policy_uri": policy_uri,
            "command": " ".join(cmd),
            "start_time": datetime.now().isoformat(),
        }

        try:
            subprocess.run(cmd, check=True)
            result["status"] = "success"
            result["duration_seconds"] = time.time() - start_time
            print(f"\n✓ Evaluation completed in {result['duration_seconds']:.1f}s\n")
        except subprocess.CalledProcessError as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["duration_seconds"] = time.time() - start_time
            print(f"\n✗ Evaluation failed: {e}\n")

        return result

    def run_benchmark(
        self, architectures: list[str] | None = None, levels: list[str] | None = None
    ):
        """Run the full benchmark suite.

        Args:
            architectures: List of architecture names to test (None = all)
            levels: List of difficulty levels to test (None = all)
        """
        architectures = architectures or list(ARCHITECTURES.keys())
        levels = levels or list(LEVELS.keys())

        total_runs = len(architectures) * len(levels)
        current_run = 0

        print(f"\n{'='*80}")
        print(f"BENCHMARK CONFIGURATION")
        print(f"{'='*80}")
        print(f"Architectures: {', '.join(architectures)}")
        print(f"Levels: {', '.join(levels)}")
        print(f"Total runs: {total_runs}")
        print(f"Max timesteps per run: {self.max_timesteps}")
        print(f"Output directory: {self.output_dir}")
        print(f"Skip training: {self.skip_training}")
        print(f"{'='*80}\n")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for architecture_name in architectures:
            for level in levels:
                current_run += 1
                run_name = f"benchmark_{architecture_name}_{level}_{int(time.time())}"

                print(f"\n{'#'*80}")
                print(f"# Run {current_run}/{total_runs}")
                print(f"{'#'*80}\n")

                run_result = {
                    "run_number": current_run,
                    "total_runs": total_runs,
                }

                # Training phase
                if not self.skip_training:
                    training_result = self.run_training(
                        architecture_name, level, run_name
                    )
                    run_result["training"] = training_result

                    # Only evaluate if training succeeded
                    if training_result["status"] == "success":
                        eval_result = self.run_evaluation(
                            architecture_name, level, run_name
                        )
                        run_result["evaluation"] = eval_result
                    else:
                        print(
                            f"⚠ Skipping evaluation for {run_name} due to training failure"
                        )
                        run_result["evaluation"] = {
                            "status": "skipped",
                            "reason": "training_failed",
                        }
                else:
                    # Evaluation only mode
                    eval_result = self.run_evaluation(architecture_name, level, run_name)
                    run_result["evaluation"] = eval_result

                self.results["runs"].append(run_result)

                # Save intermediate results after each run
                self._save_results()

        self.results["end_time"] = datetime.now().isoformat()
        self._save_results()
        self._print_summary()

    def _save_results(self):
        """Save results to JSON file."""
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Results saved to {results_file}")

    def _print_summary(self):
        """Print a summary of all benchmark results."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*80}\n")

        successful_training = sum(
            1
            for r in self.results["runs"]
            if "training" in r and r["training"]["status"] == "success"
        )
        successful_eval = sum(
            1
            for r in self.results["runs"]
            if "evaluation" in r and r["evaluation"]["status"] == "success"
        )
        total_runs = len(self.results["runs"])

        print(f"Total runs: {total_runs}")
        if not self.skip_training:
            print(f"Successful training runs: {successful_training}/{total_runs}")
        print(f"Successful evaluation runs: {successful_eval}/{total_runs}")
        print(f"\nResults saved to: {self.output_dir / 'benchmark_results.json'}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run architecture benchmark across difficulty levels"
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=1000000,
        help="Total timesteps per training run (default: 1000000)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate existing checkpoints",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        help="Comma-separated list of architectures to test (default: all)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        help="Comma-separated list of levels to test (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./train_dir/benchmark",
        help="Directory for results (default: ./train_dir/benchmark)",
    )

    args = parser.parse_args()

    # Parse architecture and level filters
    architectures = (
        [a.strip() for a in args.architectures.split(",")]
        if args.architectures
        else None
    )
    levels = [l.strip() for l in args.levels.split(",")] if args.levels else None

    # Validate architecture names
    if architectures:
        invalid_archs = [a for a in architectures if a not in ARCHITECTURES]
        if invalid_archs:
            print(f"Error: Invalid architectures: {invalid_archs}")
            print(f"Valid architectures: {list(ARCHITECTURES.keys())}")
            sys.exit(1)

    # Validate level names
    if levels:
        invalid_levels = [l for l in levels if l not in LEVELS]
        if invalid_levels:
            print(f"Error: Invalid levels: {invalid_levels}")
            print(f"Valid levels: {list(LEVELS.keys())}")
            sys.exit(1)

    runner = BenchmarkRunner(
        max_timesteps=args.max_timesteps,
        output_dir=args.output_dir,
        skip_training=args.skip_training,
    )

    runner.run_benchmark(architectures=architectures, levels=levels)


if __name__ == "__main__":
    main()
