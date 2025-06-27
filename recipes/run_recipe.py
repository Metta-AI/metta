#!/usr/bin/env -S uv run
"""
Recipe runner for Metta training recipes.
Provides standardized execution with better error handling and CI integration.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class RecipeRunner:
    def __init__(self, recipe_name: str, dry_run: bool = False, timeout_hours: Optional[float] = None):
        self.recipe_name = recipe_name
        self.dry_run = dry_run
        self.timeout_hours = timeout_hours
        self.recipes_dir = Path(__file__).parent
        self.repo_root = self.recipes_dir.parent

    def get_recipe_config(self) -> Dict:
        """Load recipe configuration from YAML file."""
        config_path = self.recipes_dir / f"{self.recipe_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Recipe config not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def build_launch_command(self, config: Dict) -> List[str]:
        """Build the launch command from recipe configuration."""
        cmd = [
            str(self.repo_root / "devops" / "skypilot" / "launch.py"),
            "train",
            f"run=$USER.{config['run_suffix']}",
            f"trainer={config['trainer']}",
            f"trainer.curriculum={config['curriculum']}",
            "--gpus=1"
        ]

        # Add environment overrides
        if 'env_overrides' in config:
            for key, value in config['env_overrides'].items():
                cmd.append(f"+trainer.env_overrides.{key}={value}")

        # Add initial policy if specified
        if 'initial_policy_uri' in config:
            cmd.append(f"trainer.initial_policy.uri={config['initial_policy_uri']}")

        # Add timeout if specified
        if self.timeout_hours:
            cmd.extend(["--timeout-hours", str(self.timeout_hours)])

        return cmd

    def run_recipe(self) -> int:
        """Execute the recipe."""
        try:
            config = self.get_recipe_config()
            cmd = self.build_launch_command(config)

            print(f"Running recipe: {self.recipe_name}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Expected performance: {config.get('expected_performance', 'N/A')}")
            print(f"Baseline run: {config.get('baseline_run', 'N/A')}")

            if self.dry_run:
                print("DRY RUN - Command would be:")
                print(" ".join(cmd))
                return 0

            # Change to repo root and execute
            os.chdir(self.repo_root)
            result = subprocess.run(cmd, capture_output=False)

            return result.returncode

        except Exception as e:
            print(f"Error running recipe {self.recipe_name}: {e}")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Run Metta training recipes")
    parser.add_argument("recipe", help="Recipe name (without .yaml extension)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--timeout-hours", type=float, help="Timeout in hours for the training job")
    parser.add_argument("--list", action="store_true", help="List available recipes")

    args = parser.parse_args()

    if args.list:
        recipes_dir = Path(__file__).parent
        yaml_files = list(recipes_dir.glob("*.yaml"))
        print("Available recipes:")
        for yaml_file in yaml_files:
            print(f"  {yaml_file.stem}")
        return 0

    runner = RecipeRunner(args.recipe, args.dry_run, args.timeout_hours)
    return runner.run_recipe()


if __name__ == "__main__":
    sys.exit(main())
