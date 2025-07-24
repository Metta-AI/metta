#!/usr/bin/env python3
"""Arena experiment implementation.

This demonstrates a simple experiment that launches arena training runs.
Based on the arena.sh recipe.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

from experiments.experiment import Experiment
from experiments.types import TrainingJob, TrainingJobConfig


class ArenaExperiment(Experiment):
    """Basic arena training experiment.

    Launches arena training with standard configuration.
    """

    def __init__(self, name: str = "arena_experiment"):
        super().__init__(name)

    def launch_training_runs(self) -> List[TrainingJob]:
        """Launch a single arena training run."""
        # Generate run name
        user = os.environ.get("USER", "unknown")
        date = datetime.now().strftime("%m-%d")
        run_name = f"{user}.experiments.arena.{date}"

        print(f"Launching arena training run: {run_name}")

        # Create config based on recipes/arena.sh
        config = TrainingJobConfig(
            curriculum="env/mettagrid/curriculum/arena/learning_progress",
            no_spot=True,
            wandb_tags=["arena", "experiment", self.name],
        )

        # Launch using config
        job = self.launch_training_run_from_config(run_name, config)

        if job:
            job.notes = "Arena training with learning progress curriculum"
            return [job]  # Return list of jobs

        return []  # Return empty list if launch failed

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get arena-specific analysis configuration."""
        return {
            "metrics_to_plot": [
                "overview/reward",
                "losses/policy_loss",
                "losses/value_loss",
                "losses/entropy",
                "env_agent/action.attack.agent",
                "env_agent/action.share.energy",
            ],
            "eval_suites": ["arena"],
            "description": "Arena training with learning progress curriculum",
        }


def main():
    """Run arena experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run arena experiment")
    parser.add_argument("--no-notebook", action="store_true", help="Skip notebook generation")
    parser.add_argument("--name", help="Custom experiment name")

    args = parser.parse_args()

    # Create and run experiment
    experiment = ArenaExperiment(name=args.name) if args.name else ArenaExperiment()

    # Run experiment
    results = experiment.run(generate_notebook=not args.no_notebook)

    # Save metadata
    if results["launched_jobs"]:
        experiment.save_metadata()

    print("\nExperiment complete!")
    if results["notebook_path"]:
        print(f"Analysis notebook: {results['notebook_path']}")

    return 0 if results["launched_jobs"] else 1


if __name__ == "__main__":
    exit(main())
