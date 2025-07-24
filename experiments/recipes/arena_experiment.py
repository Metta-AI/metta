#!/usr/bin/env python3
"""Arena experiment implementation.

This demonstrates a simple experiment that launches arena training runs.
Based on the arena.sh recipe.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

from experiments.experiment import Experiment
from experiments.types import TrainingJob, TrainingJobConfig, BaseExperimentConfig
from pydantic import Field


class ArenaExperimentConfig(BaseExperimentConfig):
    """Configuration specific to Arena experiments."""
    # Add any arena-specific parameters here
    # For now, just use base config
    pass


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
    parser.add_argument("name", nargs="?", default="arena", help="Name for the experiment")
    parser.add_argument("--no-launch", action="store_true", help="Skip launching new runs")
    parser.add_argument("--job-ids", nargs="+", help="Load existing SkyPilot job IDs")
    parser.add_argument("--open", action="store_true", help="Open notebook in Jupyter")
    parser.add_argument("--sections", help="Comma-separated list of notebook sections")

    args = parser.parse_args()

    # Create config
    config = ArenaExperimentConfig(
        name=args.name,
        launch=not args.no_launch,
        job_ids=args.job_ids,
        open_notebook=args.open,
        sections=args.sections.split(",") if args.sections else None,
    )

    # Create notebook
    try:
        notebook_path = ArenaExperiment.create_notebook(config)
        print(f"\nNotebook created: {notebook_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
