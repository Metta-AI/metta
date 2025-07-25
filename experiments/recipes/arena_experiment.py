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
from typing import Optional, List


class ArenaExperimentConfig(BaseExperimentConfig):
    """Configuration specific to Arena experiments."""
    # Launch configuration
    curriculum: str = Field("env/mettagrid/curriculum/arena/learning_progress", description="Path to curriculum config")
    gpus: int = Field(1, description="Number of GPUs")
    nodes: int = Field(1, description="Number of nodes")
    spot: bool = Field(False, description="Use spot instances (default: no-spot)")
    skip_git_check: bool = Field(False, description="Skip git check for uncommitted changes")
    wandb_tags: Optional[List[str]] = Field(None, description="WandB tags (space-separated)")
    additional_args: Optional[List[str]] = Field(None, description="Additional trainer args (space-separated)")


class ArenaExperiment(Experiment):
    """Basic arena training experiment.

    Launches arena training with standard configuration.
    """

    def __init__(self, name: str = "arena_experiment", config: Optional[ArenaExperimentConfig] = None):
        super().__init__(name)
        self.config = config or ArenaExperimentConfig()

    def launch_training_runs(self) -> List[TrainingJob]:
        """Launch a single arena training run."""
        # Generate run name
        user = os.environ.get("USER", "unknown")
        date = datetime.now().strftime("%m-%d")
        run_name = f"{user}.experiments.arena.{date}"

        print(f"Launching arena training run: {run_name}")

        # Create config from ArenaExperimentConfig
        tags = self.config.wandb_tags or []
        tags.extend(["arena", "experiment", self.name])
        
        config = TrainingJobConfig(
            curriculum=self.config.curriculum,
            gpus=self.config.gpus,
            nodes=self.config.nodes,
            no_spot=not self.config.spot,  # Invert because CLI flag is --spot
            skip_git_check=self.config.skip_git_check,
            wandb_tags=tags,
            additional_args=self.config.additional_args or [],
        )

        # Launch using config
        job = self.launch_training_run_from_config(run_name, config)

        if job:
            job.notes = "Arena training with learning progress curriculum"
            return [job]  # Return list of jobs

        return []  # Return empty list if launch failed



def main():
    """Run arena experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run arena experiment")
    parser.add_argument("name", nargs="?", default="arena", help="Name for the experiment")
    parser.add_argument("--no-launch", action="store_true", help="Skip launching new runs")
    parser.add_argument("--job-ids", nargs="+", help="Load existing SkyPilot job IDs")
    parser.add_argument("--open", action="store_true", help="Open notebook in Jupyter")
    parser.add_argument("--sections", help="Comma-separated list of notebook sections")
    
    # Launch configuration
    parser.add_argument("-c", "--curriculum", help="Path to curriculum config")
    parser.add_argument("-g", "--gpus", type=int, help="Number of GPUs")
    parser.add_argument("-n", "--nodes", type=int, help="Number of nodes")
    parser.add_argument("--spot", action="store_true", help="Use spot instances (default: no-spot)")
    parser.add_argument("--skip-git-check", action="store_true", help="Skip git check for uncommitted changes")
    parser.add_argument("--wandb-tags", nargs="+", help="WandB tags (space-separated)")
    parser.add_argument("--additional-args", nargs="+", help="Additional trainer args (space-separated)")

    args = parser.parse_args()

    # Create config with all parameters
    config_kwargs = {
        "name": args.name,
        "launch": not args.no_launch,
        "job_ids": args.job_ids,
        "open_notebook": args.open,
        "sections": args.sections.split(",") if args.sections else None,
    }
    
    # Add launch configuration if provided
    if args.curriculum is not None:
        config_kwargs["curriculum"] = args.curriculum
    if args.gpus is not None:
        config_kwargs["gpus"] = args.gpus
    if args.nodes is not None:
        config_kwargs["nodes"] = args.nodes
    if args.spot:
        config_kwargs["spot"] = args.spot
    if args.skip_git_check:
        config_kwargs["skip_git_check"] = args.skip_git_check
    if args.wandb_tags:
        config_kwargs["wandb_tags"] = args.wandb_tags
    if args.additional_args:
        config_kwargs["additional_args"] = args.additional_args
    
    config = ArenaExperimentConfig(**config_kwargs)

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
