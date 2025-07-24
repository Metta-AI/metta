#!/usr/bin/env python3
"""Arena experiment implementation.

This demonstrates a simple experiment that launches arena training runs.
Based on the arena.sh recipe.
"""

import os
from datetime import datetime
from typing import Any, Dict

from experiments.experiment import Experiment
from experiments.launch import launch_training_run
from experiments.types import TrainingJob


class ArenaExperiment(Experiment):
    """Basic arena training experiment.

    Launches arena training with standard configuration.
    """

    def __init__(self, name: str = "arena_experiment"):
        super().__init__(name)

    def launch_training_runs(self) -> Dict[str, Any]:
        """Launch a single arena training run."""
        # Generate run name
        user = os.environ.get("USER", "unknown")
        date = datetime.now().strftime("%m-%d")
        run_name = f"{user}.experiments.arena.{date}"

        print(f"Launching arena training run: {run_name}")

        # Launch with arena configuration
        # Based on recipes/arena.sh
        result = launch_training_run(
            run_name=run_name,
            curriculum="env/mettagrid/curriculum/arena/learning_progress",
            num_gpus=4,
            num_nodes=8,
            no_spot=True,
            additional_args=[
                "trainer.optimizer.learning_rate=0.0045",
                "trainer.optimizer.type=muon",
                "trainer.simulation.evaluate_interval=50",
            ],
            wandb_tags=["arena", "experiment", self.name],
        )

        # Store result
        self.launch_results.append(result)
        
        # Create TrainingJob object if successful
        if result["success"]:
            job = TrainingJob(
                wandb_run_id=run_name,
                skypilot_job_id=result.get("job_id"),
                config={
                    "curriculum": "env/mettagrid/curriculum/arena/learning_progress",
                    "num_gpus": 4,
                    "num_nodes": 8,
                    "optimizer": "muon",
                    "learning_rate": 0.0045,
                },
                notes="Arena training with learning progress curriculum"
            )
            self.training_jobs.append(job)

        # Return summary
        return {
            "run_names": [run_name] if result["success"] else [],
            "job_ids": [result["job_id"]] if result["job_id"] else [],
            "launch_results": [result],
            "success": result["success"],
        }

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
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--no-notebook", action="store_true", help="Skip notebook generation")
    parser.add_argument("--name", help="Custom experiment name")

    args = parser.parse_args()

    # Create and run experiment
    experiment = ArenaExperiment(name=args.name) if args.name else ArenaExperiment()

    # Override launch function for dry run
    if args.dry_run:
        print("[DRY RUN MODE]")
        original_launch = launch_training_run

        def dry_run_launch(**kwargs):
            kwargs["dry_run"] = True
            return original_launch(**kwargs)

        # Monkey patch for dry run
        import experiments.launch

        experiments.launch.launch_training_run = dry_run_launch

    # Run experiment
    results = experiment.run(generate_notebook=not args.no_notebook)

    # Save metadata
    if results["launch_summary"]["success"]:
        experiment.save_metadata()

    print("\nExperiment complete!")
    if results["notebook_path"]:
        print(f"Analysis notebook: {results['notebook_path']}")

    return 0 if results["launch_summary"]["success"] else 1


if __name__ == "__main__":
    exit(main())
