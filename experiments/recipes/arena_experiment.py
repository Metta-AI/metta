#!/usr/bin/env python3
"""Arena experiment implementation.

This demonstrates a simple experiment that launches arena training runs.
Based on the arena.sh recipe.
"""

from typing import List

from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.runner import runner


class ArenaExperimentConfig(SingleJobExperimentConfig):
    """Configuration specific to Arena experiments."""

    # Experiment overrides
    curriculum: str = "env/mettagrid/curriculum/arena/learning_progress"
    name: str = "arena_experiment"

    # Training job overrides
    wandb_tags: List[str] = ["arena"]


def main():
    """Run arena experiment from command line."""
    return runner(
        SingleJobExperiment,
        ArenaExperimentConfig,
    )


if __name__ == "__main__":
    exit(main())
