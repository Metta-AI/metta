#!/usr/bin/env python3
"""Arena experiment implementation.

This demonstrates a simple experiment that launches arena training runs.
Based on the arena.sh recipe.
"""

from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.training_run_config import TrainingRunConfig
from experiments.runner import runner


class ArenaExperimentConfig(SingleJobExperimentConfig):
    """Configuration specific to Arena experiments."""

    # Experiment metadata
    name: str = "arena_experiment"

    def __init__(self, **kwargs):
        # Always use arena curriculum for this experiment
        if "training" not in kwargs:
            kwargs["training"] = TrainingRunConfig(
                curriculum="env/mettagrid/curriculum/arena/learning_progress"
            )

        super().__init__(**kwargs)


def main():
    """Run arena experiment from command line."""
    return runner(
        SingleJobExperiment,
        ArenaExperimentConfig,
    )


if __name__ == "__main__":
    exit(main())
