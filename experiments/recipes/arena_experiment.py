#!/usr/bin/env python3
"""Arena experiment implementation.

This demonstrates a simple experiment that launches arena training runs.
Based on the arena.sh recipe.
"""

from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.runner import runner


class ArenaExperimentConfig(SingleJobExperimentConfig):
    """Configuration specific to Arena experiments."""

    # Experiment metadata
    name: str = "arena_experiment"

    # Note: We don't redefine defaults here - they're inherited from the composed configs
    # Users can override via CLI: --curriculum=... --wandb-tags=...


def main():
    """Run arena experiment from command line."""
    return runner(
        SingleJobExperiment,
        ArenaExperimentConfig,
    )


if __name__ == "__main__":
    exit(main())
