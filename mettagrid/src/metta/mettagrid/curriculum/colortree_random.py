"""ColorTree curriculum with random target sequences per episode."""

import random

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.core import SingleTrialTask as Task


class ColorTreeRandomSequenceCurriculum(Curriculum):
    """Curriculum that generates random ColorTree sequences for each episode."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.sequence_length = cfg.get("sequence_length", 4)
        self.num_colors = cfg.get("num_colors", 4)
        self.base_env_cfg = cfg.env_cfg

    def next_task(self) -> Task:
        """Generate a new task with a random target sequence."""
        # Create a copy of the base environment config
        env_cfg = DictConfig(self.base_env_cfg)

        # Generate a random target sequence
        random_sequence = [random.randint(0, self.num_colors - 1) for _ in range(self.sequence_length)]

        # Update the ColorTree action config with the random sequence
        if "game" in env_cfg and "actions" in env_cfg.game and "color_tree" in env_cfg.game.actions:
            env_cfg.game.actions.color_tree.target_sequence = random_sequence
            # Clear trial_sequences since we want the same sequence for the whole episode
            env_cfg.game.actions.color_tree.trial_sequences = []
            env_cfg.game.actions.color_tree.num_trials = 1

        # Create a unique task ID based on the sequence
        task_id = f"colortree_seq_{''.join(map(str, random_sequence))}"

        return Task(id=task_id, curriculum=self, env_cfg=env_cfg)


class ColorTreeRandomFromSetCurriculum(Curriculum):
    """Curriculum that randomly selects from a predefined set of sequences."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.base_env_cfg = cfg.env_cfg

        # Define a set of interesting sequences to randomly choose from
        self.sequence_pool = cfg.get(
            "sequence_pool",
            [
                [0, 1, 2, 3],  # Sequential
                [3, 2, 1, 0],  # Reverse
                [0, 0, 1, 1],  # Pairs
                [0, 1, 0, 1],  # Alternating
                [1, 2, 3, 0],  # Shifted
                [2, 0, 3, 1],  # Mixed
                [1, 1, 1, 1],  # All same
                [0, 2, 1, 3],  # Swapped pairs
            ],
        )

    def next_task(self) -> Task:
        """Select a random sequence from the pool."""
        env_cfg = DictConfig(self.base_env_cfg)

        # Randomly select a sequence from the pool
        selected_sequence = random.choice(self.sequence_pool)

        # Update the ColorTree action config
        if "game" in env_cfg and "actions" in env_cfg.game and "color_tree" in env_cfg.game.actions:
            env_cfg.game.actions.color_tree.target_sequence = selected_sequence
            env_cfg.game.actions.color_tree.trial_sequences = []
            env_cfg.game.actions.color_tree.num_trials = 1

        task_id = f"colortree_pool_{''.join(map(str, selected_sequence))}"

        return Task(id=task_id, curriculum=self, env_cfg=env_cfg)
