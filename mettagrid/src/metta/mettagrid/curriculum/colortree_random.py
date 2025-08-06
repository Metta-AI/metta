"""ColorTree curriculum with random target sequences per episode."""

import random

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.core import SingleTrialTask as Task


class ColorTreeRandomFromSetCurriculum(Curriculum):
    """Curriculum that randomly selects from a predefined set of sequences."""

    def __init__(self, env_cfg: DictConfig, sequence_pool: list | None = None, **kwargs):
        super().__init__()
        self.base_env_cfg = env_cfg

        # If caller did not specify a pool, fall back to a default diverse set
        self.sequence_pool = sequence_pool or [
            [0, 1, 2, 3],  # Sequential
            [3, 2, 1, 0],  # Reverse
            [0, 0, 1, 1],  # Pairs
            [0, 1, 0, 1],  # Alternating
            [1, 2, 3, 0],  # Shifted
            [2, 0, 3, 1],  # Mixed
            [1, 1, 1, 1],  # All same
            [0, 2, 1, 3],  # Swapped pairs
        ]

    def get_task(self) -> Task:
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
