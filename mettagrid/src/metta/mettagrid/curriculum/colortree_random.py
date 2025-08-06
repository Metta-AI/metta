"""ColorTree curriculum with random target sequences per episode."""

import random

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.core import SingleTrialTask as Task


class ColorTreeRandomSequenceCurriculum(Curriculum):
    """Curriculum that generates random ColorTree sequences for each episode."""

    def __init__(self, env_cfg: DictConfig, sequence_length: int = 4, num_colors: int = 4, **kwargs):
        super().__init__()
        # Store base environment config to be copied each episode
        self.base_env_cfg = env_cfg
        self.sequence_length = sequence_length
        self.num_colors = num_colors

    def get_task(self) -> Task:
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
