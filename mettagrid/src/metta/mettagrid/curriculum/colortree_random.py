"""ColorTree curriculum with random target sequences per episode."""

import random
from itertools import permutations
from typing import Dict

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTrialTask, Task
from metta.mettagrid.curriculum.random import RandomCurriculum


def generate_sequence_pool(num_colors: int, sequence_length: int = 4) -> list[list[int]]:
    """Generate a diverse pool of sequences for the given number of colors."""
    pool = []
    colors = list(range(num_colors))

    # 1. All same patterns (simple baseline)
    for color in colors:
        pool.append([color] * sequence_length)

    # 2. Alternating patterns (for 2+ colors)
    if num_colors >= 2:
        for i in range(num_colors):
            for j in range(i + 1, num_colors):
                # Simple alternating
                pattern1 = [i, j] * (sequence_length // 2)
                pattern2 = [j, i] * (sequence_length // 2)
                if len(pattern1) == sequence_length:
                    pool.extend([pattern1, pattern2])

    # 3. Sequential patterns (permutations of available colors)
    if num_colors <= sequence_length:
        # Full permutations when we have enough slots
        for perm in list(permutations(colors))[: min(6, len(list(permutations(colors))))]:
            padded = list(perm) + [perm[0]] * (sequence_length - len(perm))
            pool.append(padded[:sequence_length])
    else:
        # Sample from colors when we have more colors than slots
        for _ in range(min(8, num_colors)):
            sampled = random.sample(colors, sequence_length)
            pool.append(sampled)

    # 4. Repeating element patterns
    if num_colors >= 2:
        for double_pos in range(sequence_length - 1):
            remaining_colors = colors.copy()
            base_color = remaining_colors.pop(0)
            pattern = [
                base_color
                if i == double_pos or i == double_pos + 1
                else remaining_colors[(i - (2 if i > double_pos + 1 else 0)) % len(remaining_colors)]
                for i in range(sequence_length)
            ]
            pool.append(pattern)

    # 5. Mirror/palindrome patterns
    if sequence_length == 4 and num_colors >= 2:
        for i in range(num_colors):
            for j in range(i + 1, min(num_colors, i + 3)):  # Limit combinations
                pool.append([i, j, j, i])  # Mirror pattern

    # 6. Mixed complex patterns (random sampling with constraints)
    for _ in range(min(10, num_colors * 2)):
        # Ensure each pattern uses at least 2 different colors
        pattern = []
        used_colors = set()
        for pos in range(sequence_length):
            if len(used_colors) < 2 and pos == sequence_length - 1:
                # Force diversity in last position if needed
                available = [c for c in colors if c not in used_colors]
                if available:
                    color = random.choice(available)
                else:
                    color = random.choice(colors)
            else:
                color = random.choice(colors)
            pattern.append(color)
            used_colors.add(color)
        pool.append(pattern)

    # Remove duplicates while preserving order
    seen = set()
    unique_pool = []
    for seq in pool:
        seq_tuple = tuple(seq)
        if seq_tuple not in seen:
            seen.add(seq_tuple)
            unique_pool.append(seq)

    return unique_pool


class ColorTreeRandomFromSetCurriculum(RandomCurriculum):
    """Curriculum that randomly selects from a programmatically generated set of sequences."""

    def __init__(
        self,
        tasks: Dict[str, float] | DictConfig,
        env_overrides: DictConfig | None = None,
        num_colors: int | None = None,
        sequence_length: int = 4,
        **kwargs,
    ):
        super().__init__(tasks, env_overrides)

        # Auto-detect num_colors from task config if not provided
        if num_colors is None:
            # Get a sample task to inspect the color configuration
            sample_task = super().get_task()
            color_to_item = sample_task.env_cfg().game.actions.color_tree.color_to_item
            num_colors = len(color_to_item)

        # Generate sequence pool based on number of colors
        self.sequence_pool = generate_sequence_pool(num_colors, sequence_length)

        print(f"Generated {len(self.sequence_pool)} sequences for {num_colors} colors:")
        for _i, seq in enumerate(self.sequence_pool[:10]):  # Show first 10
            print(f"  {seq}")
        if len(self.sequence_pool) > 10:
            print(f"  ... and {len(self.sequence_pool) - 10} more")

    def get_task(self) -> Task:
        """Get a task with a randomly selected sequence from the generated pool."""
        # Get base task from parent (this handles the config resolution properly)
        task = super().get_task()

        # Randomly select a sequence from the pool
        selected_sequence = random.choice(self.sequence_pool)

        # Get the environment config and update it with our selected sequence
        env_cfg = task.env_cfg()

        # Update the ColorTree action config
        if (
            hasattr(env_cfg, "game")
            and hasattr(env_cfg.game, "actions")
            and hasattr(env_cfg.game.actions, "color_tree")
        ):
            env_cfg.game.actions.color_tree.target_sequence = selected_sequence
            env_cfg.game.actions.color_tree.trial_sequences = [
                selected_sequence,
                selected_sequence,
                selected_sequence,
                selected_sequence,
            ]
            env_cfg.game.actions.color_tree.num_trials = 4

        # Use the original task ID to avoid KeyError in completion tracking
        return SingleTrialTask(id=task.id(), curriculum=self, env_cfg=env_cfg)
