"""ColorTree curriculum with random target sequences per episode."""

import os
import random
from itertools import product
from typing import Dict

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTrialTask, Task
from metta.mettagrid.curriculum.random import RandomCurriculum


class ColorTreeRandomFromSetCurriculum(RandomCurriculum):
    """Curriculum that randomly selects from all possible sequences.

    The curriculum generates all possible sequences of a given length using the
    available colors, then randomly selects one per episode.

    Parameters (in YAML):
        tasks: Dictionary of task paths and weights
        sequence_length: Length of sequences to generate (default: auto-detect from base config)
        num_colors: Number of colors to use (default: auto-detect from color_to_item)

    Example YAML:
        _target_: metta.mettagrid.curriculum.colortree_random.ColorTreeRandomFromSetCurriculum
        tasks:
          /env/mettagrid/colortree_easy: 1.0
        sequence_length: 2  # Generate all 2-length sequences
        num_colors: 3       # Use colors 0, 1, 2
    """

    def __init__(
        self,
        tasks: Dict[str, float] | DictConfig,
        env_overrides: DictConfig | None = None,
        num_colors: int | None = None,
        sequence_length: int | None = None,
        **kwargs,
    ):
        super().__init__(tasks, env_overrides)
        self._last_selected_sequence: list[int] | None = None
        self._episode_count = 0  # Initialize here instead of in get_task
        # Create a unique RNG with a random seed for this curriculum instance
        self._rng = random.Random(int.from_bytes(os.urandom(8), "big"))
        # Counter to ensure diversity across parallel calls
        self._call_counter = 0

        # Auto-detect parameters from base config
        # NOTE: This creates a sample task just to inspect config, which could have side effects
        sample_task = super().get_task()
        env_cfg = sample_task.env_cfg()
        color_tree_cfg = env_cfg.game.actions.color_tree

        # Auto-detect num_colors
        if num_colors is None:
            num_colors = len(color_tree_cfg.color_to_item)
            print(f"[ColorTreeRandom] Auto-detected num_colors={num_colors} from color_to_item")

        # Auto-detect sequence_length
        if sequence_length is None:
            base_sequence = color_tree_cfg.target_sequence
            sequence_length = len(base_sequence) if base_sequence else 4
            print(
                f"[ColorTreeRandom] Auto-detected sequence_length={sequence_length} "
                f"from target_sequence={base_sequence}"
            )

        # Store config for validation
        self._sequence_length = sequence_length
        self._num_colors = num_colors

        # Validate parameters
        if num_colors <= 0:
            raise ValueError(f"num_colors must be positive, got {num_colors}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")

        # Generate all possible sequences to ensure complete coverage
        self.sequence_pool = [list(seq) for seq in product(range(num_colors), repeat=sequence_length)]

        if not self.sequence_pool:
            raise ValueError(f"Failed to generate sequence pool for colors={num_colors}, length={sequence_length}")

        # Critical validation: ensure all sequences have the expected length
        for i, seq in enumerate(self.sequence_pool):
            if len(seq) != sequence_length:
                raise ValueError(f"Sequence {i}: {seq} has wrong length {len(seq)}, expected {sequence_length}")

        print(
            f"[ColorTreeRandom] Initialized with {len(self.sequence_pool)} sequences "
            f"(colors={num_colors}, length={sequence_length})"
        )

        # Note: The relationship between sequence_length and max_steps determines
        # how many complete sequences can fit in an episode:
        # - max_steps=16, sequence_length=4 → 4 complete sequences max
        # - max_steps=16, sequence_length=2 → 8 complete sequences max
        # - max_steps=16, sequence_length=1 → 16 complete sequences max (every action is a sequence)

    def get_task(self) -> Task:
        """Get a task with a randomly selected sequence from the generated pool."""
        # Get base task from parent (this handles the config resolution properly)
        task = super().get_task()
        env_cfg = task.env_cfg()

        # Use call counter to seed selection for diversity
        # This ensures different parallel environments get different sequences
        self._call_counter += 1

        # Create a temporary RNG seeded with both the base RNG and call counter
        # This ensures reproducibility while maintaining diversity
        temp_seed = self._rng.randint(0, 2**31) + self._call_counter
        temp_rng = random.Random(temp_seed)

        # Select a random sequence from the pool
        selected_sequence = temp_rng.choice(self.sequence_pool)
        self._last_selected_sequence = selected_sequence

        # Debug logging for first few episodes
        self._episode_count += 1

        # Update the ColorTree action config with the selected sequence
        if (
            hasattr(env_cfg, "game")
            and hasattr(env_cfg.game, "actions")
            and hasattr(env_cfg.game.actions, "color_tree")
        ):
            color_tree_cfg = env_cfg.game.actions.color_tree

            # Log what we're about to set (first 3 episodes only)
            if self._episode_count <= 3:
                print(f"[ColorTreeRandom] Episode {self._episode_count}:")
                print(f"  - Selected sequence: {selected_sequence} (len={len(selected_sequence)})")
                print(f"  - Before: target_sequence={color_tree_cfg.target_sequence}")

            # Validate sequence length matches what we expect
            if len(selected_sequence) != self._sequence_length:
                raise ValueError(
                    f"Selected sequence length {len(selected_sequence)} doesn't match expected {self._sequence_length}"
                )

            color_tree_cfg.target_sequence = selected_sequence
            # Clear trial_sequences to ensure C++ uses target_sequence
            color_tree_cfg.trial_sequences = []
            # num_trials stays from base config

            if self._episode_count <= 3:
                print(f"  - After: target_sequence={color_tree_cfg.target_sequence}")
                print(f"  - Max steps: {env_cfg.game.max_steps}")
        else:
            raise ValueError("ColorTree action not found in environment config")

        # Use the original task ID to avoid KeyError in completion tracking
        return SingleTrialTask(id=task.id(), curriculum=self, env_cfg=env_cfg)

    def get_curriculum_stats(self) -> dict:
        """Expose the most recently selected sequence and pool size for logging/diagnostics."""
        selected = (
            ",".join(str(x) for x in self._last_selected_sequence)
            if self._last_selected_sequence is not None
            else "none"
        )
        return {
            "selected_sequence": selected,
            "sequence_pool_size": len(self.sequence_pool),
            "episode_count": self._episode_count,
            "call_count": self._call_counter,  # Track how many times get_task has been called
        }
