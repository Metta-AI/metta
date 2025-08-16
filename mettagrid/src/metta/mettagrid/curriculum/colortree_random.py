import os
import random
import threading
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
                    If different from base config, will auto-generate color mappings

    Example YAML:
        _target_: metta.mettagrid.curriculum.colortree_random.ColorTreeRandomFromSetCurriculum
        tasks:
          /env/mettagrid/colortree_easy: 1.0
        num_colors: 3       # Will auto-generate mappings for 3 colors
        sequence_length: 2  # Generate all 2-length sequences
    """

    # Standard inventory items for auto-generation (from mettagrid.yaml)
    # Using ores and batteries as they're conceptually similar colored items
    ITEM_COLORS = [
        "ore_red",
        "ore_green",
        "ore_blue",
        "battery_red",
        "battery_green",
        "battery_blue",
        "armor",
        "laser",
    ]

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
        # Thread-safe counter to ensure diversity across parallel calls
        self._call_counter = 0
        self._lock = threading.Lock()
        # Track sequences per epoch for diversity monitoring
        self._current_epoch_sequences: list = []
        self._epoch_number = 0

        # Auto-detect parameters from base config
        # NOTE: This creates a sample task just to inspect config, which could have side effects
        sample_task = super().get_task()
        env_cfg = sample_task.env_cfg()
        color_tree_cfg = env_cfg.game.actions.color_tree

        # Auto-detect or validate num_colors
        base_num_colors = len(color_tree_cfg.color_to_item)
        if num_colors is None:
            num_colors = base_num_colors
            print(f"[ColorTreeRandom] Auto-detected num_colors={num_colors} from color_to_item")
        elif num_colors != base_num_colors:
            # Generate automatic color mapping if num_colors differs from base
            print(f"[ColorTreeRandom] Generating color_to_item mapping for {num_colors} colors")
            self._auto_color_mapping = True
            self._color_items = self._generate_color_mapping(num_colors)
        else:
            self._auto_color_mapping = False

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
        if sequence_length > 8:
            # Warning: C++ implementation uses uint8_t for correctness mask
            print(
                f"[ColorTreeRandom] WARNING: sequence_length={sequence_length} > 8. "
                f"C++ correctness tracking may overflow!"
            )

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

    def _generate_color_mapping(self, num_colors: int) -> dict:
        """Generate color_to_item mapping for the specified number of colors."""
        if num_colors > len(self.ITEM_COLORS):
            raise ValueError(f"num_colors={num_colors} exceeds available inventory items ({len(self.ITEM_COLORS)})")

        mapping = {}
        for i in range(num_colors):
            mapping[i] = self.ITEM_COLORS[i]
        return mapping

    def get_task(self) -> Task:
        """Get a task with a randomly selected sequence from the generated pool."""
        # Get base task from parent (this handles the config resolution properly)
        task = super().get_task()
        env_cfg = task.env_cfg()

        # Thread-safe counter increment and sequence selection
        # Using a lock here is necessary but kept minimal for performance
        with self._lock:
            self._call_counter += 1
            call_id = self._call_counter

            # Use proper random selection without modulo bias
            # randrange is better than randint + modulo for avoiding bias
            sequence_index = self._rng.randrange(len(self.sequence_pool))
            selected_sequence = self.sequence_pool[sequence_index]
            self._last_selected_sequence = selected_sequence

            # Track episode count
            self._episode_count += 1

            # Track sequences for current epoch
            self._current_epoch_sequences.append(tuple(selected_sequence))

            # Print diversity report only if verbose logging is enabled
            # Set environment variable COLORTREE_VERBOSE=1 to see reports
            if os.environ.get("COLORTREE_VERBOSE", "0") == "1":
                if call_id == 4:  # After 4 calls (typical initialization)
                    print("\n[ColorTreeRandom] Initial diversity report after 4 calls:", flush=True)
                    self._print_epoch_diversity_report(1, min(4, len(self._current_epoch_sequences)))
                elif call_id == 8:  # If we get 8 calls
                    self._print_epoch_diversity_report(2, min(8, len(self._current_epoch_sequences)))
                elif call_id > 8 and call_id % 32 == 0:  # Then every 32 calls
                    self._epoch_number += 1
                    self._print_epoch_diversity_report(self._epoch_number + 2, 32)

        # Update the ColorTree action config with the selected sequence
        if (
            hasattr(env_cfg, "game")
            and hasattr(env_cfg.game, "actions")
            and hasattr(env_cfg.game.actions, "color_tree")
        ):
            color_tree_cfg = env_cfg.game.actions.color_tree

            # Validate sequence length matches what we expect
            if len(selected_sequence) != self._sequence_length:
                raise ValueError(
                    f"Selected sequence length {len(selected_sequence)} doesn't match expected {self._sequence_length}"
                )

            color_tree_cfg.target_sequence = selected_sequence
            # Clear trial_sequences to ensure C++ uses target_sequence
            color_tree_cfg.trial_sequences = []
            # num_trials stays from base config

            # Apply auto-generated color mapping if needed
            if hasattr(self, "_auto_color_mapping") and self._auto_color_mapping:
                color_tree_cfg.color_to_item = self._color_items

            # Compute attempts_per_trial dynamically from max_steps and sequence_length
            # We cap at least 1 attempt to ensure a window can complete
            try:
                current_max_steps = int(getattr(env_cfg.game, "max_steps", 0))
            except Exception:
                current_max_steps = 0

            if current_max_steps > 0 and self._sequence_length > 0:
                computed_attempts = max(1, current_max_steps // self._sequence_length)
                color_tree_cfg.attempts_per_trial = int(computed_attempts)
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

    def _print_epoch_diversity_report(self, epoch_num: int, interval: int) -> None:
        """Print a diversity report for the sequences in the recent epoch."""
        if not self._current_epoch_sequences:
            return

        # Get the last 'interval' sequences (most recent epoch)
        sequences = self._current_epoch_sequences[-interval:]
        unique_sequences = set(sequences)

        # Count occurrences
        from collections import Counter

        seq_counts = Counter(sequences)

        # Format sequences for printing
        def format_seq(seq):
            return "[" + ",".join(str(x) for x in seq) + "]"

        print(f"\n{'=' * 60}", flush=True)
        print(
            f"[ColorTreeRandom] Epoch {epoch_num} Diversity Report (last {len(sequences)} calls)",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)
        print(f"Total environments: {len(sequences)}")
        print(f"Unique sequences: {len(unique_sequences)}")
        print(f"Diversity ratio: {len(unique_sequences) / len(sequences) * 100:.1f}%")
        print(f"Sequence pool size: {len(self.sequence_pool)} possible sequences")

        # Show distribution
        if len(unique_sequences) <= 10:
            print("\nAll unique sequences used:")
            for seq in sorted(unique_sequences):
                count = seq_counts[seq]
                print(f"  {format_seq(seq)}: {count}x")
        else:
            print("\nTop 5 most common sequences:")
            for seq, count in seq_counts.most_common(5):
                print(f"  {format_seq(seq)}: {count}x")
            print(f"  ... and {len(unique_sequences) - 5} more unique sequences")

        # Diversity assessment
        if len(unique_sequences) == len(sequences):
            print("\n✅ PERFECT: Every environment has a unique sequence!")
        elif len(unique_sequences) >= len(sequences) * 0.8:
            print("\n✅ EXCELLENT: High diversity (>80% unique)")
        elif len(unique_sequences) >= len(sequences) * 0.5:
            print("\n🔶 GOOD: Moderate diversity (>50% unique)")
        else:
            print(f"\n⚠️  LOW DIVERSITY: Only {len(unique_sequences)}/{len(sequences)} unique")
            max_repeat = max(seq_counts.values())
            if max_repeat == 64:
                print("    Pattern detected: 64x repetition (possible 4 workers × 64 envs)")
            elif max_repeat == 32:
                print("    Pattern detected: 32x repetition (possible 8 workers × 32 envs)")

        print(f"{'=' * 60}\n", flush=True)
