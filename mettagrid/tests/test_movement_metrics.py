#!/usr/bin/env python3
"""Test script to verify movement metrics are being tracked correctly."""

import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.room.random import Random
from metta.mettagrid.util.hydra import get_cfg


def test_navigation_metrics():
    """Test that navigation metrics are correctly tracked."""

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")

    # Remove map_builder since we'll provide a level directly
    del cfg.game.map_builder

    # Simplify config for testing
    cfg.game.num_agents = 1
    cfg.game.max_steps = 100
    cfg.game.episode_truncates = True
    cfg.game.track_movement_metrics = True  # Enable movement metrics

    # Create a simple level with one agent
    level_builder = Random(width=5, height=5, objects=OmegaConf.create({}), agents=1, border_width=1)
    level = level_builder.build()

    # Create curriculum and environment
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None, level=level)

    obs, _ = env.reset()

    # Get action indices
    action_names = env.action_names
    rotate_idx = action_names.index("rotate") if "rotate" in action_names else None
    move_idx = action_names.index("move") if "move" in action_names else None

    if rotate_idx is None:
        print("ERROR: Rotate action not available")
        return

    print("Testing navigation metrics...")
    print("Action indices - Rotate:", rotate_idx, "Move:", move_idx)

    # Test sequence to verify movement metrics:
    # This creates a specific movement pattern to test both metrics:
    # 1. Direction distribution: time spent facing each direction
    # 2. Sequential rotations: sequences of consecutive rotation actions

    # Expected behavior:
    # - Agent starts facing Up (default)
    # - Rotates to Right, Down, Up, Down (4 rotations total)
    # - First 3 rotations are sequential (should count as 3-rotation sequence)
    # - 4th rotation is also sequential (should count as 1-rotation sequence)
    # - Then moves forward (breaks rotation sequence)

    actions_sequence = [
        [rotate_idx, 3],  # Rotate to Right (from Up) - start of sequence
        [rotate_idx, 1],  # Rotate to Down (from Right) - continue sequence
        [rotate_idx, 0],  # Rotate to Up (from Down) - continue sequence
        [rotate_idx, 1],  # Rotate to Down (from Up) - new sequence
        [move_idx, 0],  # Move forward (breaks sequence, stays Down)
    ]

    for i, (action, arg) in enumerate(actions_sequence):
        actions = np.array([[action, arg]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        print(f"Step {i + 1}: action={action}, arg={arg}")
        print(f"  Action success: {env.action_success}")
        print(f"  Rewards: {rewards}")
        print(f"  Terminals: {terminals}")
        print(f"  Truncations: {truncations}")
        print(f"  Info keys: {list(info.keys()) if info else 'None'}")

        # Check if episode ended
        if terminals.any() or truncations.any():
            # Get episode stats
            stats = info.get("agent", {})
            print(f"\nEpisode ended at step {i + 1}")
            print("\n" + "=" * 60)
            print("MOVEMENT METRICS RESULTS")
            print("=" * 60)

            # Direction distribution
            print("\nDirection facing counts:")
            print("(How many steps the agent spent facing each direction)")
            total_steps = 0
            for direction in ["up", "down", "left", "right"]:
                key = f"movement/facing/{direction}"
                value = stats.get(key, 0)
                total_steps += value
                print(f"  {key}: {value}")
            print(f"  Total direction steps: {total_steps}")

            # Sequential rotation behavior
            print("\nSequential rotation behavior:")
            print("(Sum of all sequential rotation sequence lengths)")
            key = "movement/sequential_rotations"
            value = stats.get(key, 0)
            print(f"  {key}: {value}")

            # Expected: 3 (first sequence) + 1 (second sequence) = 4 total
            print("  Expected: 4 (3-rotation sequence + 1-rotation sequence)")

            # Show existing action metrics for comparison
            print("\nExisting action metrics (for comparison):")
            print("(These show detailed action usage and are already tracked)")
            for action_type in ["rotate", "move"]:
                for result in ["success", "failed"]:
                    key = f"action.{action_type}.{result}"
                    value = stats.get(key, 0)
                    if value > 0:
                        print(f"  {key}: {value}")

            return

    # Force episode end to get stats
    print("\nForcing episode end...")
    for step in range(env.max_steps):
        actions = np.array([[move_idx or 0, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        print(f"Force step {step + 1}: terminals={terminals.any()}, truncations={truncations.any()}")
        if terminals.any() or truncations.any():
            break

    # Print final stats
    stats = info.get("agent", {})
    print("\n" + "=" * 60)
    print("FINAL MOVEMENT METRICS RESULTS")
    print("=" * 60)
    print(f"All agent stats keys: {list(stats.keys()) if stats else 'None'}")

    print("\nDirection facing counts:")
    print("(How many steps the agent spent facing each direction)")
    total_steps = 0
    for direction in ["up", "down", "left", "right"]:
        key = f"movement/facing/{direction}"
        value = stats.get(key, 0)
        total_steps += value
        print(f"  {key}: {value}")
    print(f"  Total direction steps: {total_steps}")

    print("\nSequential rotation behavior:")
    print("(Sum of all sequential rotation sequence lengths)")
    key = "movement/sequential_rotations"
    value = stats.get(key, 0)
    print(f"  {key}: {value}")

    print("\nExisting action metrics (for comparison):")
    print("(These show detailed action usage and are already tracked)")
    for action_type in ["rotate", "move"]:
        for result in ["success", "failed"]:
            key = f"action.{action_type}.{result}"
            value = stats.get(key, 0)
            if value > 0:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    test_navigation_metrics()
