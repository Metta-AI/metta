#!/usr/bin/env python3
"""Test script to verify refactored movement metrics are working correctly."""

import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.room.random import Random
from metta.mettagrid.util.hydra import get_cfg


def test_movement_metrics_refactored():
    """Test that movement metrics are correctly tracked with the refactored implementation."""

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
    noop_idx = action_names.index("noop") if "noop" in action_names else None

    if rotate_idx is None or move_idx is None:
        print(f"ERROR: Required actions not available. rotate: {rotate_idx}, move: {move_idx}")
        return

    print("Testing refactored movement metrics...")
    print(f"Action indices - Rotate: {rotate_idx}, Move: {move_idx}, Noop: {noop_idx}")

    # Test sequence to verify movement metrics:
    # 1. Movement directions: track when agent actually moves in each direction
    # 2. Sequential rotations: noop doesn't break the sequence

    # Expected behavior:
    # - Move forward (should track movement.direction based on orientation)
    # - Rotate, noop, noop, rotate, noop, rotate, move
    #   Should count as 2 sequential rotations (only rotations that follow another rotation)

    actions_sequence = [
        [move_idx, 0],  # Move forward (direction depends on initial orientation)
        [rotate_idx, 1],  # Rotate to Down - start sequence
        [noop_idx, 0],  # Noop - doesn't break sequence
        [noop_idx, 0],  # Noop - doesn't break sequence
        [rotate_idx, 0],  # Rotate to Up - continue sequence
        [noop_idx, 0],  # Noop - doesn't break sequence
        [rotate_idx, 2],  # Rotate to Left - continue sequence
        [move_idx, 0],  # Move forward (left) - breaks sequence, tracks direction
        [move_idx, 1],  # Move backward (right) - tracks direction
    ]

    for i, (action, arg) in enumerate(actions_sequence):
        actions = np.array([[action, arg]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        print(f"\nStep {i + 1}: action={action_names[action] if action < len(action_names) else 'invalid'}, arg={arg}")
        print(f"  Action success: {env.action_success}")

        # Check if episode ended
        if terminals.any() or truncations.any():
            stats = info.get("agent", {})
            print_results(stats, early_end=True, step=i + 1)
            return

    # Force episode end to get final stats
    print("\nForcing episode end to collect final stats...")
    for _step in range(env.max_steps):
        actions = np.array([[noop_idx or 0, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        if terminals.any() or truncations.any():
            break

    # Print final stats
    stats = info.get("agent", {})
    print_results(stats)


def print_results(stats, early_end=False, step=None):
    """Print the movement metrics results."""
    if early_end:
        print(f"\nEpisode ended early at step {step}")

    print("\n" + "=" * 60)
    print("MOVEMENT METRICS RESULTS (REFACTORED)")
    print("=" * 60)

    # Movement direction counts
    print("\nMovement direction counts:")
    print("(Number of successful moves in each direction)")
    total_moves = 0
    for direction in ["up", "down", "left", "right"]:
        key = f"movement.direction.{direction}"
        value = stats.get(key, 0)
        total_moves += value
        if value > 0:
            print(f"  {key}: {value}")
    print(f"  Total moves: {total_moves}")

    # Sequential rotation behavior
    print("\nSequential rotation behavior:")
    print("(Sum of all sequential rotation sequence lengths)")
    key = "movement.sequential_rotations"
    value = stats.get(key, 0)
    print(f"  {key}: {value}")
    print("  Expected: 2 (rotations that follow another rotation, ignoring the first)")

    # Show existing action metrics for comparison
    print("\nExisting action metrics (for comparison):")
    for action_type in ["rotate", "move", "noop"]:
        for result in ["success", "failed"]:
            key = f"action.{action_type}.{result}"
            value = stats.get(key, 0)
            if value > 0:
                print(f"  {key}: {value}")


def test_movement_stats_reset_between_episodes():
    """Test that movement stats are properly reset between episodes."""

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")
    del cfg.game.map_builder
    cfg.game.num_agents = 1
    cfg.game.max_steps = 20  # Longer episodes to ensure moves
    cfg.game.episode_truncates = True
    cfg.game.track_movement_metrics = True

    # Create a larger level to ensure movement space
    level_builder = Random(width=10, height=10, objects=OmegaConf.create({}), agents=1, border_width=1)
    level = level_builder.build()

    # Create curriculum and environment
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None, level=level)

    action_names = env.action_names
    move_idx = action_names.index("move")
    rotate_idx = action_names.index("rotate")

    print("\nTesting stats reset between episodes...")

    # Episode 1: Force successful movements by trying different directions
    env.reset()
    successful_moves = 0
    for direction in range(4):  # Try all 4 directions
        # Rotate to this direction
        actions = np.array([[rotate_idx, direction]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        if terminals.any() or truncations.any():
            break

        # Try to move forward
        actions = np.array([[move_idx, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        if env.action_success[0]:  # If move succeeded
            successful_moves += 1
        if terminals.any() or truncations.any():
            break

    # Force episode end to get stats
    while not (terminals.any() or truncations.any()):
        actions = np.array([[0, 0]], dtype=np.int32)  # noop
        obs, rewards, terminals, truncations, info = env.step(actions)

    # Get stats from episode 1
    stats1 = info.get("agent", {})
    move_count1 = sum(stats1.get(f"movement.direction.{d}", 0) for d in ["up", "down", "left", "right"])

    print(f"Episode 1 - Attempted moves: 4, Successful: {successful_moves}, Tracked: {move_count1}")

    # Episode 2: Only do rotations (no moves)
    env.reset()
    for _ in range(3):
        actions = np.array([[rotate_idx, 1]], dtype=np.int32)  # Just rotate
        obs, rewards, terminals, truncations, info = env.step(actions)
        if terminals.any() or truncations.any():
            break

    # Force episode end
    while not (terminals.any() or truncations.any()):
        actions = np.array([[0, 0]], dtype=np.int32)  # noop
        obs, rewards, terminals, truncations, info = env.step(actions)

    # Get stats from episode 2
    stats2 = info.get("agent", {})
    move_count2 = sum(stats2.get(f"movement.direction.{d}", 0) for d in ["up", "down", "left", "right"])

    print(f"Episode 2 - Total moves: {move_count2}")
    print(f"Stats properly reset: {move_count1 > 0 and move_count2 == 0}")

    return move_count1, move_count2


if __name__ == "__main__":
    test_movement_metrics_refactored()
    test_movement_stats_reset_between_episodes()
