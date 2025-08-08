#!/usr/bin/env python3
import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.util.hydra import get_cfg


def test_no_agent_interference():
    """Test that movement metrics are correctly tracked"""

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")

    # Simplify config for testing
    cfg.game.num_agents = 2
    cfg.game.max_steps = 100
    cfg.game.episode_truncates = True
    cfg.game.track_movement_metrics = True  # Enable movement metrics
    cfg.game.no_agent_interference = True

    # Create a simple level with one agent
    cfg.game.map_builder = OmegaConf.create(
        {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 5,
            "height": 5,
            "objects": {},
            "agents": 2,
            "border_width": 1,
        }
    )
    # Create curriculum and environment
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)

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

    # Test sequence to verify agent don't interfere with each other
    # - Both agents move into the same corner and then walk around together

    # Expected behavior:
    # - Both agents walk around the map overlapping each other

    actions_sequence = [
        [move_idx, 0],  # Move forward (direction depends on initial orientation)
        [move_idx, 0],  # Move forward (direction depends on initial orientation)
        [move_idx, 0],  # Move forward (direction depends on initial orientation)
        [rotate_idx, 2],  # Rotate to Left - continue sequence,
        [move_idx, 0],  # Move forward (direction depends on initial orientation)
        [move_idx, 0],  # Move forward (direction depends on initial orientation)
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
        actions = np.array([[action, arg], [action, arg]], dtype=np.int32)
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
        actions = np.array([[noop_idx or 0, 0], [noop_idx or 0, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        if terminals.any() or truncations.any():
            break

    # Print final stats
    stats = info.get("agent", {})
    print_results(stats, early_end=False, step=len(actions_sequence))


def print_results(stats, early_end=False, step=None):
    """Print test results"""
    print(f"\n{'=' * 50}")
    print(f"TEST RESULTS {'(Early End)' if early_end else ''}")
    if step:
        print(f"Steps completed: {step}")
    print(f"{'=' * 50}")

    if not stats:
        print("No stats available")
        return

    print("Agent stats:")
    for agent_id, agent_stats in stats.items():
        print(f"  Agent {agent_id}:")
        for stat_name, stat_value in agent_stats.items():
            print(f"    {stat_name}: {stat_value}")


if __name__ == "__main__":
    test_no_agent_interference()
