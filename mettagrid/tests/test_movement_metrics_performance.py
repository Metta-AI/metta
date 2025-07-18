#!/usr/bin/env python3
"""Performance test script to compare movement metrics overhead."""

import time

import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.room.random import Random
from metta.mettagrid.util.hydra import get_cfg


def run_performance_test(enable_movement_metrics: bool, num_steps: int = 1000):
    """Run a performance test with or without movement metrics."""

    # Get config
    cfg = get_cfg("benchmark")
    del cfg.game.map_builder

    # Configure environment
    cfg.game.num_agents = 4
    cfg.game.max_steps = num_steps + 100  # Extra buffer
    cfg.game.episode_truncates = True
    cfg.game.track_movement_metrics = enable_movement_metrics

    # Create level
    level_builder = Random(width=20, height=20, objects=OmegaConf.create({}), agents=4, border_width=1)
    level = level_builder.build()

    # Create environment
    curriculum = SingleTaskCurriculum("perf_test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None, level=level)

    # Reset environment
    env.reset()

    # Get actions
    action_names = env.action_names
    rotate_idx = action_names.index("rotate") if "rotate" in action_names else 0
    move_idx = action_names.index("move") if "move" in action_names else 0

    # Prepare random actions (mix of rotate and move)
    np.random.seed(42)  # For reproducibility
    action_types = np.random.choice([rotate_idx, move_idx], size=num_steps)
    action_args = np.random.randint(0, 4, size=num_steps)  # Random args

    # Run timing test
    start_time = time.time()

    for i in range(num_steps):
        # Create action for all agents
        actions = np.array([[action_types[i], action_args[i]]] * env.num_agents, dtype=np.int32)

        # Step environment
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check if episode ended
        if terminals.any() or truncations.any():
            env.reset()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Movement metrics {'ENABLED' if enable_movement_metrics else 'DISABLED'}:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Steps per second: {num_steps / total_time:.1f}")
    print(f"  Milliseconds per step: {(total_time / num_steps) * 1000:.2f}ms")

    return total_time


def main():
    """Run the performance comparison."""
    print("Movement Metrics Performance Test")
    print("=" * 40)
    print("Testing focused movement metrics:")
    print("- Direction distribution (movement/facing/*)")
    print("- Sequential rotation behavior (movement/sequential_rotations)")
    print("- Action usage already tracked by existing action.*.success metrics")

    num_steps = 1000

    # Test without movement metrics
    print(f"\nRunning {num_steps} steps without movement metrics...")
    time_without = run_performance_test(enable_movement_metrics=False, num_steps=num_steps)

    # Test with movement metrics
    print(f"\nRunning {num_steps} steps with movement metrics...")
    time_with = run_performance_test(enable_movement_metrics=True, num_steps=num_steps)

    # Calculate overhead
    overhead_pct = ((time_with - time_without) / time_without) * 100
    overhead_ms = ((time_with - time_without) / num_steps) * 1000

    print("\nPerformance Impact:")
    print(f"  Overhead: {overhead_pct:.1f}%")
    print(f"  Additional time per step: {overhead_ms:.3f}ms")

    if overhead_pct < 5:
        print("  ✅ Low overhead - safe to enable")
    elif overhead_pct < 10:
        print("  ⚠️  Moderate overhead - consider for training")
    else:
        print("  ❌ High overhead - use sparingly")


if __name__ == "__main__":
    main()
