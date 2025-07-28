#!/usr/bin/env python3
"""Test script to verify movement metrics appear in the info dict for wandb."""

import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid import MettaGridEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.util.hydra import get_cfg


def test_wandb_movement_metrics():
    """Test that movement metrics appear in info dict when episode ends."""

    # Get the benchmark config and modify it
    cfg = get_cfg("benchmark")

    # Remove map_builder since we'll provide a level directly
    del cfg.game.map_builder

    # Simplify config for testing
    cfg.game.num_agents = 1
    cfg.game.max_steps = 20  # Short episode
    cfg.game.episode_truncates = True
    cfg.game.track_movement_metrics = True  # Enable movement metrics

    cfg.game.map_builder = OmegaConf.create(
        {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 5,
            "height": 5,
            "objects": {},
            "agents": 1,
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

    print("Testing movement metrics in info dict...")
    print(f"Action indices - Rotate: {rotate_idx}, Move: {move_idx}, Noop: {noop_idx}")

    # Execute some movements
    episode_ended = False
    info_dict = None

    for step in range(25):  # Run past max_steps to force truncation
        if step < 10:
            # Do some movements and rotations
            if step % 3 == 0:
                action = move_idx
            elif step % 3 == 1:
                action = rotate_idx
            else:
                action = noop_idx
            arg = step % 2
        else:
            action = noop_idx
            arg = 0

        actions = np.array([[action, arg]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        if terminals.any() or truncations.any():
            episode_ended = True
            info_dict = info
            print(f"Episode ended at step {step + 1}")
            break

    if not episode_ended:
        print("ERROR: Episode did not end!")
        return

    # Check info dict structure
    print("\nInfo dict keys:", list(info_dict.keys()))

    # Look for agent stats
    if "agent" in info_dict:
        agent_stats = info_dict["agent"]
        print("\nAgent stats keys:", list(agent_stats.keys())[:20], "...")  # First 20 keys

        # Look for movement metrics
        movement_metrics = {k: v for k, v in agent_stats.items() if "movement" in k}
        print("\nMovement metrics found in info['agent']:")
        for key, value in sorted(movement_metrics.items()):
            print(f"  {key}: {value}")

        # Also check for them with different patterns
        print("\nAll keys containing 'direction':")
        direction_keys = [k for k in agent_stats.keys() if "direction" in k]
        for key in direction_keys:
            print(f"  {key}: {agent_stats[key]}")

        print("\nAll keys containing 'sequential':")
        sequential_keys = [k for k in agent_stats.keys() if "sequential" in k]
        for key in sequential_keys:
            print(f"  {key}: {agent_stats[key]}")
    else:
        print("\nERROR: No 'agent' key in info dict!")

    # Also check if they appear at top level or under other keys
    print("\nChecking other locations in info dict...")
    for key, value in info_dict.items():
        if isinstance(value, dict):
            movement_in_subdict = {k: v for k, v in value.items() if "movement" in str(k)}
            if movement_in_subdict:
                print(f"\nMovement metrics found in info['{key}']:")
                for k, v in movement_in_subdict.items():
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    test_wandb_movement_metrics()
