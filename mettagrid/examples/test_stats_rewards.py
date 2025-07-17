#!/usr/bin/env python3
"""
Example script demonstrating stats-based rewards in MettaGrid.

This script shows how to configure and use stats rewards that give agents
rewards based on their tracked statistics like movement, attacks, and item collection.
"""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def create_stats_reward_environment():
    """Create an environment with both inventory and stats rewards."""
    game_config = {
        "max_steps": 50,
        "num_agents": 4,  # Changed to 4 to match the map
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": ["ore_red", "heart", "laser"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}},
            "get_items": {"enabled": True},
        },
        "groups": {
            "team_red": {"id": 0, "sprite": 0},
            "team_blue": {"id": 1, "sprite": 4},
        },
        "objects": {
            "wall": {"type_id": 1},
            "mine_red": {
                "type_id": 2,
                "output_resources": {"ore_red": 1},
                "initial_resource_count": 10,
                "max_output": 10,
                "conversion_ticks": 1,
                "cooldown": 5,
            },
        },
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 5,
            "rewards": {
                # Inventory rewards (traditional)
                "inventory": {
                    "ore_red": 0.1,  # 0.1 reward per ore in inventory
                    "ore_red_max": 1.0,  # Max 1.0 total from ore
                    "heart": 1.0,  # 1.0 per heart
                },
                # Stats rewards (new feature)
                "stats": {
                    # Movement rewards
                    "action.move.success": 0.01,  # Small reward for exploration
                    "action.move.success_max": 0.5,  # Cap movement rewards
                    # Combat rewards
                    "action.attack.success": 0.2,  # Reward successful attacks
                    "action.attack.success_max": 2.0,  # Cap attack rewards
                    # Collection bonus (in addition to inventory reward)
                    "ore_red.gained": 0.05,  # Bonus for collecting ore
                    # Penalties
                    "action.failure_penalty": -0.05,  # Penalty for failed actions
                    "status.frozen.ticks": -0.01,  # Penalty for being frozen
                },
            },
        },
    }

    # Create a simple map
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.team_red", ".", "mine_red", ".", "agent.team_blue", "wall"],
        ["wall", ".", ".", ".", ".", ".", "wall"],
        ["wall", ".", "mine_red", ".", "mine_red", ".", "wall"],
        ["wall", ".", ".", ".", ".", ".", "wall"],
        ["wall", "agent.team_red", ".", "mine_red", ".", "agent.team_blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


def main():
    """Run a demonstration of stats rewards."""
    print("=== MettaGrid Stats Rewards Demo ===\n")

    env = create_stats_reward_environment()
    obs, info = env.reset()

    print(f"Environment created with {env.num_agents} agents")
    print("Reward configuration:")
    print("  Inventory rewards: ore_red=0.1, heart=1.0")
    print("  Stats rewards: move=0.01, attack=0.2, ore_gained=0.05")
    print("  Penalties: failed_action=-0.05, frozen=-0.01\n")

    # Get action indices
    action_names = env.action_names()
    move_idx = action_names.index("move")
    attack_idx = action_names.index("attack") if "attack" in action_names else None
    get_idx = (
        action_names.index("get_output")
        if "get_output" in action_names
        else action_names.index("get_items")
        if "get_items" in action_names
        else None
    )

    # Run a few steps to demonstrate rewards
    total_rewards = np.zeros(env.num_agents)

    for step in range(10):
        # Simple policy: move randomly, attack if possible
        actions = []
        for _ in range(env.num_agents):
            if step % 3 == 0:
                # Try to move
                action = move_idx
                direction = np.random.randint(4)  # random direction
            elif step % 3 == 1 and attack_idx is not None:
                # Try to attack
                action = attack_idx
                direction = 0
            elif get_idx is not None:
                # Try to collect items
                action = get_idx
                direction = 0
            else:
                action = 0  # noop
                direction = 0

            actions.append([action, direction])

        actions = np.array(actions, dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Track rewards
        total_rewards += rewards

        # Print step info
        if any(rewards != 0):
            print(f"Step {step + 1}:")
            for i, reward in enumerate(rewards):
                if reward != 0:
                    print(f"  Agent {i}: reward={reward:.3f}, total={total_rewards[i]:.3f}")

            # Get action success info
            success = env.action_success()
            print("  Actions:")
            for i, s in enumerate(success):
                action_name = action_names[actions[i, 0]]
                print(f"    Agent {i} - '{action_name}': {'✓' if s else '✗'}")

    # Print episode stats
    print("\n=== Episode Stats ===")
    stats = env.get_episode_stats()

    print("\nAgent Statistics (non-zero values):")
    for i, agent_stats in enumerate(stats["agent"]):
        print(f"\nAgent {i}:")
        printed_something = False
        for stat_name, value in sorted(agent_stats.items()):
            # Show meaningful stats only
            if value > 0 and not any(
                stat_name.endswith(suffix)
                for suffix in (
                    ".first_step",
                    ".last_step",
                    ".rate",
                    ".updates",
                    ".avg",
                    ".max",
                    ".min",
                    ".activity_rate",
                )
            ):
                print(f"  {stat_name}: {value}")
                printed_something = True
        if not printed_something:
            print("  (no significant stats)")

    print(f"\nFinal rewards: {total_rewards}")
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
