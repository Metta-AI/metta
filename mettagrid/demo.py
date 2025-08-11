"""
Demo showing how to create a MettaGridEnv and run a simulation with random actions.
"""

import numpy as np

from metta.mettagrid.config import object as objects
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    EnvConfig,
    GameConfig,
    GroupConfig,
    InventoryRewards,
    MapConfig,
)
from metta.mettagrid.mettagrid_env import MettaGridEnv


def create_minimal_config():
    """Create a minimal environment configuration with basic objects."""
    return EnvConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=500,
            inventory_item_names=["ore_red", "battery_red", "heart"],
            # Agent configuration with simple rewards
            agent=AgentConfig(
                default_resource_limit=50,
                resource_limits={"heart": 100, "battery_red": 50},
                rewards=AgentRewards(inventory=InventoryRewards(ore_red=0.1, battery_red=0.2, heart=1.0)),
            ),
            groups={"agent": GroupConfig(id=0, sprite=0, props=AgentConfig())},
            # Simple actions
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            # Objects: Use predefined objects from objects.py
            objects={
                "wall": objects.wall,  # Required for boundaries
                "altar": objects.altar,
                "mine_red": objects.mine_red,
                "generator_red": objects.generator_red,
            },
            # Map configuration - use dict to avoid circular import
            map=MapConfig(
                map_gen={
                    "width": 20,
                    "height": 20,
                    "border_width": 2,
                    "seed": 42,
                    "root": {
                        "type": "metta.map.scenes.random.Random",
                        "params": {
                            "objects": {"altar": 2, "mine_red": 3, "generator_red": 2},
                            "agents": 2,
                        },
                    },
                }
            ),
        ),
        desync_episodes=True,
    )


def run_simulation():
    """Run a simulation with random actions."""
    print("Creating environment configuration...")
    env_config = create_minimal_config()

    # Create environment
    print("Creating MettaGridEnv...")
    env = MettaGridEnv(env_config=env_config, is_training=False)

    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset()

    # Get action space info
    num_agents = env.num_agents
    action_space = env._action_space
    print(f"Number of agents: {num_agents}")
    print(f"Action space: {action_space}")
    print(f"Action names: {env.action_names}")
    print(f"Max action args: {env.max_action_args}")

    # Run simulation loop
    print("\nRunning simulation with random actions...")
    done = False
    step_count = 0
    total_rewards = np.zeros(num_agents)

    while not done and step_count < 100:  # Limit to 100 steps for demo
        # Generate random actions for all agents
        # Action format: [action_id, arg1] for each agent
        actions = np.zeros((num_agents, 2), dtype=np.int32)

        for agent_idx in range(num_agents):
            # Random action ID (0 to num_actions-1)
            action_id = np.random.randint(0, action_space.nvec[0])
            # Random argument (0 to max_args-1 for that action)
            max_arg = env.max_action_args[action_id] if action_id < len(env.max_action_args) else 1
            arg = np.random.randint(0, max(1, max_arg))

            actions[agent_idx] = [action_id, arg]

        # Step environment
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Track rewards
        total_rewards += rewards

        # Check if all agents are done
        done = np.all(terminals) or np.all(truncations)

        step_count += 1

        # Print progress every 10 steps
        if step_count % 10 == 0:
            print(f"Step {step_count}: Rewards = {rewards}, Total = {total_rewards}")

    print(f"\nSimulation complete after {step_count} steps")
    print(f"Final total rewards: {total_rewards}")

    # Demonstrate set_env_cfg functionality
    print("\n--- Testing set_env_cfg ---")

    # Prepare a new configuration
    new_env_config = create_minimal_config()
    new_env_config.game.max_steps = 1000

    print(f"Current max_steps: {env._env_config.game.max_steps}")

    # Set the new configuration
    env.set_env_cfg(new_env_config)
    print(f"After set_env_cfg, max_steps: {env._env_config.game.max_steps}")

    # Reset to apply the new configuration
    obs, info = env.reset()
    print(f"After reset, max_steps is still: {env._env_config.game.max_steps}")

    # Clean up
    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    run_simulation()
