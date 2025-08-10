"""
Demo showing how to create a MettaGridEnv and run a simulation with random actions.
"""

import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv


def create_minimal_config():
    """Create a minimal game configuration with basic objects."""
    return OmegaConf.create(
        {
            "game": {
                "num_agents": 2,
                "max_steps": 500,
                "game_module": "metta_gridcraft",
                "randomize_player_spawns": True,
                "map_builder": {
                    "_target_": "metta.map.mapgen.MapGen",
                    "width": 20,
                    "height": 20,
                    "root": {
                        "_target_": "metta.map.scenes.random_objects.RandomObjects",
                        "objects": {
                            "object.altar": 2,
                            "object.mine": 3,
                            "object.generator": 2,
                            "agent": 2,
                        },
                    },
                },
            }
        }
    )


def run_simulation():
    """Run a simulation with random actions."""
    print("Creating environment configuration...")
    task_cfg = create_minimal_config()

    # Create curriculum with single task
    curriculum = SingleTaskCurriculum("demo_task", task_cfg)

    # Create environment
    print("Creating MettaGridEnv...")
    env = MettaGridEnv(curriculum=curriculum, is_training=False)

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

    # Demonstrate set_next_env_cfg functionality
    print("\n--- Testing set_next_env_cfg ---")

    # Prepare a new configuration
    new_config = OmegaConf.to_container(task_cfg.game)
    new_config["max_steps"] = 1000
    new_config["num_agents"] = 2  # Keep same number of agents

    print(f"Current max_steps: {env._env_cfg['max_steps']}")

    # Set the next configuration
    env.set_next_env_cfg(new_config)
    print(f"Set next_env_cfg with max_steps: {env._next_env_cfg['max_steps']}")

    # Reset to apply the new configuration
    obs, info = env.reset()
    print(f"After reset, max_steps: {env._env_cfg['max_steps']}")

    # Clean up
    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    run_simulation()
