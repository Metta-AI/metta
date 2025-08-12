#!/usr/bin/env -S uv run

"""
Demo showing how to create a MettaGridEnv and run a simulation with random actions.
"""

import argparse
import time

import numpy as np
from rich.console import Console
from rich.text import Text

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
)
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.utils import make_level_map


def run_simulation(render=False):
    """Run a simulation with random actions.

    Args:
        render: If True, enable text-based rendering and display the grid.
    """
    print("Creating environment configuration...")

    # Create a minimal environment configuration with basic objects
    env_config = EnvConfig(
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
            # Create random level map
            level_map=make_level_map(
                width=20,
                height=20,
                objects={"altar": 2, "mine_red": 3, "generator_red": 2},
                num_agents=2,
                border_width=2,
                seed=42,
            ),
        ),
        desync_episodes=True,
    )

    # Create environment
    render_mode = "human" if render else None
    print(f"Creating MettaGridEnv (render mode: {render_mode or 'disabled'})...")
    env = MettaGridEnv(
        env_config=env_config,
        is_training=False,
        render_mode=render_mode,  # Use built-in renderer with nethack mode
    )

    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset()

    # Get action space info
    num_agents = env.num_agents
    action_space = env.action_space
    print(f"Number of agents: {num_agents}")
    print(f"Action space: {action_space}")
    print(f"Action names: {env.action_names}")
    print(f"Max action args: {env.max_action_args}")

    # Create console for rich text rendering if render mode enabled
    console = Console() if render else None

    # Clear the screen once at the start if rendering
    if render and console:
        console.clear()
        # Show initial state
        initial_render = env.render()
        if initial_render:
            console.print(Text("MettaGrid Simulation - Initial State", style="bold magenta"))
            console.print(Text("=" * 40, style="dim"))
            console.print(Text(initial_render, style="cyan"))
            console.print(Text("=" * 40, style="dim"))
            console.print("Press Ctrl+C to stop the simulation")
            time.sleep(2)  # Pause to show initial state

    # Run simulation loop
    if not render:
        print("\nRunning simulation with random actions...")
    done = False
    step_count = 0
    total_rewards = np.zeros(num_agents)

    while not done and step_count < 100:  # Limit to 100 steps for demo
        # Generate random actions for all agents using the action space
        # The action space is a MultiDiscrete space with shape (num_agents, 2)
        # where each agent has [action_id, arg1]
        actions = action_space.sample()

        # Step environment
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Track rewards
        total_rewards += rewards

        # Render if enabled
        if render and console:
            # Use the environment's built-in render method
            rendered_text = env.render()
            if rendered_text:
                # Move cursor to home position and display the grid
                print("\033[H", end="")  # Move to home position
                console.print(Text("MettaGrid Simulation", style="bold magenta"))
                console.print(Text("=" * 40, style="dim"))
                console.print(Text(rendered_text, style="cyan"))
                console.print(Text("-" * 40, style="dim"))
                console.print(f"Step: {step_count:3d} | Rewards: {rewards} | Total: {total_rewards}")
                console.print(Text("=" * 40, style="dim"))
                # Add a small delay to make the animation visible
                time.sleep(0.1)

        # Check if all agents are done
        done = np.all(terminals) or np.all(truncations)

        step_count += 1

        # Print progress every 10 steps (when not rendering)
        if not render and step_count % 10 == 0:
            print(f"Step {step_count}: Rewards = {rewards}, Total = {total_rewards}")

    print(f"\nSimulation complete after {step_count} steps")
    print(f"Final total rewards: {total_rewards}")

    # Demonstrate set_next_env_cfg functionality
    print("\n--- Testing set_next_env_cfg ---")

    # Prepare a new configuration
    new_env_config = EnvConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=1000,  # Different from original
            inventory_item_names=["ore_red", "battery_red", "heart"],
            agent=AgentConfig(
                default_resource_limit=50,
                resource_limits={"heart": 100, "battery_red": 50},
                rewards=AgentRewards(inventory=InventoryRewards(ore_red=0.1, battery_red=0.2, heart=1.0)),
            ),
            groups={"agent": GroupConfig(id=0, sprite=0, props=AgentConfig())},
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            objects={
                "wall": objects.wall,
                "altar": objects.altar,
                "mine_red": objects.mine_red,
                "generator_red": objects.generator_red,
            },
            level_map=make_level_map(
                width=20,
                height=20,
                objects={"altar": 2, "mine_red": 3, "generator_red": 2},
                num_agents=2,
                border_width=2,
                seed=42,
            ),
        ),
        desync_episodes=True,
    )

    print(f"Current max_steps: {env._env_config.game.max_steps}")

    # Set the new configuration for the next reset
    env.set_next_env_cfg(new_env_config)
    print(f"After set_next_env_cfg, max_steps still shows current: {env._env_config.game.max_steps}")

    # Reset to apply the new configuration
    obs, info = env.reset()
    print(f"After reset, max_steps updated to: {env._env_config.game.max_steps}")

    # Clean up
    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MettaGrid demo with random actions")
    parser.add_argument(
        "--render", action="store_true", help="Enable text-based rendering (nethack style) to visualize the simulation"
    )
    args = parser.parse_args()

    run_simulation(render=args.render)
