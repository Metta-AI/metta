#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "gymnasium",
#     "omegaconf",
#     "typing-extensions",
#     "pydantic",
#     "rich",
# ]
# ///

"""Rendering Demo - Visual MettaGrid environment demonstration.

This demo shows how to use MettaGrid environments with visual rendering enabled,
including both text-based (nethack-style) and pygame rendering modes.

Run with: uv run python mettagrid/demos/demo_render.py (from project root)
"""

import argparse
import time

import numpy as np

# Shared demo configuration
from demo_config import create_demo_config
from rich.console import Console
from rich.text import Text

# MettaGrid imports
from metta.mettagrid import AutoResetEnv
from metta.mettagrid.config import EnvConfig
from metta.mettagrid.room.random import Random as RandomRoomBuilder
from metta.mettagrid.room.utils import make_level_map


def create_visual_demo_config() -> EnvConfig:
    """Create a demo config optimized for visual rendering with guaranteed visible objects."""
    # Use the shared demo config as base
    base_config = create_demo_config(
        num_agents=2, map_width=15, map_height=15, max_steps=200, obs_width=7, obs_height=7
    )

    # Create a custom level with explicit object placement to ensure visibility
    map_builder = RandomRoomBuilder(
        agents=2,
        width=15,
        height=15,
        border_width=2,
        objects={
            "mine_red": 4,  # More resource sources
            "generator_red": 3,  # More generators
            "altar": 2,  # Multiple altars
        },
    )

    # Build the level map
    level_map = map_builder.build()

    # Create new config with the explicit level
    return EnvConfig(game=base_config.game, level_map=level_map)


def demo_text_rendering():
    """Demonstrate text-based (nethack-style) rendering."""
    print("TEXT-BASED RENDERING DEMO")
    print("=" * 60)
    print("This demo shows nethack-style text rendering in the terminal.")
    print("Press Ctrl+C to stop the demo.")
    print()

    try:
        env_config = create_visual_demo_config()

        # Create environment with text rendering
        env = AutoResetEnv(
            env_config=env_config,
            render_mode="human",  # Text-based rendering mode using NethackRenderer
            is_training=False,
        )

        print("Created text-based environment:")
        map_shape = env_config.level_map.grid.shape
        print(f"   - Map size: {map_shape[1]}x{map_shape[0]}")
        print(f"   - Agents: {env.num_agents}")
        print(f"   - Max steps: {env.max_steps}")
        print("   - Render mode: human (text via NethackRenderer)")
        print(f"   - Objects in game config: {list(env_config.game.objects.keys())}")
        print()

        # Create console for rich text rendering
        console = Console()
        console.clear()

        # Reset and run simulation
        observations, _ = env.reset(seed=42)

        steps = 0
        max_demo_steps = 100
        total_rewards = np.zeros(env.num_agents)

        print("Starting text-based simulation...")

        # Show initial state
        initial_render = env.render()
        if initial_render:
            print("Initial state:")
            print(initial_render)
            print("=" * 50)
        else:
            print("Warning: No initial rendering output")

        while steps < max_demo_steps:
            # Take random actions
            actions = env.action_space.sample()
            observations, rewards, terminals, truncations, _ = env.step(actions)

            total_rewards += rewards

            # Get text rendering
            rendered_text = env.render()
            if rendered_text:
                # Clear screen and display with Rich
                console.clear()
                console.print(Text("MettaGrid Text Rendering Demo", style="bold magenta"))
                console.print(Text("=" * 50, style="dim"))
                console.print(Text(rendered_text, style="cyan"))
                console.print(Text("-" * 50, style="dim"))
                console.print(f"Step: {steps:3d} | Rewards: {rewards} | Total: {total_rewards}")
                console.print(Text("=" * 50, style="dim"))
                console.print("Press Ctrl+C to stop")

            # Add delay to make it watchable
            time.sleep(0.2)
            steps += 1

            # Reset if episode ends
            if terminals.any() or truncations.any():
                print(f"\nEpisode completed at step {steps}")
                observations, _ = env.reset()
                total_rewards = np.zeros(env.num_agents)
                steps = 0
                time.sleep(1)  # Pause between episodes

        env.close()
        print("\nText rendering demo completed")

    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        env.close()
    except Exception as e:
        print(f"Text rendering failed: {e}")
        print("Note: Text rendering may require specific terminal capabilities")


def demo_pygame_rendering():
    """Demonstrate pygame-based rendering with AutoResetEnv."""
    print("\nPYGAME RENDERING DEMO")
    print("=" * 60)
    print("This demo will open a pygame window showing the environment.")
    print("Close the window or press Ctrl+C to continue.")
    print()

    try:
        env_config = create_visual_demo_config()

        # Create environment with pygame rendering
        env = AutoResetEnv(
            env_config=env_config,
            render_mode="human",  # This enables pygame rendering
            is_training=False,
        )

        print("Created visual environment:")
        map_shape = env_config.level_map.grid.shape
        print(f"   - Map size: {map_shape[1]}x{map_shape[0]}")
        print(f"   - Agents: {env.num_agents}")
        print(f"   - Max steps: {env.max_steps}")
        print("   - Render mode: human (pygame)")

        # Reset and run for a while
        observations, _ = env.reset(seed=42)
        print("Environment reset, starting visual demonstration...")

        steps = 0
        max_demo_steps = 300  # Run for 300 steps or until episode ends

        while steps < max_demo_steps:
            # Take random actions
            actions = env.action_space.sample()
            observations, rewards, terminals, truncations, _ = env.step(actions)

            # Add a small delay to make it watchable
            time.sleep(0.05)
            steps += 1

            # Print occasional updates
            if steps % 50 == 0:
                print(f"   Step {steps}: reward sum = {rewards.sum():.2f}")

            # Reset if episode ends
            if terminals.any() or truncations.any():
                print(f"   Episode completed at step {steps}")
                observations, _ = env.reset()
                steps = 0

        env.close()
        print("Pygame rendering demo completed")

    except Exception as e:
        print(f"Pygame rendering failed: {e}")
        print("Note: pygame rendering requires pygame to be installed")
        print("Install with: pip install pygame")


def demo_custom_visual_level():
    """Demonstrate rendering with a custom designed level."""
    print("\nCUSTOM VISUAL LEVEL DEMO")
    print("=" * 60)

    try:
        # Create a more interesting custom level
        grid = np.array(
            [
                ["wall", "wall", "wall", "wall", "wall", "wall", "wall", "wall", "wall"],
                ["wall", "empty", "empty", "mine_red", "empty", "mine_red", "empty", "empty", "wall"],
                ["wall", "empty", "agent.agent", "empty", "empty", "empty", "generator_red", "empty", "wall"],
                ["wall", "empty", "empty", "empty", "wall", "empty", "empty", "empty", "wall"],
                ["wall", "mine_red", "empty", "empty", "altar", "empty", "empty", "empty", "wall"],
                ["wall", "empty", "empty", "empty", "wall", "empty", "empty", "empty", "wall"],
                ["wall", "empty", "generator_red", "empty", "empty", "empty", "agent.agent", "empty", "wall"],
                ["wall", "empty", "empty", "mine_red", "empty", "mine_red", "empty", "empty", "wall"],
                ["wall", "wall", "wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ],
            dtype="<U50",
        )

        # Create level map
        level_map = make_level_map(grid, labels=["custom_visual_level"])

        # Create environment config with custom level
        base_config = create_demo_config(num_agents=2, max_steps=150)
        env_config = EnvConfig(game=base_config.game, level_map=level_map)

        # Create environment with rendering
        env = AutoResetEnv(
            env_config=env_config,
            render_mode="human",  # Use text rendering for custom level
            is_training=False,
        )

        print("Created custom visual level:")
        print(f"   - Grid shape: {grid.shape}")
        print(f"   - Labels: {level_map.labels}")
        print(f"   - Agents: {env.num_agents}")

        # Reset and demonstrate
        observations, _ = env.reset(seed=123)
        print("Custom level rendering - agents will move around the designed map...")

        # Show initial state
        initial_render = env.render()
        if initial_render:
            print("\nInitial state:")
            print(initial_render)
            print("\nRunning simulation...")
            time.sleep(2)

        for step in range(75):  # Shorter demo
            actions = env.action_space.sample()
            observations, rewards, terminals, truncations, _ = env.step(actions)

            # Render every few steps
            if step % 10 == 0:
                rendered = env.render()
                if rendered:
                    print(f"\nStep {step}:")
                    print(rendered)

            time.sleep(0.1)

            if step % 25 == 0:
                print(f"   Step {step}: total reward = {rewards.sum():.2f}")

            if terminals.any() or truncations.any():
                print(f"   Episode completed at step {step}")
                break

        env.close()
        print("Custom visual level demo completed")

    except Exception as e:
        print(f"Custom visual level failed: {e}")


def run_interactive_simulation(render_mode: str = "ansi"):
    """Run an interactive simulation with the specified render mode."""
    print(f"\nINTERACTIVE SIMULATION ({render_mode.upper()} MODE)")
    print("=" * 60)

    env_config = create_visual_demo_config()

    # Create environment with specified render mode
    env = AutoResetEnv(
        env_config=env_config,
        render_mode=render_mode,
        is_training=False,
    )

    print(f"Environment created with {render_mode} rendering")
    print(f"Number of agents: {env.num_agents}")
    print(f"Action space: {env.action_space}")

    # Get action space info for random sampling
    num_agents = env.num_agents
    action_space = env.action_space

    # Reset environment
    obs, info = env.reset(seed=42)

    # Show initial render if available
    initial_render = env.render()
    if initial_render and render_mode == "human":
        print("\nInitial state:")
        print(initial_render)

    # Run simulation
    done = False
    step_count = 0
    total_rewards = np.zeros(num_agents)

    try:
        while not done and step_count < 150:
            # Generate random actions
            actions = action_space.sample()

            # Step environment
            obs, rewards, terminals, truncations, info = env.step(actions)

            # Track rewards
            total_rewards += rewards

            # Render if text mode and every few steps
            if render_mode == "human" and step_count % 10 == 0:
                rendered_text = env.render()
                if rendered_text:
                    print(f"\nStep {step_count}:")
                    print(rendered_text)
                    print(f"Rewards: {rewards} | Total: {total_rewards}")

            # Check if done
            done = np.all(terminals) or np.all(truncations)
            step_count += 1

            # Add delay
            time.sleep(0.1)

            # Print progress
            if step_count % 25 == 0:
                print(f"Step {step_count}: Rewards = {rewards}, Total = {total_rewards}")

        print(f"\nSimulation complete after {step_count} steps")
        print(f"Final total rewards: {total_rewards}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    finally:
        env.close()


def main():
    """Run visual rendering demos."""
    parser = argparse.ArgumentParser(description="Run MettaGrid visual rendering demos")
    parser.add_argument(
        "--mode",
        choices=["text", "pygame", "custom", "interactive", "all"],
        default="all",
        help="Which rendering demo to run",
    )
    parser.add_argument(
        "--render-mode", choices=["human", "miniscope"], default="human", help="Rendering mode for interactive demo"
    )

    args = parser.parse_args()

    print("METTAGRID VISUAL RENDERING DEMO")
    print("=" * 80)
    print("This demo showcases MettaGrid's rendering capabilities.")
    print()

    try:
        if args.mode in ["text", "all"]:
            demo_text_rendering()

        if args.mode in ["pygame", "all"]:
            demo_pygame_rendering()

        if args.mode in ["custom", "all"]:
            demo_custom_visual_level()

        if args.mode == "interactive":
            run_interactive_simulation(args.render_mode)

        if args.mode == "all":
            print("\n" + "=" * 80)
            print("ALL RENDERING DEMOS COMPLETED")
            print("=" * 80)
            print("Rendering demonstrations completed successfully!")
            print()
            print("Key features demonstrated:")
            print("   - Text-based (ansi) rendering for terminal display")
            print("   - Pygame-based (human) rendering for graphical display")
            print("   - Custom level visualization")
            print("   - Interactive simulation with real-time rendering")
            print()
            print("Usage patterns:")
            print("   - Use render_mode='human' for text-based rendering (NethackRenderer)")
            print("   - Use render_mode='miniscope' for miniscope visualization")
            print("   - Custom levels work with any render mode")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\nRendering demos interrupted by user")
    except Exception as e:
        print(f"\nRendering demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
