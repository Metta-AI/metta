#!/usr/bin/env python
"""Simplified interactive play script.

This allows you to play as an agent or watch a trained policy play.
"""

import argparse

import numpy as np
import torch

from configs.python.agents import simple_cnn_agent

# Import Python configs
from configs.python.environments import NavigationMaze, NavigationWalls
from metta.agent import MettaAgent
from metta.agent.policy_store import PolicyStore
from mettagrid.curriculum import curriculum_from_config_path
from mettagrid.mettagrid_env import MettaGridEnv


class HumanAgent:
    """Human-controlled agent using keyboard input."""

    def __init__(self, env: MettaGridEnv):
        self.env = env
        self.action_map = {
            "w": (0, 1),  # move up
            "a": (0, 3),  # move left
            "d": (0, 5),  # move right
            "s": (0, 7),  # move down
            "q": (0, 2),  # move up-left
            "e": (0, 8),  # move up-right
            "z": (0, 6),  # move down-left
            "c": (0, 9),  # move down-right
            " ": (2, 0),  # pickup
            "r": (1, 0),  # rotate
            "f": (3, 0),  # use/fire
            "x": (0, 4),  # stay
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from keyboard input."""
        print("\nControls: WASD=move, QE/ZC=diagonal, Space=pickup, R=rotate, F=use, X=stay")

        while True:
            key = input("Action: ").lower().strip()
            if key in self.action_map:
                action_type, action_arg = self.action_map[key]
                return np.array([[action_type, action_arg]], dtype=np.int32)
            else:
                print(f"Unknown key '{key}'. Try again.")


def play_human(env_name: str = "navigation/walls", render_mode: str = "human"):
    """Play as a human using keyboard controls."""
    # Create environment
    if env_name == "navigation/walls":
        env_config = NavigationWalls()
    elif env_name == "navigation/maze":
        env_config = NavigationMaze()
    else:
        # Default
        env_config = NavigationWalls()

    curriculum = curriculum_from_config_path(env_config.to_dict(), {})
    env = MettaGridEnv(curriculum, render_mode=render_mode)

    # Create human agent
    agent = HumanAgent(env)

    # Game loop
    obs, info = env.reset()
    env.render()

    total_reward = 0
    steps = 0

    print(f"\n🎮 Playing {env_name}")
    print("Try to reach all the altars!")

    while True:
        # Get human action
        action = agent.get_action(obs)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward[0]
        steps += 1

        # Render
        env.render()

        # Print info
        if reward[0] != 0:
            print(f"Reward: {reward[0]:.2f} (Total: {total_reward:.2f})")

        # Check if done
        if terminated[0] or truncated[0]:
            print("\n🏁 Episode finished!")
            print(f"Total reward: {total_reward:.2f} in {steps} steps")

            play_again = input("\nPlay again? (y/n): ").lower().strip()
            if play_again == "y":
                obs, info = env.reset()
                env.render()
                total_reward = 0
                steps = 0
            else:
                break

    env.close()


def play_agent(
    agent_source: str,
    env_name: str = "navigation/walls",
    render_mode: str = "human",
    speed: float = 0.5,
    device: str = "cuda",
):
    """Watch a trained agent play."""
    # Load agent
    if agent_source.startswith(("wandb://", "s3://", "file://")):
        # Load from URI
        print(f"📦 Loading policy from {agent_source}")
        config = {"device": device, "wandb": {"entity": "your-entity", "project": "metta"}}
        policy_store = PolicyStore(config, None)
        policy_pr = policy_store.policy(agent_source)
        agent = policy_pr.policy().to(device)
    else:
        # Load from checkpoint
        print(f"📂 Loading checkpoint from {agent_source}")
        agent_config = simple_cnn_agent()
        agent = MettaAgent(**agent_config)
        checkpoint = torch.load(agent_source, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.to(device)

    agent.eval()

    # Create environment
    if env_name == "navigation/walls":
        env_config = NavigationWalls()
    elif env_name == "navigation/maze":
        env_config = NavigationMaze()
    else:
        env_config = NavigationWalls()

    curriculum = curriculum_from_config_path(env_config.to_dict(), {})
    env = MettaGridEnv(curriculum, render_mode=render_mode)

    # Activate agent actions
    agent.activate_actions(env.action_names, env.max_action_args, device)

    # Game loop
    print(f"\n🤖 Watching agent play {env_name}")
    print("Press Ctrl+C to stop")

    try:
        episodes = 0
        while True:
            obs, info = env.reset()
            env.render()

            total_reward = 0
            steps = 0

            # Initialize LSTM state
            from metta.agent.policy_state import PolicyState

            state = PolicyState()

            while True:
                # Get agent action
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                    actions, _, _, _, _ = agent(obs_tensor, state)
                    action = actions[0].cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
                total_reward += reward[0]
                steps += 1

                # Render
                env.render()

                # Sleep for visualization
                import time

                time.sleep(speed)

                # Check if done
                if terminated[0] or truncated[0]:
                    episodes += 1
                    print(f"Episode {episodes}: {total_reward:.2f} reward in {steps} steps")
                    time.sleep(1)  # Pause between episodes
                    break

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Play Metta environments")
    parser.add_argument(
        "--mode", type=str, default="human", choices=["human", "agent"], help="Play mode: human control or watch agent"
    )
    parser.add_argument("--agent", type=str, help="Agent checkpoint or URI (for agent mode)")
    parser.add_argument("--env", type=str, default="navigation/walls", help="Environment to play")
    parser.add_argument("--render", type=str, default="human", choices=["human", "miniscope"], help="Render mode")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed for agent mode (seconds per step)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for agent mode")

    args = parser.parse_args()

    if args.mode == "human":
        play_human(env_name=args.env, render_mode=args.render)
    elif args.mode == "agent":
        if not args.agent:
            parser.error("--agent required for agent mode")
        play_agent(
            agent_source=args.agent,
            env_name=args.env,
            render_mode=args.render,
            speed=args.speed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
