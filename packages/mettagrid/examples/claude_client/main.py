#!/usr/bin/env python3
"""Main script for Claude-powered MettaGrid RPC client.

This script demonstrates using Claude to play MettaGrid via the RPC server.

Usage:
    # Start the RPC server first:
    bazel run //cpp:mettagrid_rpc_server

    # Then run this client:
    python examples/claude_client/main.py
"""

import argparse
import struct
import time

from claude_player import ClaudePlayer
from game_config import create_simple_config, create_simple_map
from rpc_client import RPCClient


def decode_rewards(rewards_bytes: bytes, num_agents: int) -> list[float]:
    """Decode rewards from bytes (float32 array)."""
    return list(struct.unpack(f"{num_agents}f", rewards_bytes))


def decode_terminals(terminals_bytes: bytes, num_agents: int) -> list[bool]:
    """Decode terminals from bytes (bool array)."""
    return [bool(b) for b in terminals_bytes[:num_agents]]


def play_episode(
    client: RPCClient, player: ClaudePlayer, game_id: str, verbose: bool = True
):
    """Play a single episode using Claude."""

    # Create game config and map
    num_agents = 1
    config = create_simple_config(num_agents=num_agents)
    map_def = create_simple_map(width=10, height=10, num_agents=num_agents)

    # Create game
    client.create_game(game_id=game_id, config=config, map_def=map_def, seed=42)

    # Get initial state
    state = client.get_state(game_id)

    total_reward = 0.0
    step = 0
    max_steps = config.max_steps

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Starting episode with {num_agents} agent(s)")
        print(f"Max steps: {max_steps}")
        print(f"{'=' * 60}\n")

    # Game loop
    while step < max_steps:
        step += 1

        # Decode last rewards
        rewards = decode_rewards(state.rewards, num_agents)
        last_reward = rewards[0] if rewards else 0.0
        total_reward += last_reward

        # Ask Claude for action
        action = player.choose_action(
            step_num=step,
            total_steps=max_steps,
            obs_bytes=state.observations,
            num_agents=num_agents,
            tokens_per_agent=config.num_observation_tokens,
            last_reward=last_reward,
        )

        if verbose:
            reward_str = f"{last_reward:+.3f}"
            total_str = f"{total_reward:+.3f}"
            print(
                f"Step {step:3d}/{max_steps} | Action: {action} | Reward: {reward_str} | Total: {total_str}"
            )

        # Step the game
        state = client.step_game(game_id=game_id, flat_actions=[action] * num_agents)

        # Check for episode end
        terminals = decode_terminals(state.terminals, num_agents)
        if any(terminals):
            if verbose:
                print("\nEpisode terminated")
            break

        # Small delay to avoid overwhelming the API
        time.sleep(0.1)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Episode complete!")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"{'=' * 60}\n")

    # Clean up
    client.delete_game(game_id)

    return total_reward, step


def main():
    parser = argparse.ArgumentParser(description="Claude-powered MettaGrid RPC client")
    parser.add_argument("--host", default="127.0.0.1", help="RPC server host")
    parser.add_argument("--port", type=int, default=5858, help="RPC server port")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to play"
    )
    parser.add_argument(
        "--model", default="claude-3-5-sonnet-20241022", help="Claude model to use"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    print("Claude-powered MettaGrid RPC Client")
    print("=" * 60)
    print(f"Server: {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Initialize player
    player = ClaudePlayer(model=args.model)
    print(f"\nInitialized Claude player with model: {args.model}")

    # Connect to RPC server
    with RPCClient(host=args.host, port=args.port) as client:
        # Play episodes
        total_rewards = []
        total_steps = []

        for episode in range(args.episodes):
            print(f"\n{'*' * 60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'*' * 60}")

            game_id = f"claude_game_{episode}"
            reward, steps = play_episode(
                client=client,
                player=player,
                game_id=game_id,
                verbose=not args.quiet,
            )

            total_rewards.append(reward)
            total_steps.append(steps)

        # Summary
        if args.episodes > 1:
            print(f"\n{'=' * 60}")
            print(f"SUMMARY ({args.episodes} episodes)")
            print(f"{'=' * 60}")
            print(f"Average reward: {sum(total_rewards) / len(total_rewards):.3f}")
            print(f"Average steps: {sum(total_steps) / len(total_steps):.1f}")
            print(f"Total reward: {sum(total_rewards):.3f}")
            print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
