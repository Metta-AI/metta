#!/usr/bin/env python3
"""Test script to verify RPC connection without requiring Claude API.

This script tests that:
1. The RPC client can connect to the server
2. Games can be created and deleted
3. Basic state queries work
"""

import sys

from game_config import create_simple_config, create_simple_map
from rpc_client import RPCClient


def test_connection():
    """Test basic RPC connection and operations."""
    print("Testing RPC connection...")

    try:
        # Connect to server
        with RPCClient(host="127.0.0.1", port=5858) as client:
            # Create game config
            config = create_simple_config(num_agents=1)
            map_def = create_simple_map(width=10, height=10, num_agents=1)

            # Create game
            game_id = "test_game"
            print(f"\nCreating game '{game_id}'...")
            client.create_game(game_id=game_id, config=config, map_def=map_def, seed=42)

            # Get initial state
            print("Fetching initial state...")
            state = client.get_state(game_id)
            print(f"  - Observations: {len(state.observations)} bytes")
            print(f"  - Rewards: {len(state.rewards)} bytes")
            print(f"  - Terminals: {len(state.terminals)} bytes")

            # Step with random action
            print("\nStepping game with noop action (0)...")
            state = client.step_game(game_id=game_id, flat_actions=[0])
            print(f"  - Observations: {len(state.observations)} bytes")
            print(f"  - Rewards: {len(state.rewards)} bytes")

            # Delete game
            print(f"\nDeleting game '{game_id}'...")
            client.delete_game(game_id)

            print("\n✓ All RPC operations successful!")
            return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_connection())
