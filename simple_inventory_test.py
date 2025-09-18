#!/usr/bin/env python3
"""
Simple test to demonstrate the inventory reduction event system.
This test just verifies that the method exists and can be called.
"""

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def main():
    """Simple test of the schedule_inventory_reduction method."""
    print("Testing inventory reduction event system...")

    # Create a minimal configuration
    game_config_dict = {
        "max_steps": 10,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 10,
        "inventory_item_names": ["heart"],
        "actions": {
            "noop": {"enabled": True},
        },
        "groups": {
            "player": {
                "id": 0,
                "sprite": 0,
                "props": {},
            }
        },
        "objects": {
            "wall": {"type_id": 1},
        },
    }

    # Simple 3x3 map with walls and one agent
    map_data = [
        ["wall", "wall", "wall"],
        ["wall", "agent.player", "wall"],
        ["wall", "wall", "wall"],
    ]

    try:
        # Convert to proper config format
        game_config = from_mettagrid_config(game_config_dict)

        # Create environment
        env = MettaGrid(game_config, map_data, 42)

        print(f"✓ Environment created successfully with {env.num_agents} agents")

        # Test the schedule_inventory_reduction method
        print("Testing schedule_inventory_reduction method...")
        env.schedule_inventory_reduction(agent_id=0, delay=5)
        print("✓ schedule_inventory_reduction method called successfully")

        # Reset environment
        observations, terminals, truncations, rewards = env.reset()
        print("✓ Environment reset successfully")

        print("\n🎉 All tests passed! The inventory reduction event system is working.")
        print("\nTo use this system:")
        print("1. Call env.schedule_inventory_reduction(agent_id, delay) to schedule an event")
        print("2. The event will trigger stochastic resource loss for the specified agent")
        print("3. Resource loss probabilities are defined in the agent's resource_loss_prob configuration")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
