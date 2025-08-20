import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid, dtype_actions
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


class TestPerAgentResourceLoss:
    """Test per-agent resource loss functionality."""

    def test_different_agents_have_different_loss_rates(self):
        """Test that different agents can have different resource loss probabilities."""
        # Create a simple 3x3 grid with two agents from different groups
        game_map = [
            ["wall", "wall", "wall"],
            ["wall", "agent.player1", "agent.player2"],
            ["wall", "wall", "wall"],
        ]

        # Create environment configuration with different per-agent resource loss rates
        game_config = {
            "max_steps": 100,
            "num_agents": 2,
            "obs_width": 3,
            "obs_height": 3,
            "num_observation_tokens": 50,
            "inventory_item_names": ["heart", "battery_blue", "laser", "armor"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},  # Disable attack action to avoid resource conflicts
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": True},
            },
            "groups": {
                "player1": {
                    "id": 0,
                    "sprite": 0,
                    "props": {
                        "initial_inventory": {"heart": 5, "battery_blue": 3},
                        "resource_loss_probs": {"heart": 1.0, "battery_blue": 1.0},  # Complete loss for player1 agents
                    },
                },
                "player2": {
                    "id": 1,
                    "sprite": 0,
                    "props": {
                        "initial_inventory": {"heart": 5, "battery_blue": 3},
                        "resource_loss_probs": {"heart": 0.0, "battery_blue": 0.0},  # No loss for player2 agents
                    },
                }
            },
            "objects": {
                "wall": {"type_id": 1},
            },
            "agent": {
                "initial_inventory": {"heart": 5, "battery_blue": 3},
                "resource_loss_probs": {"heart": 0.0, "battery_blue": 0.0},  # No loss for default agent
            },
        }

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Reset environment
        obs, info = env.reset()

        # Get the agents' inventories through grid_objects
        grid_objects = env.grid_objects()
        agents = []
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agents.append(obj)

        assert len(agents) == 2, "Should find two agents in grid objects"

        # Verify both agents have the items initially
        for agent in agents:
            inventory = agent["inventory"]
            assert inventory[0] == 5, "Agent should have 5 hearts initially"
            assert inventory[1] == 3, "Agent should have 3 battery_blue initially"

        # Take a step with noop action
        noop_idx = env.action_names().index("noop")
        actions = np.array([[noop_idx, 0], [noop_idx, 0]], dtype=dtype_actions)

        obs, rewards, terminals, truncations, info = env.step(actions)

        # After one step, check that agents have different outcomes based on their loss rates
        grid_objects_after = env.grid_objects()
        agents_after = []
        for _obj_id, obj in grid_objects_after.items():
            if "agent_id" in obj:  # This is an agent
                agents_after.append(obj)

        assert len(agents_after) == 2, "Should find two agents in grid objects after step"

        # Sort agents by agent_id to ensure consistent testing
        agents_after.sort(key=lambda x: x["agent_id"])
        
        # Player1 agent (agent_id=0) should have lost all items
        player1_inventory = agents_after[0]["inventory"]
        assert 0 not in player1_inventory, "Player1 should have lost all hearts after one step"
        assert 1 not in player1_inventory, "Player1 should have lost all battery_blue after one step"
        assert len(player1_inventory) == 0, "Player1 should have no items left"
        
        # Player2 agent (agent_id=1) should have kept all items
        player2_inventory = agents_after[1]["inventory"]
        assert player2_inventory[0] == 5, "Player2 should still have 5 hearts after one step"
        assert player2_inventory[1] == 3, "Player2 should still have 3 battery_blue after one step"

    def test_agent_with_no_loss_keeps_items(self):
        """Test that an agent with no resource loss keeps its items."""
        # Create a simple 3x3 grid with one agent
        game_map = [
            ["wall", "wall", "wall"],
            ["wall", "agent.player", "wall"],
            ["wall", "wall", "wall"],
        ]

        # Create environment configuration with no resource loss for the agent
        game_config = {
            "max_steps": 100,
            "num_agents": 1,
            "obs_width": 3,
            "obs_height": 3,
            "num_observation_tokens": 50,
            "inventory_item_names": ["heart", "battery_blue", "laser", "armor"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},  # Disable attack action to avoid resource conflicts
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": True},
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
            "agent": {
                "initial_inventory": {"heart": 5, "battery_blue": 3},
                "resource_loss_probs": {},  # No resource loss configured
            },
        }

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Reset environment
        obs, info = env.reset()

        # Get the agent's inventory through grid_objects
        grid_objects = env.grid_objects()
        agent_obj = None
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj = obj
                break

        assert agent_obj is not None, "Should find an agent in grid objects"

        # Verify agent has the items initially
        inventory = agent_obj["inventory"]
        assert inventory[0] == 5, "Agent should have 5 hearts initially"
        assert inventory[1] == 3, "Agent should have 3 battery_blue initially"

        # Take multiple steps with noop action
        noop_idx = env.action_names().index("noop")
        actions = np.array([[noop_idx, 0]], dtype=dtype_actions)

        # Take 10 steps
        for _ in range(10):
            obs, rewards, terminals, truncations, info = env.step(actions)

        # After multiple steps with no resource loss, items should remain
        grid_objects_after = env.grid_objects()
        agent_obj_after = None
        for _obj_id, obj in grid_objects_after.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj_after = obj
                break

        assert agent_obj_after is not None, "Should find an agent in grid objects after steps"
        inventory_after = agent_obj_after["inventory"]
        assert inventory_after[0] == 5, "Hearts should remain after multiple steps"
        assert inventory_after[1] == 3, "Battery_blue should remain after multiple steps"
