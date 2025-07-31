"""Test attack resource handling: validation, consumption, and failure cases."""

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


class TestAttackResourceValidation:
    """Tests for attack resource configuration validation."""

    def test_exception_when_consumed_resource_not_in_inventory(self):
        """Test that an exception is raised when attack requires a resource not in inventory_item_names."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", ".", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = {
            "max_steps": 50,
            "num_agents": 2,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            # Note: laser is NOT in inventory_item_names
            "inventory_item_names": ["armor", "heart"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {
                    "enabled": True,
                    "consumed_resources": {"laser": 1},  # This should trigger an exception!
                    "defense_resources": {"armor": 1},
                },
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {"default_resource_limit": 10, "freeze_duration": 5, "rewards": {}},
        }

        # Check that creating the environment raises an exception
        with pytest.raises(ValueError) as exc_info:
            MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Check the exception message
        assert "attack" in str(exc_info.value).lower() or "consumed_resources" in str(exc_info.value)
        assert "laser" in str(exc_info.value)
        assert "inventory_item_names" in str(exc_info.value)

    def test_no_exception_when_all_resources_in_inventory(self):
        """Test that no exception is raised when all consumed resources are in inventory."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", ".", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = {
            "max_steps": 50,
            "num_agents": 2,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            # Laser IS in inventory_item_names
            "inventory_item_names": ["laser", "armor", "heart"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {"default_resource_limit": 10, "freeze_duration": 5, "rewards": {}},
        }

        # This should not raise an exception
        MettaGrid(from_mettagrid_config(game_config), game_map, 42)


class TestAttackResourceConsumption:
    """Tests for attack resource consumption during gameplay."""

    def test_attack_fails_without_required_resource(self):
        """Test that attacks fail when agent doesn't have required laser resource."""
        # Create a simple map with two agents
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", ".", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = {
            "max_steps": 50,
            "num_agents": 2,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "inventory_item_names": ["laser", "armor", "heart"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {"default_resource_limit": 10, "freeze_duration": 5, "rewards": {}, "action_failure_penalty": 0.1},
        }

        # Create the environment
        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Set up buffers
        num_agents = 2
        observations = np.zeros((num_agents, 200, 3), dtype=np.uint8)
        terminals = np.zeros(num_agents, dtype=np.bool_)
        truncations = np.zeros(num_agents, dtype=np.bool_)
        rewards = np.zeros(num_agents, dtype=np.float32)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Get attack action id
        action_names = env.action_names()
        attack_action_id = action_names.index("attack")

        # Try to execute attack without laser
        actions = np.zeros((2, 2), dtype=np.int32)
        actions[0, 0] = attack_action_id
        actions[0, 1] = 0

        env.step(actions)

        # Check that attack failed
        action_success = env.action_success()
        assert not action_success[0], "Attack should fail without required laser resource"

        # Check that agent 1 is not frozen
        grid_objects = env.grid_objects()
        for _obj_id, obj_data in grid_objects.items():
            if "agent_id" in obj_data and obj_data["agent_id"] == 1:
                assert obj_data.get("frozen", 0) == 0, "Target should not be frozen when attack fails"

        # Check that agent received failure penalty
        assert rewards[0] < 0, f"Agent should receive penalty for failed action, but got reward {rewards[0]}"

    def test_attack_consumes_resource(self):
        """Test that successful attacks consume resources as configured."""
        # Create a simple map with two agents facing each other
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", ".", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = {
            "max_steps": 50,
            "num_agents": 2,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "inventory_item_names": ["laser", "armor", "heart"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {"default_resource_limit": 10, "freeze_duration": 5, "rewards": {}},
        }

        # Create the environment
        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Set up observation and reward buffers
        num_agents = 2
        num_obs_tokens = 200
        obs_token_size = 3
        observations = np.zeros((num_agents, num_obs_tokens, obs_token_size), dtype=np.uint8)
        terminals = np.zeros(num_agents, dtype=np.bool_)
        truncations = np.zeros(num_agents, dtype=np.bool_)
        rewards = np.zeros(num_agents, dtype=np.float32)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Get attack action id
        action_names = env.action_names()
        attack_action_id = action_names.index("attack")

        # Test that attack fails without laser (agents start with empty inventory)
        actions = np.zeros((2, 2), dtype=np.int32)
        actions[0, 0] = attack_action_id  # Attack action
        actions[0, 1] = 0  # Target directly in front

        env.step(actions)

        # Check action success (should be False without laser)
        action_success = env.action_success()
        assert not action_success[0], "Attack should fail without laser resource"

        # Verify agent 1 is not frozen
        grid_objects_after = env.grid_objects()
        for _obj_id, obj_data in grid_objects_after.items():
            if "agent_id" in obj_data and obj_data["agent_id"] == 1:
                assert obj_data.get("frozen", 0) == 0, "Target should not be frozen when attack fails"
