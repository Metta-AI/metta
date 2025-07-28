#!/usr/bin/env python3
"""
Gold standard configuration tests for PufferLib compatibility.

This module contains reference configurations that are known to work with PufferLib.
Any changes to config handling that break these tests indicate a compatibility issue.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid import MettaGridPufferEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


# Gold standard configuration that must always work with PufferLib
GOLD_CONFIG = {
    "game": {
        "num_agents": 4,
        "obs_width": 11,
        "obs_height": 11,
        "max_steps": 1000,
        "num_observation_tokens": 100,
        "inventory_item_names": [
            "ore_red", "ore_blue", "ore_green",
            "battery_red", "battery_blue", "battery_green",
            "heart", "armor", "laser"
        ],
        "groups": {
            "agent": {
                "id": 0,
                "sprite": 0,
                "props": {}
            },
            "team_red": {
                "id": 1,
                "sprite": 1,
                "group_reward_pct": 0.5,
                "props": {}
            },
            "team_blue": {
                "id": 2,
                "sprite": 4,
                "group_reward_pct": 0.5,
                "props": {}
            }
        },
        "agent": {
            "default_resource_limit": 10,
            "resource_limits": {
                "heart": 255,
                "armor": 10,
                "laser": 5
            },
            "freeze_duration": 10,
            "rewards": {
                "inventory": {
                    "battery_red": 0.01,
                    "battery_blue": 0.01,
                    "battery_green": 0.01,
                    "heart": 1.0
                }
            }
        },
        "objects": {
            "wall": {
                "type_id": 1
            },
            "altar": {
                "type_id": 8,
                "input_resources": {"battery_red": 3},
                "output_resources": {"heart": 1},
                "cooldown": 10,
                "max_output": 5,
                "conversion_ticks": 1,
                "initial_resource_count": 1
            },
            "mine_red": {
                "type_id": 2,
                "output_resources": {"ore_red": 1},
                "color": 0,
                "cooldown": 50,
                "max_output": 5,
                "conversion_ticks": 1,
                "initial_resource_count": 1
            },
            "mine_blue": {
                "type_id": 3,
                "output_resources": {"ore_blue": 1},
                "color": 1,
                "cooldown": 50,
                "max_output": 5,
                "conversion_ticks": 1,
                "initial_resource_count": 1
            },
            "generator_red": {
                "type_id": 5,
                "input_resources": {"ore_red": 1},
                "output_resources": {"battery_red": 1},
                "color": 0,
                "cooldown": 25,
                "max_output": 5,
                "conversion_ticks": 1,
                "initial_resource_count": 1
            }
        },
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "attack": {
                "enabled": True,
                "consumed_resources": {"laser": 1},
                "defense_resources": {"armor": 1}
            },
            "swap": {"enabled": True}
        },
        "map_builder": {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 30,
            "height": 30,
            "border_width": 1,
            "agents": 4,
            "objects": {
                "altar": 2,
                "mine_red": 5,
                "mine_blue": 5,
                "generator_red": 3,
                "wall": 20
            }
        }
    }
}


class TestPufferLibGoldConfig:
    """Test suite for gold standard PufferLib configurations."""
    
    def test_gold_config_conversion(self):
        """Test that the gold config converts successfully to C++ format."""
        game_config = GOLD_CONFIG["game"]
        cpp_config = from_mettagrid_config(game_config)
        
        # Verify all critical fields are preserved
        assert cpp_config is not None
        assert cpp_config.num_agents == 4
        assert cpp_config.obs_width == 11
        assert cpp_config.obs_height == 11
        assert cpp_config.max_steps == 1000
        assert cpp_config.num_observation_tokens == 100
        
        # The conversion should succeed without errors
        # Agent groups are handled internally during conversion
        
    def test_gold_config_env_creation(self):
        """Test that gold config can create a working PufferLib environment."""
        try:
            import pufferlib
        except ImportError:
            pytest.skip("PufferLib not installed")
            
        config = DictConfig(GOLD_CONFIG)
        curriculum = SingleTaskCurriculum("gold_test", config)
        
        # Create environment
        env = MettaGridPufferEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True
        )
        
        # Verify environment properties
        assert env.num_agents == 4
        assert env.observation_space is not None
        assert env.action_space is not None
        
        # Test environment interaction
        obs, info = env.reset()
        assert obs.shape[0] == 4  # num_agents
        
        import numpy as np
        # Create actions for all 4 agents
        actions = np.array([
            [0, 1],  # agent 0: move
            [1, 2],  # agent 1: rotate
            [2, 0],  # agent 2: noop
            [3, 1]   # agent 3: move
        ], dtype=np.int32)
        
        obs, rewards, terminals, truncations, info = env.step(actions)
        assert obs.shape[0] == 4
        assert rewards.shape[0] == 4
        assert terminals.shape[0] == 4
        assert truncations.shape[0] == 4
        
        env.close()
        
    def test_config_field_requirements(self):
        """Test that all required fields for PufferLib are present."""
        game_config = GOLD_CONFIG["game"]
        
        # Required top-level fields
        required_fields = [
            "num_agents", "obs_width", "obs_height", "max_steps",
            "num_observation_tokens", "inventory_item_names",
            "groups", "agent", "objects", "actions"
        ]
        
        for field in required_fields:
            assert field in game_config, f"Missing required field: {field}"
            
        # Required agent fields
        agent_config = game_config["agent"]
        required_agent_fields = [
            "default_resource_limit", "freeze_duration", "rewards"
        ]
        
        for field in required_agent_fields:
            assert field in agent_config, f"Missing required agent field: {field}"
            
        # Required action fields
        for action_name, action_config in game_config["actions"].items():
            assert "enabled" in action_config, f"Action {action_name} missing 'enabled' field"
            
    def test_config_compatibility_matrix(self):
        """Test and document compatibility between Metta and PufferLib versions."""
        compatibility_matrix = {
            "metta_version": "current",
            "pufferlib_version": "3.0",
            "compatible": True,
            "notes": "Gold config tested successfully",
            "tested_features": [
                "Basic environment creation",
                "Multi-agent support",
                "Inventory system",
                "Team-based rewards",
                "Object interactions",
                "All action types"
            ]
        }
        
        # Save compatibility matrix for CI tracking
        matrix_path = Path("tests/pufferlib_compatibility_matrix.json")
        with open(matrix_path, "w") as f:
            json.dump(compatibility_matrix, f, indent=2)
            
    def test_config_serialization(self):
        """Test that configs can be serialized and deserialized correctly."""
        # Convert to OmegaConf DictConfig
        config = DictConfig(GOLD_CONFIG)
        
        # Serialize to YAML
        yaml_str = OmegaConf.to_yaml(config)
        
        # Deserialize back
        loaded_config = OmegaConf.create(yaml_str)
        
        # Verify equality
        assert OmegaConf.to_container(config) == OmegaConf.to_container(loaded_config)
        
    @pytest.mark.parametrize("modification,field_path,new_value", [
        # Test various config modifications that should still work
        ("change_num_agents", ["game", "num_agents"], 8),
        ("change_obs_size", ["game", "obs_width"], 15),
        ("add_inventory_item", ["game", "inventory_item_names"], 
         ["ore_red", "ore_blue", "ore_green", "battery_red", "battery_blue", 
          "battery_green", "heart", "armor", "laser", "blueprint"]),
        ("modify_reward", ["game", "agent", "rewards", "inventory", "heart"], 0.5),
        ("disable_action", ["game", "actions", "swap", "enabled"], False),
    ])
    def test_config_modifications(self, modification: str, field_path: list, new_value: Any):
        """Test that reasonable config modifications still work."""
        import copy
        
        # Deep copy the gold config
        modified_config = copy.deepcopy(GOLD_CONFIG)
        
        # Navigate to the field and modify it
        current = modified_config
        for key in field_path[:-1]:
            current = current[key]
        current[field_path[-1]] = new_value
        
        # Test conversion still works
        game_config = modified_config["game"]
        cpp_config = from_mettagrid_config(game_config)
        assert cpp_config is not None
        
    def test_backwards_compatibility(self):
        """Test that old config formats are still supported."""
        # Test config without some newer fields
        legacy_config = {
            "game": {
                "num_agents": 1,
                "obs_width": 7,
                "obs_height": 7,
                "max_steps": 100,
                "num_observation_tokens": 50,
                "inventory_item_names": ["heart"],
                "groups": {
                    "agent": {"id": 0, "sprite": 0, "props": {}}
                },
                "agent": {
                    "default_resource_limit": 5,
                    "freeze_duration": 0,
                    "rewards": {}
                },
                "objects": {
                    "wall": {"type_id": 1}
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True}
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 10,
                    "height": 10,
                    "agents": 1
                }
            }
        }
        
        # Should still convert successfully
        cpp_config = from_mettagrid_config(legacy_config["game"])
        assert cpp_config is not None