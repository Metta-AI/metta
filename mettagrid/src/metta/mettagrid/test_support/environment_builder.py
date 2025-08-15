"""Test utilities and environment builder for mettagrid tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from metta.mettagrid.level_builder import create_grid
from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate, dtype_actions  # noqa: F401
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


@dataclass
class DefaultEnvConfig:
    """Default environment configuration values."""

    NUM_AGENTS: int = 2
    OBS_HEIGHT: int = 11
    OBS_WIDTH: int = 11
    NUM_OBS_TOKENS: int = 200
    OBS_TOKEN_SIZE: int = 3


class TestEnvironmentBuilder:
    """Helper class to build test environments with different configurations."""

    @staticmethod
    def create_basic_grid(width: int = 8, height: int = 4) -> np.ndarray:
        """Create a basic grid with walls around perimeter."""
        game_map = create_grid(height, width)
        game_map[0, :] = "wall"
        game_map[-1, :] = "wall"
        game_map[:, 0] = "wall"
        game_map[:, -1] = "wall"
        return game_map

    @staticmethod
    def place_agents(
        game_map: np.ndarray, positions: List[Tuple[int, int]], agent_type: str = "agent.red"
    ) -> np.ndarray:
        """Place agents at specified positions.

        Note: positions are (y, x) indexing into the numpy array.
        """
        for _, (y, x) in enumerate(positions):
            game_map[y, x] = agent_type
        return game_map

    @staticmethod
    def make_test_config(
        num_agents: int = 1,
        map: Optional[list[list[str]]] = None,
        actions: Optional[dict[str, dict[str, Any]]] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        num_observation_tokens: Optional[int] = None,
        max_steps: int = 1000,
        inventory_item_names: Optional[list[str]] = None,
        no_agent_interference: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a test configuration for MettaGrid.

        Args:
            num_agents: Number of agents in the environment
            map: 2D list of strings representing the map
            actions: Action configuration overrides
            obs_width: Observation width (defaults to EnvConfig.OBS_WIDTH)
            obs_height: Observation height (defaults to EnvConfig.OBS_HEIGHT)
            num_observation_tokens: Number of observation tokens (defaults to EnvConfig.NUM_OBS_TOKENS)
            max_steps: Maximum steps per episode
            inventory_item_names: List of inventory item names
            **kwargs: Additional configuration parameters

        Returns:
            Complete configuration dictionary for MettaGrid
        """
        if map is None:
            map = [
                [".", ".", "."],
                [".", "agent.player", "."],
                [".", ".", "."],
            ]

        # Default inventory items
        if inventory_item_names is None:
            inventory_item_names = [
                "ore_red",
                "ore_blue",
                "ore_green",
                "battery_red",
                "battery_blue",
                "battery_green",
                "heart",
                "armor",
                "laser",
                "blueprint",
            ]

        # Default actions configuration
        default_actions = {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        }

        # Override with provided actions
        if actions:
            for action_name, action_config in actions.items():
                if action_name in default_actions:
                    default_actions[action_name].update(action_config)
                else:
                    default_actions[action_name] = action_config

        config = {
            "num_agents": num_agents,
            "max_steps": max_steps,
            "episode_truncates": False,
            "obs_width": obs_width or DefaultEnvConfig.OBS_WIDTH,
            "obs_height": obs_height or DefaultEnvConfig.OBS_HEIGHT,
            "num_observation_tokens": num_observation_tokens or DefaultEnvConfig.NUM_OBS_TOKENS,
            "inventory_item_names": inventory_item_names,
            "no_agent_interference": no_agent_interference,
            "global_obs": {
                "episode_completion_pct": True,
                "last_action": True,
                "last_reward": True,
                "resource_rewards": False,
            },
            "track_movement_metrics": False,
            "recipe_details_obs": False,
            "actions": default_actions,
            "agent": {
                "default_resource_limit": 0,
                "resource_limits": {},
                "freeze_duration": 0,
                "rewards": {
                    "inventory": {},
                    "stats": {},
                },
                "action_failure_penalty": 0,
            },
            "groups": {
                "player": {
                    "id": 0,
                    "sprite": None,
                    "group_reward_pct": 0,
                    "props": {},
                },
                "red": {
                    "id": 0,
                    "sprite": None,
                    "group_reward_pct": 0,
                    "props": {},
                },
                "enemy": {
                    "id": 1,
                    "sprite": None,
                    "group_reward_pct": 0,
                    "props": {},
                },
            },
            "objects": {
                "wall": {"type_id": 1, "swappable": False},
                "altar": {
                    "type_id": 2,
                    "input_resources": {},
                    "output_resources": {},
                    "max_output": -1,
                    "max_conversions": -1,
                    "conversion_ticks": 0,
                    "cooldown": 0,
                    "initial_resource_count": 0,
                    "color": 0,
                },
            },
        }

        # Apply any additional kwargs (except map which is handled separately)
        for k, v in kwargs.items():
            if k != "map":
                config[k] = v

        # Store map separately for MettaGrid constructor
        config["map"] = map

        return config

    @staticmethod
    def create_environment(
        game_map: Optional[np.ndarray | list[list[str]]] = None,
        max_steps: int = 1000,
        num_agents: Optional[int] = None,
        actions: Optional[dict[str, dict[str, Any]]] = None,
        **kwargs,
    ) -> MettaGrid:
        """Create a MettaGrid environment from a game map or configuration.

        Args:
            game_map: Either a numpy array or list of lists representing the map
            max_steps: Maximum steps per episode
            num_agents: Number of agents (defaults to EnvConfig.NUM_AGENTS)
            actions: Action configuration overrides
            **kwargs: Additional configuration parameters

        Returns:
            Configured MettaGrid environment
        """
        if num_agents is None:
            num_agents = DefaultEnvConfig.NUM_AGENTS

        # Convert numpy array to list if needed
        if isinstance(game_map, np.ndarray):
            map_list = game_map.tolist()
        else:
            map_list = game_map

        # Create configuration
        game_config = TestEnvironmentBuilder.make_test_config(
            num_agents=num_agents, map=map_list, max_steps=max_steps, actions=actions, **kwargs
        )

        # Extract map from config (it's stored separately)
        map_data = game_config.pop("map")

        return MettaGrid(from_mettagrid_config(game_config), map_data, 42)

    @staticmethod
    def create_simple_environment(
        width: int = 8, height: int = 4, agent_positions: Optional[List[Tuple[int, int]]] = None, **kwargs
    ) -> MettaGrid:
        """Create a simple environment with walls and agents.

        Args:
            width: Grid width
            height: Grid height
            agent_positions: List of (y, x) positions for agents
            **kwargs: Additional configuration parameters

        Returns:
            Configured MettaGrid environment
        """
        # Create basic grid with walls
        game_map = TestEnvironmentBuilder.create_basic_grid(width, height)

        # Place agents if positions provided
        if agent_positions:
            game_map = TestEnvironmentBuilder.place_agents(game_map, agent_positions)
            num_agents = len(agent_positions)
        else:
            num_agents = kwargs.get("num_agents", DefaultEnvConfig.NUM_AGENTS)

        return TestEnvironmentBuilder.create_environment(game_map=game_map, num_agents=num_agents, **kwargs)
