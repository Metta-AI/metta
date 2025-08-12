"""Test utilities and environment builder for mettagrid tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from metta.mettagrid.config import object as obj
from metta.mettagrid.map_builder.random import RandomMapBuilderConfig
from metta.mettagrid.map_builder.utils import create_grid
from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate, dtype_actions  # noqa: F401
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    ConverterConfig,
    EnvConfig,
    GameConfig,
    GroupConfig,
    InventoryRewards,
)


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

    @staticmethod
    def benchmark_env_config():
        """Create a benchmark environment configuration matching benchmark.yaml."""
        # Map builder configuration
        map_builder = RandomMapBuilderConfig(
            width=50,
            height=50,
            border_width=1,
            seed=42,
            agents=6,
            objects={
                "mine_red": 10,
                "generator_red": 2,
                "altar": 1,
                "armory": 1,
                "lasery": 1,
                "lab": 1,
                "factory": 1,
                "temple": 1,
                "block": 20,
                "wall": 20,
            },
        )

        # Agent configuration
        agent_config = AgentConfig(
            default_resource_limit=50,
            freeze_duration=10,
            rewards=AgentRewards(
                inventory=InventoryRewards(
                    ore_red=0.005,
                    ore_blue=0.005,
                    ore_green=0.005,
                    battery_red=0.01,
                    battery_blue=0.01,
                    battery_green=0.01,
                    battery_red_max=5,
                    battery_blue_max=5,
                    battery_green_max=5,
                    heart=1,
                ),
            ),
        )

        # Groups configuration
        groups = {
            "agent": GroupConfig(id=0, sprite=0, props={}),
            "team_1": GroupConfig(id=1, sprite=1, group_reward_pct=0.5, props={}),
            "team_2": GroupConfig(id=2, sprite=4, group_reward_pct=0.5, props={}),
            "team_3": GroupConfig(id=3, sprite=8, group_reward_pct=0.5, props={}),
            "team_4": GroupConfig(id=4, sprite=1, group_reward_pct=0.5, props={}),
            "prey": GroupConfig(id=5, sprite=12, props={}),
            "predator": GroupConfig(id=6, sprite=6, props={}),
        }

        # Objects configuration - use pre-defined objects where possible and extend with missing fields
        # Common converter defaults for benchmark config
        converter_defaults = {
            "max_output": 5,
            "conversion_ticks": 1,
            "initial_resource_count": 1,
        }

        objects = {
            # Use pre-defined wall and block objects directly
            "wall": obj.wall,
            "block": obj.block,
            # Use pre-defined altar with additional fields
            "altar": obj.altar.model_copy(update={**converter_defaults}),
            # Use pre-defined mines with additional fields and colors
            "mine_red": obj.mine_red.model_copy(update={**converter_defaults, "color": 0}),
            "mine_blue": obj.mine_blue.model_copy(update={**converter_defaults, "color": 1}),
            "mine_green": obj.mine_green.model_copy(update={**converter_defaults, "color": 2}),
            # Use pre-defined generators with additional fields and colors
            "generator_red": obj.generator_red.model_copy(update={**converter_defaults, "color": 0}),
            "generator_blue": obj.generator_blue.model_copy(update={**converter_defaults, "color": 1}),
            "generator_green": obj.generator_green.model_copy(update={**converter_defaults, "color": 2}),
            # Armory needs custom type_id (9 instead of 16 from object.py)
            "armory": ConverterConfig(
                type_id=9,
                input_resources={"ore_red": 3},
                output_resources={"armor": 1},
                cooldown=10,
                **converter_defaults,
            ),
            # Lasery needs custom type_id (10 instead of 15) and different input_resources
            "lasery": ConverterConfig(
                type_id=10,
                input_resources={"ore_red": 1, "battery_red": 2},
                output_resources={"laser": 1},
                cooldown=10,
                **converter_defaults,
            ),
            # Lab, factory, and temple are not in object.py, so define them here
            "lab": ConverterConfig(
                type_id=11,
                input_resources={"ore_red": 3, "battery_red": 3},
                output_resources={"blueprint": 1},
                cooldown=5,
                **converter_defaults,
            ),
            "factory": ConverterConfig(
                type_id=12,
                input_resources={"blueprint": 1, "ore_red": 5, "battery_red": 5},
                output_resources={"armor": 5, "laser": 5},
                cooldown=5,
                **converter_defaults,
            ),
            "temple": ConverterConfig(
                type_id=13,
                input_resources={"heart": 1, "blueprint": 1},
                output_resources={"heart": 5},
                cooldown=5,
                **converter_defaults,
            ),
        }

        # Actions configuration
        actions = ActionsConfig(
            noop=ActionConfig(enabled=True),
            move=ActionConfig(enabled=True),
            rotate=ActionConfig(enabled=True),
            put_items=ActionConfig(enabled=True),
            get_items=ActionConfig(enabled=True),
            attack=AttackActionConfig(
                enabled=True,
                consumed_resources={"laser": 1},
                defense_resources={"armor": 1},
            ),
            swap=ActionConfig(enabled=True),
            change_color=ActionConfig(enabled=True),
            change_glyph=ChangeGlyphActionConfig(enabled=True, number_of_glyphs=4),
        )

        # Inventory item names
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

        # Create game configuration
        game_config = GameConfig(
            num_agents=24,
            num_observation_tokens=100,
            inventory_item_names=inventory_item_names,
            map_builder=map_builder,
            obs_width=11,
            obs_height=11,
            max_steps=1000,
            agent=agent_config,
            groups=groups,
            objects=objects,
            actions=actions,
        )

        # Create environment configuration
        env_cfg = EnvConfig(game=game_config)
        return env_cfg
