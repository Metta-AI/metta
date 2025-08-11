"""Test utilities for mettagrid tests."""

from typing import Any, Optional

import numpy as np

from metta.mettagrid.level_builder import LevelMap
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
    EnvConfig,
    GameConfig,
    GroupConfig,
    WallConfig,
)
from metta.mettagrid.utils import make_level_map


def make_test_config(
    num_agents: int = 1,
    map: Optional[list[list[str]]] = None,
    actions: Optional[dict[str, dict[str, Any]]] = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a test configuration for MettaGrid.

    Args:
        num_agents: Number of agents in the environment
        map: 2D list of strings representing the map
        actions: Action configuration overrides
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
        "max_steps": 1000,
        "episode_truncates": False,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": [
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
        ],
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


class EnvConfigBuilder:
    """Builder class for creating test EnvConfig instances."""

    def __init__(self):
        self._num_agents = 1
        self._max_steps = 5
        self._obs_width = 11
        self._obs_height = 11
        self._num_observation_tokens = 200
        self._level_map = None
        self._actions_config = None
        self._agent_config = AgentConfig()
        self._groups = {"agent": GroupConfig(id=0)}
        self._objects = {"wall": WallConfig(type_id=1, swappable=False)}
        self._desync_episodes = True
        self._width = 5
        self._height = 5
        self._seed = 42

    def with_num_agents(self, num_agents: int):
        self._num_agents = num_agents
        return self

    def with_max_steps(self, max_steps: int):
        self._max_steps = max_steps
        return self

    def with_obs_size(self, width: int, height: int, num_tokens: int | None = None):
        self._obs_width = width
        self._obs_height = height
        if num_tokens is not None:
            self._num_observation_tokens = num_tokens
        return self

    def with_map_size(self, width: int, height: int):
        self._width = width
        self._height = height
        return self

    def with_level_map(self, level_map):
        self._level_map = level_map
        return self

    def with_actions(self, **action_configs):
        if self._actions_config is None:
            self._actions_config = ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                rotate=ActionConfig(),
                put_items=ActionConfig(),
                get_items=ActionConfig(),
                attack=AttackActionConfig(),
                swap=ActionConfig(),
            )
        for action_name, config in action_configs.items():
            if hasattr(self._actions_config, action_name):
                setattr(self._actions_config, action_name, config)
        return self

    def with_seed(self, seed: int):
        self._seed = seed
        return self

    def build(self) -> EnvConfig:
        """Build the EnvConfig instance."""
        if self._level_map is None:
            self._level_map = make_level_map(
                width=self._width, height=self._height, num_agents=self._num_agents, border_width=1, seed=self._seed
            )

        if self._actions_config is None:
            self._actions_config = ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                rotate=ActionConfig(),
                put_items=ActionConfig(),
                get_items=ActionConfig(),
                attack=AttackActionConfig(),
                swap=ActionConfig(),
            )

        game_config = GameConfig(
            num_agents=self._num_agents,
            max_steps=self._max_steps,
            obs_width=self._obs_width,
            obs_height=self._obs_height,
            num_observation_tokens=self._num_observation_tokens,
            agent=self._agent_config,
            groups=self._groups,
            actions=self._actions_config,
            objects=self._objects,
            level_map=self._level_map,
        )

        return EnvConfig(game=game_config, desync_episodes=self._desync_episodes)


def env_cfg_builder() -> EnvConfigBuilder:
    """Create a new EnvConfigBuilder for test environment configuration.

    Returns:
        EnvConfigBuilder instance for fluent configuration.

    Example:
        env_config = env_cfg_builder().with_num_agents(2).with_max_steps(100).build()
    """
    return EnvConfigBuilder()


def make_test_level_map(width: int = 25, height: int = 25, num_agents: int = 1) -> LevelMap:
    """Create a simple test level map with walls and agents.

    Args:
        width: Width of the map
        height: Height of the map
        num_agents: Number of agents to place

    Returns:
        LevelMap with walls around perimeter and agents placed
    """
    grid = np.full((height, width), "empty", dtype="<U50")

    # Add walls around perimeter
    grid[0, :] = "wall"
    grid[-1, :] = "wall"
    grid[:, 0] = "wall"
    grid[:, -1] = "wall"

    # Place agents in valid positions
    agent_positions = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if len(agent_positions) < num_agents:
                agent_positions.append((y, x))
            else:
                break
        if len(agent_positions) >= num_agents:
            break

    for _i, (y, x) in enumerate(agent_positions):
        grid[y, x] = "agent.agent"

    return LevelMap(grid=grid, labels=["test"])
