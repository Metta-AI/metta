"""Compact configuration builders for MettaGrid demos.

This module provides simple, compact ways to create common MettaGrid configurations,
similar to the old config/builder.py approach but using the new Pydantic models.
"""

from metta.mettagrid.config import objects
from metta.mettagrid.config.mettagrid_config import (
    EnvConfig,
    PyActionConfig,
    PyActionsConfig,
    PyAgentConfig,
    PyAgentRewards,
    PyAttackActionConfig,
    PyChangeGlyphActionConfig,
    PyGameConfig,
    PyGroupConfig,
    PyInventoryRewards,
)
from metta.mettagrid.room.random import Random as RandomRoomBuilder


def arena(
    num_agents: int = 2,
    combat: bool = False,
    max_steps: int = 500,
    map_width: int = 20,
    map_height: int = 20,
    obs_width: int = 5,
    obs_height: int = 5,
) -> EnvConfig:
    """Create an arena-style environment configuration.

    Args:
        num_agents: Number of agents
        combat: Whether to include combat objects and actions
        max_steps: Maximum steps per episode
        map_width: Map width
        map_height: Map height
        obs_width: Observation width
        obs_height: Observation height

    Returns:
        Complete EnvConfig ready for use
    """
    # Basic objects for resource collection
    game_objects = {
        "wall": objects.wall,
        "altar": objects.altar,
        "mine_red": objects.mine_red,
        "generator_red": objects.generator_red,
    }

    # Basic actions
    actions = PyActionsConfig(
        noop=PyActionConfig(enabled=True),
        move=PyActionConfig(enabled=True),
        rotate=PyActionConfig(enabled=True),
        put_items=PyActionConfig(enabled=True),
        get_items=PyActionConfig(enabled=True),
        swap=PyActionConfig(enabled=True),
        change_color=PyActionConfig(enabled=False),
        change_glyph=PyChangeGlyphActionConfig(enabled=False, number_of_glyphs=0),
    )

    # Add combat if requested
    if combat:
        game_objects["lasery"] = objects.lasery
        game_objects["armory"] = objects.armory

        actions.attack = PyAttackActionConfig(
            enabled=True, required_resources={"laser": 1}, defense_resources={"armor": 1}
        )

    # Set up inventory items
    inventory_items = ["heart", "ore_red", "battery_red"]
    if combat:
        inventory_items.extend(["laser", "armor"])

    # Create rewards
    inventory_rewards = PyInventoryRewards(heart=1.0)
    agent_rewards = PyAgentRewards(inventory=inventory_rewards)

    # Create agent config
    agent_config = PyAgentConfig(
        default_resource_limit=50,
        resource_limits={"heart": 255},
        freeze_duration=0,
        rewards=agent_rewards,
        action_failure_penalty=0.0,
    )

    # Create group config
    group_config = PyGroupConfig(id=0, sprite=0, props=agent_config)

    # Create game config
    game_config = PyGameConfig(
        max_steps=max_steps,
        num_agents=num_agents,
        obs_width=obs_width,
        obs_height=obs_height,
        num_observation_tokens=obs_width * obs_height,
        inventory_item_names=inventory_items,
        groups={"agent": group_config},
        agent=agent_config,
        actions=actions,
        objects=game_objects,
    )

    # Create map with appropriate objects
    map_objects = {"mine_red": 3, "generator_red": 2, "altar": 2}
    if combat:
        map_objects.update({"lasery": 1, "armory": 1})

    map_builder = RandomRoomBuilder(
        agents=num_agents, width=map_width, height=map_height, border_width=2, objects=map_objects
    )

    level_map = map_builder.build()

    return EnvConfig(game=game_config, level_map=level_map)


def simple_arena(num_agents: int = 1) -> EnvConfig:
    """Create a simple single-agent arena for quick testing."""
    return arena(
        num_agents=num_agents, combat=False, max_steps=100, map_width=12, map_height=12, obs_width=5, obs_height=5
    )


def combat_arena(num_agents: int = 2) -> EnvConfig:
    """Create a combat arena with weapons and armor."""
    return arena(
        num_agents=num_agents, combat=True, max_steps=200, map_width=16, map_height=16, obs_width=7, obs_height=7
    )


def empty_arena(num_agents: int = 2, map_width: int = 10, map_height: int = 10, max_steps: int = 50) -> EnvConfig:
    """Create an empty arena for basic multi-agent interaction."""
    # Minimal objects
    game_objects = {
        "wall": objects.wall,
    }

    actions = PyActionsConfig(
        noop=PyActionConfig(enabled=True),
        move=PyActionConfig(enabled=True),
        rotate=PyActionConfig(enabled=True),
        put_items=PyActionConfig(enabled=True),
        get_items=PyActionConfig(enabled=True),
        attack=PyAttackActionConfig(enabled=True),
        swap=PyActionConfig(enabled=True),
        change_color=PyActionConfig(enabled=False),
        change_glyph=PyChangeGlyphActionConfig(enabled=False, number_of_glyphs=0),
    )

    # Simple rewards
    inventory_rewards = PyInventoryRewards(heart=1.0)
    agent_rewards = PyAgentRewards(inventory=inventory_rewards)

    agent_config = PyAgentConfig(
        default_resource_limit=5,
        resource_limits={"heart": 255},
        freeze_duration=0,
        rewards=agent_rewards,
        action_failure_penalty=0.0,
    )

    group_config = PyGroupConfig(id=0, sprite=0, props=agent_config)

    game_config = PyGameConfig(
        max_steps=max_steps,
        num_agents=num_agents,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=9,
        inventory_item_names=["heart"],
        groups={"agent": group_config},
        agent=agent_config,
        actions=actions,
        objects=game_objects,
    )

    # Empty map
    map_builder = RandomRoomBuilder(agents=num_agents, width=map_width, height=map_height, border_width=1, objects={})

    level_map = map_builder.build()

    return EnvConfig(game=game_config, level_map=level_map)
