"""
BenchMARL wrapper utilities for MettaGrid environments.

This module provides helper functions to create pre-configured MettaGrid tasks
for use with BenchMARL benchmarking, following the same pattern as other
environment wrapper utilities.
"""

from typing import Any, Optional

from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AssemblerConfig,
    AttackActionConfig,
    ConverterConfig,
    GameConfig,
    MettaGridConfig,
    RecipeConfig,
    WallConfig,
)
from mettagrid.envs.benchmarl_env import MettaGridTask
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder


def create_navigation_task(
    num_agents: int = 1,
    max_steps: int = 1000,
    obs_width: int = 11,
    obs_height: int = 11,
    **kwargs: Any,
) -> MettaGridTask:
    """
    Create a navigation task for BenchMARL.

    Args:
        num_agents: Number of agents
        max_steps: Maximum steps per episode
        obs_width: Width of observation window
        obs_height: Height of observation window
        **kwargs: Additional arguments passed to MettaGridTask

    Returns:
        BenchMARL task for navigation
    """
    # Create dynamic navigation map based on number of agents
    if num_agents == 1:
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "1", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ]
    elif num_agents == 2:
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", "1", ".", "2", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ]
    else:  # 3 or more agents
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "1", ".", "2", ".", "3", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", ".", "#", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ]
        # For more than 3 agents, add additional positions
        if num_agents > 3:
            for i in range(4, num_agents + 1):
                # Add agent positions to empty spots
                for row_idx, row in enumerate(map_data):
                    for col_idx, cell in enumerate(row):
                        if cell == "." and str(i) not in [cell for row in map_data for cell in row]:
                            map_data[row_idx][col_idx] = str(i)
                            break
                    else:
                        continue
                    break

    # Create agents with proper configuration (team_id should match map positions)
    agents = []
    for i in range(num_agents):
        agents.append(
            AgentConfig(
                team_id=i + 1,  # Map has agents 1, 2, 3 not 0, 1, 2
            )
        )

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            agents=agents,
            actions=ActionsConfig(
                move=ActionConfig(enabled=True),
                rotate=ActionConfig(enabled=True),
                noop=ActionConfig(enabled=False),
            ),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=AsciiMapBuilder.Config(map_data=map_data),
        )
    )

    return MettaGridTask(
        mg_config=config,
        task_name="mettagrid_navigation",
        max_steps=max_steps,
        **kwargs,
    )


def create_cooperative_task(
    map_name: Optional[str] = None,
    num_agents: int = 4,
    max_steps: int = 1000,
    obs_width: int = 11,
    obs_height: int = 11,
    **kwargs: Any,
) -> MettaGridTask:
    """
    Create a cooperative task for BenchMARL.

    Args:
        map_name: Optional map file name. If None, creates a procedural map
        num_agents: Number of agents
        max_steps: Maximum steps per episode
        obs_width: Width of observation window
        obs_height: Height of observation window
        **kwargs: Additional arguments passed to MettaGridTask

    Returns:
        BenchMARL task for cooperation
    """
    # Create agents with team assignments (all on same team for cooperation)
    agents = [AgentConfig(team_id=0) for _ in range(num_agents)]

    if map_name:
        map_builder = AsciiMapBuilder.Config(map_name=map_name)
        task_name = f"mettagrid_cooperative_{map_name.replace('/', '_')}"
    else:
        # Create procedural map with resources and converters
        map_builder = RandomMapBuilder.Config(
            agents=num_agents,
            width=20,
            height=20,
            border_width=1,
            border_object="wall",
        )
        task_name = "mettagrid_cooperative_procedural"

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            agents=agents,
            actions=ActionsConfig(
                move=ActionConfig(enabled=True),
                rotate=ActionConfig(enabled=True),
                put_items=ActionConfig(enabled=True),
                get_items=ActionConfig(enabled=True),
                noop=ActionConfig(enabled=False),
            ),
            objects={
                "wall": WallConfig(type_id=1, swappable=False),
                "converter": ConverterConfig(
                    type_id=2,
                    input_resources={"ore_red": 1},
                    output_resources={"battery_red": 1},
                    max_conversions=10,
                    cooldown=5,
                ),
            },
            map_builder=map_builder,
        )
    )

    return MettaGridTask(
        mg_config=config,
        task_name=task_name,
        max_steps=max_steps,
        **kwargs,
    )


def create_competitive_task(
    num_agents: int = 4,
    max_steps: int = 1000,
    map_width: int = 20,
    map_height: int = 20,
    obs_width: int = 11,
    obs_height: int = 11,
    **kwargs: Any,
) -> MettaGridTask:
    """
    Create a competitive task for BenchMARL.

    Args:
        num_agents: Number of agents (should be even for balanced teams)
        max_steps: Maximum steps per episode
        map_width: Width of the map
        map_height: Height of the map
        obs_width: Width of observation window
        obs_height: Height of observation window
        **kwargs: Additional arguments passed to MettaGridTask

    Returns:
        BenchMARL task for competition
    """
    # Create agents with team assignments (split into two teams)
    agents = []
    for i in range(num_agents):
        team_id = i % 2  # Alternate between team 0 and team 1
        agents.append(
            AgentConfig(
                team_id=team_id,
                rewards={
                    "stats": {
                        "action.attack.agent": 1.0,  # Reward for attacking opposing agents
                        "inventory.heart.gained": 0.5,  # Reward for collecting health
                    }
                },
            )
        )

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            agents=agents,
            actions=ActionsConfig(
                move=ActionConfig(enabled=True),
                rotate=ActionConfig(enabled=True),
                attack=AttackActionConfig(enabled=True),
                get_items=ActionConfig(enabled=True),
                noop=ActionConfig(enabled=False),
            ),
            objects={
                "wall": WallConfig(type_id=1, swappable=False),
            },
            map_builder=RandomMapBuilder.Config(
                agents=num_agents,
                width=map_width,
                height=map_height,
                border_width=1,
                border_object="wall",
            ),
        )
    )

    return MettaGridTask(
        mg_config=config,
        task_name="mettagrid_competitive",
        max_steps=max_steps,
        **kwargs,
    )


def create_mixed_task(
    num_agents: int = 6,
    max_steps: int = 2000,
    map_width: int = 30,
    map_height: int = 30,
    obs_width: int = 13,
    obs_height: int = 13,
    **kwargs: Any,
) -> MettaGridTask:
    """
    Create a mixed cooperation/competition task for BenchMARL.

    This task involves teams that must cooperate within their group
    while competing against other teams for resources.

    Args:
        num_agents: Number of agents (should be divisible by 3 for 3 teams)
        max_steps: Maximum steps per episode
        map_width: Width of the map
        map_height: Height of the map
        obs_width: Width of observation window
        obs_height: Height of observation window
        **kwargs: Additional arguments passed to MettaGridTask

    Returns:
        BenchMARL task for mixed cooperation/competition
    """
    # Create agents with team assignments (3 teams)
    agents = []
    for i in range(num_agents):
        team_id = i % 3  # Distribute into 3 teams
        agents.append(
            AgentConfig(
                team_id=team_id,
                rewards={
                    "inventory": {
                        "battery_red": 2.0,
                        "battery_blue": 2.0,
                        "battery_green": 2.0,
                    },
                    "stats": {
                        "action.attack.agent": 0.5,  # Small reward for attacking other teams
                        "action.put_items.converter": 1.0,  # Reward for using converters
                    },
                },
            )
        )

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            agents=agents,
            actions=ActionsConfig(
                move=ActionConfig(enabled=True),
                rotate=ActionConfig(enabled=True),
                attack=AttackActionConfig(enabled=True),
                put_items=ActionConfig(enabled=True),
                get_items=ActionConfig(enabled=True),
                swap=ActionConfig(enabled=True),
                noop=ActionConfig(enabled=False),
            ),
            objects={
                "wall": WallConfig(type_id=1, swappable=False),
                "converter_red": ConverterConfig(
                    type_id=2,
                    input_resources={"ore_red": 2},
                    output_resources={"battery_red": 1},
                    max_conversions=20,
                    cooldown=5,
                    color=1,
                ),
                "converter_blue": ConverterConfig(
                    type_id=3,
                    input_resources={"ore_blue": 2},
                    output_resources={"battery_blue": 1},
                    max_conversions=20,
                    cooldown=5,
                    color=2,
                ),
                "converter_green": ConverterConfig(
                    type_id=4,
                    input_resources={"ore_green": 2},
                    output_resources={"battery_green": 1},
                    max_conversions=20,
                    cooldown=5,
                    color=3,
                ),
                "assembler": AssemblerConfig(
                    name="battery_assembler",
                    type_id=5,
                    recipes=[
                        (
                            ["Any"],
                            RecipeConfig(
                                input_resources={
                                    "battery_red": 1,
                                    "battery_blue": 1,
                                    "battery_green": 1,
                                },
                                output_resources={"blueprint": 1},
                                cooldown=10,
                            ),
                        )
                    ],
                ),
            },
            map_builder=RandomMapBuilder.Config(
                agents=num_agents,
                width=map_width,
                height=map_height,
                border_width=1,
                border_object="wall",
            ),
        )
    )

    return MettaGridTask(
        mg_config=config,
        task_name="mettagrid_mixed",
        max_steps=max_steps,
        **kwargs,
    )
