"""Simple game configuration for Claude-powered client.

Creates a minimal navigation-style game with a single agent and basic actions.
"""

import os
import sys

# Add python directory to path for protobuf imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from mettagrid.rpc.v1 import mettagrid_service_pb2 as pb


def create_simple_config(num_agents: int = 1) -> pb.GameConfig:
    """Create a simple game configuration with basic navigation."""

    # Define actions: noop, move, rotate
    actions = [
        pb.ActionDefinition(
            name="noop",
            type=pb.ActionDefinition.ACTION_NOOP,
            noop=pb.NoopActionConfig(),
        ),
        pb.ActionDefinition(
            name="move",
            type=pb.ActionDefinition.ACTION_MOVE,
            move=pb.MoveActionConfig(),
        ),
        pb.ActionDefinition(
            name="rotate",
            type=pb.ActionDefinition.ACTION_ROTATE,
            rotate=pb.RotateActionConfig(),
        ),
    ]

    # Define a simple agent type
    agent_config = pb.ObjectDefinition(
        name="agent",
        agent=pb.AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="team_a",
            action_failure_penalty=-0.01,
            freeze_duration=0,
        ),
    )

    # Define a wall type
    wall_config = pb.ObjectDefinition(
        name="wall",
        wall=pb.WallConfig(
            type_id=2,
            type_name="wall",
            swappable=False,
        ),
    )

    return pb.GameConfig(
        num_agents=num_agents,
        max_steps=100,
        episode_truncates=True,
        obs_width=5,
        obs_height=5,
        num_observation_tokens=50,
        global_obs=pb.GlobalObsConfig(
            episode_completion_pct=True,
            last_action=True,
            last_reward=True,
            visitation_counts=False,
        ),
        actions=actions,
        objects=[agent_config, wall_config],
        resource_loss_prob=0.0,
        track_movement_metrics=False,
        recipe_details_obs=False,
        allow_diagonals=False,
        inventory_regen_interval=0,
    )


def create_simple_map(
    width: int = 10, height: int = 10, num_agents: int = 1
) -> pb.MapDefinition:
    """Create a simple map with agents and walls.

    Layout:
    - Walls around the border
    - Agents placed in the center
    - Open space for movement
    """
    cells = []

    # Add border walls
    for row in range(height):
        for col in range(width):
            # Border walls
            if row == 0 or row == height - 1 or col == 0 or col == width - 1:
                cells.append(
                    pb.MapCell(
                        row=row,
                        col=col,
                        object_type="wall",
                    )
                )

    # Add agents in the center area
    center_row = height // 2
    center_col = width // 2

    for i in range(num_agents):
        # Place agents in a line horizontally from center
        agent_col = center_col - num_agents // 2 + i
        cells.append(
            pb.MapCell(
                row=center_row,
                col=agent_col,
                object_type="agent",
            )
        )

    return pb.MapDefinition(
        height=height,
        width=width,
        cells=cells,
    )
