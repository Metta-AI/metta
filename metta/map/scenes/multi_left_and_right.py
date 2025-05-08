from numpy import random

from metta.map.scene import Scene
from metta.map.scenes.random import Random
from metta.map.scenes.room_grid import RoomGrid


class MultiLeftAndRight(Scene):
    """
    Produce multiple left-or-right maps in a grid, with agents assigned randomly
    to teams, and rooms all identical otherwise.
    """

    def __init__(self, rows: int, columns: int):
        # Pregenerate seeds so that we could make rooms deterministic.
        agent_seed = random.randint(0, int(1e9))
        altar_seed = random.randint(0, int(1e9))
        altar_side_seed = random.randint(0, int(1e9))

        agent_groups = [
            "team_1",
            "team_2",
        ]

        super().__init__(
            children=[
                {
                    "where": "full",
                    "scene": RoomGrid(
                        rows=rows,
                        columns=columns,
                        border_width=6,
                        children=[
                            {
                                # This scene is mostly identical to `left_or_right.yaml`.
                                # It adds seeds and place agents into groups.
                                "scene": RoomGrid(
                                    border_width=0,
                                    layout=[
                                        [
                                            "maybe_altars",
                                            "empty",
                                            "empty",
                                            "agents",
                                            "empty",
                                            "empty",
                                            "maybe_altars",
                                        ],
                                    ],
                                    children=[
                                        {
                                            "scene": lambda agent_group=agent_group: Random(
                                                # agents=1,
                                                agents={
                                                    agent_group: 1,
                                                },
                                                seed=agent_seed,
                                            ),
                                            "where": {"tags": ["agents"]},
                                        },
                                        {
                                            "scene": lambda: Random(
                                                objects={"altar": 2},
                                                seed=altar_seed,
                                            ),
                                            "where": {"tags": ["maybe_altars"]},
                                            "limit": 1,
                                            "order_by": "random",
                                            "order_by_seed": altar_side_seed,
                                        },
                                    ],
                                ),
                                "lock": "rooms",
                                # Place this scene in 1/len(agent_groups) of the rooms.
                                "limit": rows * columns // len(agent_groups),
                            }
                            for agent_group in agent_groups
                        ],
                    ),
                }
            ]
        )

    def _render(self, node):
        pass
