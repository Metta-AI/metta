from numpy import random

from metta.map.scene import Scene
from metta.map.scenes.random import Random
from metta.map.scenes.room_grid import RoomGrid


class MultiLeftAndRight(Scene):
    """
    Produce multiple left-or-right maps in a grid, with agents assigned randomly
    to teams, and rooms all identical otherwise. Altars are placed asymmetrically
    with configurable total count and ratio between sides. The side with more altars
    is randomly determined at the start of each episode.
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        total_altars: int = 4,
        altar_ratio: float = 0.75,  # Ratio of altars that go to the side with more
    ):
        # Pregenerate seeds so that we could make rooms deterministic.
        agent_seed = random.randint(0, int(1e9))
        altar_seed = random.randint(0, int(1e9))
        altar_side_seed = random.randint(0, int(1e9))
        altar_distribution_seed = random.randint(0, int(1e9))

        # Calculate altar counts based on ratio
        more_altars = int(total_altars * altar_ratio)
        less_altars = total_altars - more_altars

        # Randomly determine which side gets more altars
        random.seed(altar_distribution_seed)
        left_altars = more_altars if random.random() < 0.5 else less_altars
        right_altars = total_altars - left_altars

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
                                "scene": RoomGrid(
                                    border_width=0,
                                    layout=[
                                        [
                                            "maybe_altars_left",
                                            "empty",
                                            "empty",
                                            "agents",
                                            "empty",
                                            "empty",
                                            "maybe_altars_right",
                                        ],
                                    ],
                                    children=[
                                        {
                                            "scene": lambda agent_group=agent_group: Random(
                                                agents={
                                                    agent_group: 1,
                                                },
                                                seed=agent_seed,
                                            ),
                                            "where": {"tags": ["agents"]},
                                        },
                                        {
                                            "scene": lambda: Random(
                                                objects={"altar": left_altars},
                                                seed=altar_seed,
                                            ),
                                            "where": {"tags": ["maybe_altars_left"]},
                                            "limit": 1,
                                            "order_by": "random",
                                            "order_by_seed": altar_side_seed,
                                        },
                                        {
                                            "scene": lambda: Random(
                                                objects={"altar": right_altars},
                                                seed=altar_seed + 1,
                                            ),
                                            "where": {"tags": ["maybe_altars_right"]},
                                            "limit": 1,
                                            "order_by": "random",
                                            "order_by_seed": altar_side_seed + 1,
                                        },
                                    ],
                                ),
                                "lock": "rooms",
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
