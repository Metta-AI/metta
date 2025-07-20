from numpy import random

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.scenes.random import Random
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import AreaWhere, ChildrenAction


class MultiLeftAndRightParams(Config):
    rows: int
    columns: int
    altar_ratio: float
    total_altars: int


class MultiLeftAndRight(Scene[MultiLeftAndRightParams]):
    """
    Produce multiple left-or-right maps in a grid, with agents assigned randomly
    to teams, and rooms all identical otherwise. Altars are placed asymmetrically
    with configurable total count and ratio between sides. The side with more altars
    is randomly determined at the start of each episode.
    """

    def get_children(self):
        # Pregenerate seeds so that we could make rooms deterministic.
        agent_seed = random.randint(0, int(1e9))
        altar_seed = random.randint(0, int(1e9))
        altar_distribution_seed = random.randint(0, int(1e9))

        # Calculate altar counts based on ratio
        more_altars = int(self.params.total_altars * self.params.altar_ratio)
        less_altars = self.params.total_altars - more_altars

        # Randomly determine which side gets more altars
        random.seed(altar_distribution_seed)
        left_altars = more_altars if random.random() < 0.5 else less_altars
        right_altars = self.params.total_altars - left_altars

        agent_groups = [
            "team_1",
            "team_2",
        ]

        rows = self.params.rows
        columns = self.params.columns

        return [
            ChildrenAction(
                where="full",
                scene=RoomGrid.factory(
                    params=dict(
                        rows=rows,
                        columns=columns,
                        border_width=6,
                    ),
                    children=[
                        ChildrenAction(
                            scene=RoomGrid.factory(
                                params=dict(
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
                                ),
                                children=[
                                    ChildrenAction(
                                        scene=Random.factory(
                                            params={
                                                "agents": {
                                                    agent_group: 1,
                                                }
                                            },
                                            seed=agent_seed,
                                        ),
                                        where=AreaWhere(tags=["agents"]),
                                    ),
                                    ChildrenAction(
                                        scene=Random.factory(
                                            params={
                                                "objects": {"altar": left_altars},
                                            },
                                            seed=altar_seed,
                                        ),
                                        where=AreaWhere(tags=["maybe_altars_left"]),
                                    ),
                                    ChildrenAction(
                                        scene=Random.factory(
                                            params={
                                                "objects": {"altar": right_altars},
                                            },
                                            seed=altar_seed + 1,
                                        ),
                                        where=AreaWhere(tags=["maybe_altars_right"]),
                                    ),
                                ],
                            ),
                            lock="rooms",
                            limit=rows * columns // len(agent_groups),
                        )
                        for agent_group in agent_groups
                    ],
                ),
            ),
            *self.children,
        ]

    def render(self):
        pass
