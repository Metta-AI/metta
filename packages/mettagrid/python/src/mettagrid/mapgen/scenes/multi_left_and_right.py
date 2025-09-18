from numpy import random

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.random import Random
from mettagrid.mapgen.scenes.room_grid import RoomGrid
from mettagrid.mapgen.types import AreaWhere


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
                    params=RoomGrid.Params(
                        rows=rows,
                        columns=columns,
                        border_width=6,
                    ),
                    children_actions=[
                        ChildrenAction(
                            scene=RoomGrid.factory(
                                params=RoomGrid.Params(
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
                                children_actions=[
                                    ChildrenAction(
                                        scene=Random.factory(
                                            params=Random.Params(
                                                agents={
                                                    agent_group: 1,
                                                }
                                            ),
                                            seed=agent_seed,
                                        ),
                                        where=AreaWhere(tags=["agents"]),
                                    ),
                                    ChildrenAction(
                                        scene=Random.factory(
                                            params=Random.Params(
                                                objects={"altar": left_altars},
                                            ),
                                            seed=altar_seed,
                                        ),
                                        where=AreaWhere(tags=["maybe_altars_left"]),
                                    ),
                                    ChildrenAction(
                                        scene=Random.factory(
                                            params=Random.Params(
                                                objects={"altar": right_altars},
                                            ),
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
            *self.children_actions,
        ]

    def render(self):
        pass
