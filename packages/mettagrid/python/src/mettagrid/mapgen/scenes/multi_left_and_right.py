from numpy import random

from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig
from mettagrid.mapgen.scenes.random import Random
from mettagrid.mapgen.scenes.room_grid import RoomGrid


class MultiLeftAndRightConfig(SceneConfig):
    rows: int
    columns: int
    assembler_ratio: float
    total_assemblers: int


class MultiLeftAndRight(Scene[MultiLeftAndRightConfig]):
    """
    Produce multiple left-or-right maps in a grid, with agents assigned randomly
    to teams, and rooms all identical otherwise. assemblers are placed asymmetrically
    with configurable total count and ratio between sides. The side with more assemblers
    is randomly determined at the start of each episode.
    """

    def get_children(self):
        # Pregenerate seeds so that we could make rooms deterministic.
        agent_seed = random.randint(0, int(1e9))
        assembler_seed = random.randint(0, int(1e9))
        assembler_distribution_seed = random.randint(0, int(1e9))

        # Calculate assembler counts based on ratio
        more_assemblers = int(self.config.total_assemblers * self.config.assembler_ratio)
        less_assemblers = self.config.total_assemblers - more_assemblers

        # Randomly determine which side gets more assemblers
        random.seed(assembler_distribution_seed)
        left_assemblers = more_assemblers if random.random() < 0.5 else less_assemblers
        right_assemblers = self.config.total_assemblers - left_assemblers

        agent_groups = [
            "team_1",
            "team_2",
        ]

        rows = self.config.rows
        columns = self.config.columns

        return [
            ChildrenAction(
                where="full",
                scene=RoomGrid.Config(
                    rows=rows,
                    columns=columns,
                    border_width=6,
                    children=[
                        ChildrenAction(
                            scene=RoomGrid.Config(
                                border_width=0,
                                layout=[
                                    [
                                        "maybe_assemblers_left",
                                        "empty",
                                        "empty",
                                        "agents",
                                        "empty",
                                        "empty",
                                        "maybe_assemblers_right",
                                    ],
                                ],
                                children=[
                                    ChildrenAction(
                                        scene=Random.Config(
                                            agents={
                                                agent_group: 1,
                                            },
                                            seed=agent_seed,
                                        ),
                                        where=AreaWhere(tags=["agents"]),
                                    ),
                                    ChildrenAction(
                                        scene=Random.Config(
                                            objects={"assembler": left_assemblers},
                                            seed=assembler_seed,
                                        ),
                                        where=AreaWhere(tags=["maybe_assemblers_left"]),
                                    ),
                                    ChildrenAction(
                                        scene=Random.Config(
                                            objects={"assembler": right_assemblers},
                                            seed=assembler_seed + 1,
                                        ),
                                        where=AreaWhere(tags=["maybe_assemblers_right"]),
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
        ]

    def render(self):
        pass
