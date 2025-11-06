import numpy

import mettagrid.mapgen.area
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.random
import mettagrid.mapgen.scenes.room_grid


class MultiLeftAndRightConfig(mettagrid.mapgen.scene.SceneConfig):
    rows: int
    columns: int
    altar_ratio: float
    total_altars: int


class MultiLeftAndRight(mettagrid.mapgen.scene.Scene[MultiLeftAndRightConfig]):
    """
    Produce multiple left-or-right maps in a grid, with agents assigned randomly
    to teams, and rooms all identical otherwise. Altars are placed asymmetrically
    with configurable total count and ratio between sides. The side with more altars
    is randomly determined at the start of each episode.
    """

    def get_children(self):
        # Pregenerate seeds so that we could make rooms deterministic.
        agent_seed = numpy.random.randint(0, int(1e9))
        altar_seed = numpy.random.randint(0, int(1e9))
        altar_distribution_seed = numpy.random.randint(0, int(1e9))

        # Calculate altar counts based on ratio
        more_altars = int(self.config.total_altars * self.config.altar_ratio)
        less_altars = self.config.total_altars - more_altars

        # Randomly determine which side gets more altars
        numpy.random.seed(altar_distribution_seed)
        left_altars = more_altars if numpy.random.random() < 0.5 else less_altars
        right_altars = self.config.total_altars - left_altars

        agent_groups = [
            "team_1",
            "team_2",
        ]

        rows = self.config.rows
        columns = self.config.columns

        return [
            mettagrid.mapgen.scene.ChildrenAction(
                where="full",
                scene=mettagrid.mapgen.scenes.room_grid.RoomGrid.Config(
                    rows=rows,
                    columns=columns,
                    border_width=6,
                    children=[
                        mettagrid.mapgen.scene.ChildrenAction(
                            scene=mettagrid.mapgen.scenes.room_grid.RoomGrid.Config(
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
                                    mettagrid.mapgen.scene.ChildrenAction(
                                        scene=mettagrid.mapgen.scenes.random.Random.Config(
                                            agents={
                                                agent_group: 1,
                                            },
                                            seed=agent_seed,
                                        ),
                                        where=mettagrid.mapgen.area.AreaWhere(tags=["agents"]),
                                    ),
                                    mettagrid.mapgen.scene.ChildrenAction(
                                        scene=mettagrid.mapgen.scenes.random.Random.Config(
                                            objects={"altar": left_altars},
                                            seed=altar_seed,
                                        ),
                                        where=mettagrid.mapgen.area.AreaWhere(tags=["maybe_altars_left"]),
                                    ),
                                    mettagrid.mapgen.scene.ChildrenAction(
                                        scene=mettagrid.mapgen.scenes.random.Random.Config(
                                            objects={"altar": right_altars},
                                            seed=altar_seed + 1,
                                        ),
                                        where=mettagrid.mapgen.area.AreaWhere(tags=["maybe_altars_right"]),
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
