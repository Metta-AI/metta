from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.bsp import BSP, BSPParams
from mettagrid.mapgen.scenes.maze import Maze, MazeParams
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate, RandomSceneParams
from mettagrid.mapgen.scenes.varied_terrain import VariedTerrain, VariedTerrainParams


class QuadrantLayoutParams(Config):
    weight_bsp10: float = 1.0
    weight_bsp8: float = 1.0
    weight_maze: float = 2.0
    weight_terrain_balanced: float = 2.0
    weight_terrain_maze: float = 2.0


class QuadrantLayout(Scene[QuadrantLayoutParams]):
    """
    For each quadrant tag, randomly pick one of several layout scenes (maze/BSP variants).
    """

    def _candidates(self):
        p = self.params
        return [
            RandomSceneCandidate(
                scene=BSP.factory(
                    BSPParams(rooms=10, min_room_size=4, min_room_size_ratio=0.2, max_room_size_ratio=0.55)
                ),
                weight=p.weight_bsp10,
            ),
            RandomSceneCandidate(
                scene=BSP.factory(
                    BSPParams(rooms=8, min_room_size=5, min_room_size_ratio=0.25, max_room_size_ratio=0.5)
                ),
                weight=p.weight_bsp8,
            ),
            # Use terrain styles that always render structure
            RandomSceneCandidate(
                scene=VariedTerrain.factory(
                    VariedTerrainParams(
                        objects={"wall": 0},
                        agents=0,
                        style="maze",
                        labyrinth_scatter_symbol=None,
                        labyrinth_scatter_probability=0.0,
                    )
                ),
                weight=p.weight_terrain_maze,
            ),
            RandomSceneCandidate(
                scene=Maze.factory(MazeParams(algorithm="dfs")),
                weight=p.weight_maze,
            ),
        ]

    def get_children(self):
        # Fill this quadrant area with one randomly chosen layout
        candidates = self._candidates()
        return [
            ChildrenAction(
                scene=RandomScene.factory(RandomSceneParams(candidates=candidates)),
                where="full",
                limit=1,
                lock="quad",
                order_by="first",
            )
        ]

    def render(self):
        pass
