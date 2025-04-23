import numpy as np
import numpy.typing as npt

from mettagrid.config.room.room import Room

from .scene import SceneCfg, make_scene

MapGrid = npt.NDArray[np.str_]


# Root map generator, based on nodes.
class MapGen(Room):
    _grid: MapGrid

    def __init__(self, width: int, height: int, root: SceneCfg, border_width: int = 1):
        super().__init__()
        self._width = width
        self._height = height
        self._border_width = border_width
        self._root_config = root

        self._root = make_scene(self._root_config)

        self._grid = np.full((height + 2 * border_width, width + 2 * border_width), "empty", dtype="<U50")
        self._grid[:border_width, :] = "wall"
        self._grid[-border_width:, :] = "wall"
        self._grid[:, :border_width] = "wall"
        self._grid[:, -border_width:] = "wall"

    def _inner_grid(self):
        if self._border_width > 0:
            return self._grid[
                self._border_width : -self._border_width,
                self._border_width : -self._border_width,
            ]
        else:
            return self._grid

    def build(self):
        root_node = self._root.make_node(self._inner_grid())

        root_node.render()
        return self._grid
