# Root map generator, based on nodes.
import hydra
import numpy as np
from omegaconf import DictConfig

from .scene import Scene


class MapGen:
    def __init__(self, width: int, height: int, root: DictConfig, border_width: int = 1):
        self._width = width
        self._height = height
        self._border_width = border_width
        self._root_config = root

        if isinstance(self._root_config, DictConfig):
            # recursive_map_builder is set to false, so we need to instantiate the root config
            self._root = hydra.utils.instantiate(self._root_config, _recursive_=False)
        elif isinstance(self._root_config, Scene):
            self._root = self._root_config
        else:
            raise ValueError("Root config must be a DictConfig or a Node")

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
