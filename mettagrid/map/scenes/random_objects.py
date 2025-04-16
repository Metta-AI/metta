import numpy as np
from typing import Optional

from mettagrid.map.scenes.random import Random
from mettagrid.map.utils.random import MaybeSeed, sample_distribution
from ..scene import Scene, TypedChild
from ..node import Node
from omegaconf import DictConfig


class RandomObjects(Scene):
    """
    Fill the grid with random objects. Unlike Random, this scene takes the percentage ranges of objects, not the absolute count.

    It's rarely useful to pick the random number of agents, so this scene doesn't have that parameter.
    """

    def __init__(
        self,
        object_ranges: DictConfig | dict = {},
        seed: MaybeSeed = None,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._object_ranges = object_ranges

    def get_children(self, node: Node) -> list[TypedChild]:
        size = node.height * node.width
        objects = {}
        for obj_name, distribution in self._object_ranges.items():
            percentage = sample_distribution(distribution, self._rng)
            objects[obj_name] = int(size * percentage)

        return [
            {
                "scene": Random(objects=objects, seed=self._rng),
                "where": "full",
            }
        ]

    def _render(self, _):
        pass
