from typing import Any, List, Optional, TypedDict

import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import OmegaConf

from .node import Node


class TypedChild(TypedDict):
    scene: "Scene"
    where: Optional[Any]
    # TODO - more props; use dataclasses instead, or structured configs?


# Base class for all map scenes.
class Scene:
    def __init__(self, children: Optional[List[TypedChild]] = None):
        self._children = children if children is not None else []
        pass

    def make_node(self, grid: npt.NDArray[np.str_]):
        return Node(self, grid)

    # Render does two things:
    # - updates `node.grid` as it sees fit
    # - creates areas of interest in a node through `node.make_area()`
    def _render(self, node: Node) -> None:
        raise NotImplementedError("Subclass must implement render method")

    # Subclasses can override this to provide a list of children based on the node.
    # By default, children are static, which makes them configurable in the config file, but they can't depend
    # on the node.
    def get_children(self, node) -> List[TypedChild]:
        return self._children

    def render(self, node: Node):
        self._render(node)

        for query in self.get_children(node):
            sweep = query.get("sweep")
            subqueries: list[TypedChild] = [query]
            if sweep:
                subqueries = [
                    OmegaConf.merge(entry, query)  # type: ignore
                    for entry in sweep
                ]

            for query in subqueries:
                areas = node.select_areas(query)
                for area in areas:
                    scene = query["scene"]
                    if not isinstance(scene, Scene):
                        scene = hydra.utils.instantiate(scene, _recursive_=False)
                    child_node = scene.make_node(area.grid)
                    child_node.render()
