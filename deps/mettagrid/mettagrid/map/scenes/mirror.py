from typing import Any, List, Literal, Optional

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene

Symmetry = Literal["horizontal", "vertical", "x4"]


class Mirror(Scene):
    def __init__(self, scene: Scene, symmetry: Symmetry = "horizontal", children: Optional[List[Any]] = None):
        super().__init__(children=children)
        self._scene = scene
        self._symmetry = symmetry

    def _render(self, node: Node):
        if self._symmetry == "horizontal":
            left_width = (node.width + 1) // 2  # take half, plus one for odd width
            left_grid = node.grid[:, :left_width]
            child_node = self._scene.make_node(left_grid)
            child_node.render()

            node.grid[:, node.width - left_width :] = child_node.grid[:, ::-1]

        elif self._symmetry == "vertical":
            top_height = (node.height + 1) // 2  # take half, plus one for odd width
            top_grid = node.grid[:top_height, :]
            child_node = self._scene.make_node(top_grid)
            child_node.render()

            node.grid[node.height - top_height :, :] = child_node.grid[::-1, :]

        elif self._symmetry == "x4":
            # fill top left quadrant
            sub_width = (node.width + 1) // 2  # take half, plus one for odd width
            sub_height = (node.height + 1) // 2  # take half, plus one for odd width

            sub_grid = node.grid[:sub_height, :sub_width]
            child_node = self._scene.make_node(sub_grid)
            child_node.render()

            # reflect to the right
            node.grid[:sub_height, node.width - sub_width :] = child_node.grid[:, ::-1]

            # reflect to the bottom
            node.grid[node.height - sub_height :, :] = node.grid[:sub_height, :][::-1, :]
