from typing import Literal

from metta.map.node import Node, make_node
from metta.map.scene import SceneCfg
from metta.util.config import Config

Symmetry = Literal["horizontal", "vertical", "x4"]


class MirrorParams(Config):
    scene: SceneCfg
    symmetry: Symmetry = "horizontal"


class Mirror(Node[MirrorParams]):
    params_type = MirrorParams

    def render(self):
        symmetry = self.params.symmetry
        scene = self.params.scene

        if symmetry == "horizontal":
            left_width = (self.width + 1) // 2  # take half, plus one for odd width
            left_grid = self.grid[:, :left_width]
            child_node = make_node(scene, left_grid)
            child_node.render_with_children()

            self.grid[:, self.width - left_width :] = child_node.grid[:, ::-1]

        elif symmetry == "vertical":
            top_height = (self.height + 1) // 2  # take half, plus one for odd width
            top_grid = self.grid[:top_height, :]
            child_node = make_node(scene, top_grid)
            child_node.render_with_children()

            self.grid[self.height - top_height :, :] = child_node.grid[::-1, :]

        elif symmetry == "x4":
            # fill top left quadrant
            sub_width = (self.width + 1) // 2  # take half, plus one for odd width
            sub_height = (self.height + 1) // 2  # take half, plus one for odd width

            sub_grid = self.grid[:sub_height, :sub_width]
            child_node = make_node(scene, sub_grid)
            child_node.render_with_children()

            # reflect to the right
            self.grid[:sub_height, self.width - sub_width :] = child_node.grid[:, ::-1]

            # reflect to the bottom
            self.grid[self.height - sub_height :, :] = self.grid[:sub_height, :][::-1, :]
