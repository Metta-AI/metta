from typing import Literal

from pydantic import ConfigDict, Field

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import AreaWhere, ChildrenAction, SceneCfg

Symmetry = Literal["horizontal", "vertical", "x4"]


class MirrorParams(Config):
    scene: SceneCfg = Field(exclude=True)
    symmetry: Symmetry = "horizontal"


class InnerMirrorParams(Config):
    scene: SceneCfg = Field(exclude=True)


class Mirror(Scene[MirrorParams]):
    def get_children(self) -> list[ChildrenAction]:
        symmetry_to_child_class = {
            "horizontal": HorizontalMirror,
            "vertical": VerticalMirror,
            "x4": X4Mirror,
        }
        params = InnerMirrorParams(scene=self.params.scene)

        return [
            ChildrenAction(
                scene=symmetry_to_child_class[self.params.symmetry].factory(params),
                where="full",
            ),
        ]

    def render(self):
        pass


class HorizontalMirror(Scene[InnerMirrorParams]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=self.params.scene,
                where=AreaWhere(tags=["original"]),
            ),
            ChildrenAction(
                scene=Mirrored.factory(params={"parent": self, "flip_x": True}),
                where=AreaWhere(tags=["mirrored"]),
            ),
        ]

    def render(self):
        left_width = (self.width + 1) // 2  # take half, plus one for odd width
        self._original_mirror_area = self.make_area(0, 0, left_width, self.height, tags=["original"])
        self.make_area(left_width, 0, self.width - left_width, self.height, tags=["mirrored"])


class VerticalMirror(Scene[InnerMirrorParams]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=self.params.scene,
                where=AreaWhere(tags=["original"]),
            ),
            ChildrenAction(
                scene=Mirrored.factory(params={"parent": self, "flip_y": True}),
                where=AreaWhere(tags=["mirrored"]),
            ),
        ]

    def render(self):
        top_height = (self.height + 1) // 2  # take half, plus one for odd width
        self._original_mirror_area = self.make_area(0, 0, self.width, top_height, tags=["original"])
        self.make_area(0, top_height, self.width, self.height - top_height, tags=["mirrored"])


class X4Mirror(Scene[InnerMirrorParams]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(scene=self.params.scene, where=AreaWhere(tags=["original"])),
            ChildrenAction(
                scene=Mirrored.factory(params={"parent": self, "flip_x": True}),
                where=AreaWhere(tags=["mirrored_x"]),
            ),
            ChildrenAction(
                scene=Mirrored.factory(params={"parent": self, "flip_y": True}),
                where=AreaWhere(tags=["mirrored_y"]),
            ),
            ChildrenAction(
                scene=Mirrored.factory(params={"parent": self, "flip_x": True, "flip_y": True}),
                where=AreaWhere(tags=["mirrored_xy"]),
            ),
        ]

    def render(self):
        sub_width = (self.width + 1) // 2  # take half, plus one for odd width
        sub_height = (self.height + 1) // 2  # take half, plus one for odd width
        self._original_mirror_area = self.make_area(0, 0, sub_width, sub_height, tags=["original"])
        self.make_area(sub_width, 0, self.width - sub_width, sub_height, tags=["mirrored_x"])
        self.make_area(0, sub_height, sub_width, self.height - sub_height, tags=["mirrored_y"])
        self.make_area(sub_width, sub_height, self.width - sub_width, self.height - sub_height, tags=["mirrored_xy"])


class MirroredParams(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    parent: VerticalMirror | HorizontalMirror | X4Mirror = Field(exclude=True)
    flip_x: bool = False
    flip_y: bool = False


class Mirrored(Scene[MirroredParams]):
    def render(self):
        original_grid = self.params.parent._original_mirror_area.grid
        slice_x = slice(self.width - 1, None, -1) if self.params.flip_x else slice(self.width)
        slice_y = slice(self.height - 1, None, -1) if self.params.flip_y else slice(self.height)

        self.grid[:] = original_grid[slice_y, slice_x]
