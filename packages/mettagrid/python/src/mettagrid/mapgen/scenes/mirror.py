from typing import Literal

from pydantic import ConfigDict, Field

from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig

Symmetry = Literal["horizontal", "vertical", "x4"]


class MirrorConfig(SceneConfig):
    scene: SceneConfig
    symmetry: Symmetry = "horizontal"


class InnerMirrorConfig(SceneConfig):
    scene: SceneConfig


class Mirror(Scene[MirrorConfig]):
    def get_children(self) -> list[ChildrenAction]:
        symmetry_to_child_class = {
            "horizontal": HorizontalMirror,
            "vertical": VerticalMirror,
            "x4": X4Mirror,
        }

        return [
            ChildrenAction(
                scene=symmetry_to_child_class[self.config.symmetry].Config(scene=self.config.scene),
                where="full",
            ),
        ]

    def render(self):
        pass


class HorizontalMirrorConfig(InnerMirrorConfig):
    pass


class HorizontalMirror(Scene[HorizontalMirrorConfig]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=self.config.scene,
                where=AreaWhere(tags=["original"]),
            ),
            ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_x=True),
                where=AreaWhere(tags=["mirrored"]),
            ),
        ]

    def render(self):
        left_width = (self.width + 1) // 2  # take half, plus one for odd width
        self._original_mirror_area = self.make_area(0, 0, left_width, self.height, tags=["original"])
        self.make_area(left_width, 0, self.width - left_width, self.height, tags=["mirrored"])


class VerticalMirrorConfig(InnerMirrorConfig):
    pass


class VerticalMirror(Scene[VerticalMirrorConfig]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=self.config.scene,
                where=AreaWhere(tags=["original"]),
            ),
            ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_y=True),
                where=AreaWhere(tags=["mirrored"]),
            ),
        ]

    def render(self):
        top_height = (self.height + 1) // 2  # take half, plus one for odd width
        self._original_mirror_area = self.make_area(0, 0, self.width, top_height, tags=["original"])
        self.make_area(0, top_height, self.width, self.height - top_height, tags=["mirrored"])


class X4MirrorConfig(InnerMirrorConfig):
    pass


class X4Mirror(Scene[X4MirrorConfig]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(scene=self.config.scene, where=AreaWhere(tags=["original"])),
            ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_x=True),
                where=AreaWhere(tags=["mirrored_x"]),
            ),
            ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_y=True),
                where=AreaWhere(tags=["mirrored_y"]),
            ),
            ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_x=True, flip_y=True),
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


class MirroredConfig(SceneConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    parent: VerticalMirror | HorizontalMirror | X4Mirror = Field(exclude=True)
    flip_x: bool = False
    flip_y: bool = False


# Helper scene, shouldn't be used directly. (Its params are not serializable.)
class Mirrored(Scene[MirroredConfig]):
    def render(self):
        original_grid = self.config.parent._original_mirror_area.grid
        slice_x = slice(self.width - 1, None, -1) if self.config.flip_x else slice(self.width)
        slice_y = slice(self.height - 1, None, -1) if self.config.flip_y else slice(self.height)

        self.grid[:] = original_grid[slice_y, slice_x]
