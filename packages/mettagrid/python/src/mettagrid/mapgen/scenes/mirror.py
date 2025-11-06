import typing

import pydantic

import mettagrid.mapgen.area
import mettagrid.mapgen.scene

Symmetry = typing.Literal["horizontal", "vertical", "x4"]


def _make_area_if_positive(
    scene: mettagrid.mapgen.scene.Scene,
    x: int,
    y: int,
    width: int,
    height: int,
    tags: list[str],
) -> mettagrid.mapgen.area.Area | None:
    if width <= 0 or height <= 0:
        return None
    return scene.make_area(x, y, width, height, tags=tags)


class MirrorConfig(mettagrid.mapgen.scene.SceneConfig):
    scene: mettagrid.mapgen.scene.SceneConfig
    symmetry: Symmetry = "horizontal"


class InnerMirrorConfig(mettagrid.mapgen.scene.SceneConfig):
    scene: mettagrid.mapgen.scene.SceneConfig


class Mirror(mettagrid.mapgen.scene.Scene[MirrorConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        symmetry_to_child_class = {
            "horizontal": HorizontalMirror,
            "vertical": VerticalMirror,
            "x4": X4Mirror,
        }

        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=symmetry_to_child_class[self.config.symmetry].Config(scene=self.config.scene),
                where="full",
            ),
        ]

    def render(self):
        pass


class HorizontalMirrorConfig(InnerMirrorConfig):
    pass


class HorizontalMirror(mettagrid.mapgen.scene.Scene[HorizontalMirrorConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=self.config.scene,
                where=mettagrid.mapgen.area.AreaWhere(tags=["original"]),
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_x=True),
                where=mettagrid.mapgen.area.AreaWhere(tags=["mirrored"]),
            ),
        ]

    def render(self):
        left_width = (self.width + 1) // 2  # take half, plus one for odd width
        original_area = _make_area_if_positive(self, 0, 0, left_width, self.height, ["original"])
        self._original_mirror_area = original_area or self.area

        mirrored_width = self.width - left_width
        _make_area_if_positive(self, left_width, 0, mirrored_width, self.height, ["mirrored"])


class VerticalMirrorConfig(InnerMirrorConfig):
    pass


class VerticalMirror(mettagrid.mapgen.scene.Scene[VerticalMirrorConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=self.config.scene,
                where=mettagrid.mapgen.area.AreaWhere(tags=["original"]),
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_y=True),
                where=mettagrid.mapgen.area.AreaWhere(tags=["mirrored"]),
            ),
        ]

    def render(self):
        top_height = (self.height + 1) // 2  # take half, plus one for odd width
        original_area = _make_area_if_positive(self, 0, 0, self.width, top_height, ["original"])
        self._original_mirror_area = original_area or self.area

        mirrored_height = self.height - top_height
        _make_area_if_positive(self, 0, top_height, self.width, mirrored_height, ["mirrored"])


class X4MirrorConfig(InnerMirrorConfig):
    pass


class X4Mirror(mettagrid.mapgen.scene.Scene[X4MirrorConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=self.config.scene, where=mettagrid.mapgen.area.AreaWhere(tags=["original"])
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_x=True),
                where=mettagrid.mapgen.area.AreaWhere(tags=["mirrored_x"]),
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_y=True),
                where=mettagrid.mapgen.area.AreaWhere(tags=["mirrored_y"]),
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=Mirrored.Config(parent=self, flip_x=True, flip_y=True),
                where=mettagrid.mapgen.area.AreaWhere(tags=["mirrored_xy"]),
            ),
        ]

    def render(self):
        sub_width = (self.width + 1) // 2  # take half, plus one for odd width
        sub_height = (self.height + 1) // 2  # take half, plus one for odd width
        original_area = _make_area_if_positive(self, 0, 0, sub_width, sub_height, ["original"])
        self._original_mirror_area = original_area or self.area

        mirrored_width = self.width - sub_width
        mirrored_height = self.height - sub_height

        _make_area_if_positive(self, sub_width, 0, mirrored_width, sub_height, ["mirrored_x"])
        _make_area_if_positive(self, 0, sub_height, sub_width, mirrored_height, ["mirrored_y"])
        _make_area_if_positive(self, sub_width, sub_height, mirrored_width, mirrored_height, ["mirrored_xy"])


class MirroredConfig(mettagrid.mapgen.scene.SceneConfig):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    parent: VerticalMirror | HorizontalMirror | X4Mirror = pydantic.Field(exclude=True)
    flip_x: bool = False
    flip_y: bool = False


# Helper scene, shouldn't be used directly. (Its params are not serializable.)
class Mirrored(mettagrid.mapgen.scene.Scene[MirroredConfig]):
    def render(self):
        original_grid = self.config.parent._original_mirror_area.grid
        slice_x = slice(self.width - 1, None, -1) if self.config.flip_x else slice(self.width)
        slice_y = slice(self.height - 1, None, -1) if self.config.flip_y else slice(self.height)

        self.grid[:] = original_grid[slice_y, slice_x]
