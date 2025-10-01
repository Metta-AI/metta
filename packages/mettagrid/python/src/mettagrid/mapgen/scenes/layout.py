from typing import Literal

from mettagrid.mapgen.scene import Scene, SceneConfig


class LayoutArea(SceneConfig):
    width: int
    height: int
    placement: Literal["center"] = "center"  # TODO - in the future, we will support more placements
    tag: str
    # TODO - should we support `scene: SceneConfig` here directly?
    # It would be more readable than defining tags and targeting them with `children_actions`.


class LayoutConfig(SceneConfig):
    areas: list[LayoutArea]


class Layout(Scene[LayoutConfig]):
    def render(self):
        for area in self.config.areas:
            if area.width > self.width or area.height > self.height:
                raise ValueError(f"Area {area} is too large for grid {self.width}x{self.height}")

            if area.placement == "center":
                x = (self.width - area.width) // 2
                y = (self.height - area.height) // 2
                self.make_area(x, y, area.width, area.height, tags=[area.tag])
            else:
                raise ValueError(f"Unknown placement: {area.placement}")
