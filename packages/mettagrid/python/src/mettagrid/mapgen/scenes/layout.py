from typing import Literal

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class LayoutArea(Config):
    width: int
    height: int
    placement: Literal["center"] = "center"  # TODO - in the future, we will support more placements
    tag: str
    # TODO - should we support `scene: SceneCfg` here directly?
    # It would be more readable than defining tags and targeting them with `children_actions`.


class LayoutParams(Config):
    areas: list[LayoutArea]


class Layout(Scene[LayoutParams]):
    def render(self):
        for area in self.params.areas:
            if area.width > self.width or area.height > self.height:
                raise ValueError(f"Area {area} is too large for grid {self.width}x{self.height}")

            if area.placement == "center":
                x = (self.width - area.width) // 2
                y = (self.height - area.height) // 2
                self.make_area(x, y, area.width, area.height, tags=[area.tag])
            else:
                raise ValueError(f"Unknown placement: {area.placement}")
