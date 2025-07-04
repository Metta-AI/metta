from typing import Literal

from metta.common.util.config import Config
from metta.map.scene import Scene


class LayoutArea(Config):
    width: int
    height: int
    placement: Literal["center"] = "center"
    tag: str


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
