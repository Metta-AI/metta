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
            # Auto-fit area to current grid to prevent oversize placement errors
            target_width = max(1, min(area.width, self.width))
            target_height = max(1, min(area.height, self.height))

            if area.placement == "center":
                x = (self.width - target_width) // 2
                y = (self.height - target_height) // 2
                self.make_area(x, y, target_width, target_height, tags=[area.tag])
            else:
                raise ValueError(f"Unknown placement: {area.placement}")
