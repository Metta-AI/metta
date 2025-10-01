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
    dither_edges: bool = False  # If True, add random noise to edges for organic look
    dither_amount: int = 2  # How many pixels to potentially dither at edges


class Layout(Scene[LayoutParams]):
    def render(self):
        for area in self.params.areas:
            # Auto-fit area to current grid to prevent oversize placement errors
            target_width = max(1, min(area.width, self.width))
            target_height = max(1, min(area.height, self.height))

            if area.placement == "center":
                x = (self.width - target_width) // 2
                y = (self.height - target_height) // 2

                # Apply dithering if requested
                if self.params.dither_edges:
                    x, y, target_width, target_height = self._apply_edge_dither(x, y, target_width, target_height)

                self.make_area(x, y, target_width, target_height, tags=[area.tag])
            else:
                raise ValueError(f"Unknown placement: {area.placement}")

    def _apply_edge_dither(self, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
        """Apply random dithering to edges for organic boundaries."""
        dither = self.params.dither_amount

        # Randomly adjust position and size within dither range
        x_offset = self.rng.integers(-dither, dither + 1)
        y_offset = self.rng.integers(-dither, dither + 1)
        w_offset = self.rng.integers(-dither, dither + 1)
        h_offset = self.rng.integers(-dither, dither + 1)

        # Ensure we stay within bounds
        new_x = max(0, min(self.width - width, x + x_offset))
        new_y = max(0, min(self.height - height, y + y_offset))
        new_width = max(1, min(self.width - new_x, width + w_offset))
        new_height = max(1, min(self.height - new_y, height + h_offset))

        return new_x, new_y, new_width, new_height
