import pydantic

import mettagrid.mapgen.scene


class BoundedLayoutConfig(mettagrid.mapgen.scene.SceneConfig):
    max_width: int = pydantic.Field(ge=1)
    max_height: int = pydantic.Field(ge=1)
    tag: str


class BoundedLayout(mettagrid.mapgen.scene.Scene[BoundedLayoutConfig]):
    """
    Create a centered sub-area whose size is clamped by both the current zone size
    and the configured max_width/max_height.
    """

    def render(self):
        # Clamp within current zone and configured maximums
        width = max(1, min(self.width, self.config.max_width))
        height = max(1, min(self.height, self.config.max_height))

        # Optional minimum footprint to avoid tiny shapes
        min_w = min(self.width, max(10, self.config.max_width // 2))
        min_h = min(self.height, max(10, self.config.max_height // 2))
        width = max(min_w, width)
        height = max(min_h, height)

        x = (self.width - width) // 2
        y = (self.height - height) // 2
        # Ensure we don't exceed parent bounds
        width = min(width, self.width - x)
        height = min(height, self.height - y)
        if width > 0 and height > 0:
            self.make_area(x, y, width, height, tags=[self.config.tag])
