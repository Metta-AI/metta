from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class InsetAreaParams(Config):
    padding: int = 0
    min_width: int = 1
    min_height: int = 1
    max_width: int | None = None
    max_height: int | None = None
    tag: str


class InsetArea(Scene[InsetAreaParams]):
    """
    Create a centered area inset from the scene bounds by a configurable padding.

    Useful for carving nested pockets (e.g., dungeon cores) inside larger zones
    without needing to know the zone dimensions when configuring the scene tree.
    """

    def render(self):
        padding = max(0, int(self.params.padding))

        width = max(self.width - 2 * padding, 1)
        height = max(self.height - 2 * padding, 1)

        width = max(width, int(self.params.min_width))
        height = max(height, int(self.params.min_height))

        if self.params.max_width is not None:
            width = min(width, int(self.params.max_width))
        width = min(width, self.width)

        if self.params.max_height is not None:
            height = min(height, int(self.params.max_height))
        height = min(height, self.height)

        x = (self.width - width) // 2
        y = (self.height - height) // 2

        self.make_area(x, y, width, height, tags=[self.params.tag])
