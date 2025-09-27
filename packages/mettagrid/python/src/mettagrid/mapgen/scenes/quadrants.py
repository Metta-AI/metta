from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class QuadrantsParams(Config):
    base_size: int = 11


class Quadrants(Scene[QuadrantsParams]):
    """
    Reserve a centered square base area and create four quadrant areas around it.

    Tags:
    - base: the centered base area (typically 11x11)
    - quadrant, quadrant.0..3: top-left, top-right, bottom-left, bottom-right regions
    """

    def render(self):
        height, width = self.height, self.width
        base_size = self.params.base_size
        if base_size % 2 == 0:
            base_size += 1  # ensure odd for a true center cell

        cx = width // 2
        cy = height // 2

        # Full-map quadrants split by center lines; base will be stamped over later
        self.make_area(0, 0, cx, cy, tags=["quadrant", "quadrant.0"])  # top-left
        self.make_area(cx, 0, width - cx, cy, tags=["quadrant", "quadrant.1"])  # top-right
        self.make_area(0, cy, cx, height - cy, tags=["quadrant", "quadrant.2"])  # bottom-left
        self.make_area(cx, cy, width - cx, height - cy, tags=["quadrant", "quadrant.3"])  # bottom-right

        # Central base area (for later stamping)
        bx0 = max(0, cx - base_size // 2)
        by0 = max(0, cy - base_size // 2)
        bw = min(base_size, width - bx0)
        bh = min(base_size, height - by0)
        if bw > 0 and bh > 0:
            self.make_area(bx0, by0, bw, bh, tags=["base"])
