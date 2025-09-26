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

        bx0 = max(0, cx - base_size // 2)
        by0 = max(0, cy - base_size // 2)
        bx1 = min(width, bx0 + base_size)
        by1 = min(height, by0 + base_size)

        bw = max(0, bx1 - bx0)
        bh = max(0, by1 - by0)

        if bw > 0 and bh > 0:
            self.make_area(bx0, by0, bw, bh, tags=["base"])

        # Top-left
        if by0 > 0 and bx0 > 0:
            self.make_area(0, 0, bx0, by0, tags=["quadrant", "quadrant.0"])

        # Top-right
        if by0 > 0 and bx1 < width:
            self.make_area(bx1, 0, width - bx1, by0, tags=["quadrant", "quadrant.1"])

        # Bottom-left
        if by1 < height and bx0 > 0:
            self.make_area(0, by1, bx0, height - by1, tags=["quadrant", "quadrant.2"])

        # Bottom-right
        if by1 < height and bx1 < width:
            self.make_area(bx1, by1, width - bx1, height - by1, tags=["quadrant", "quadrant.3"])
