from typing import Literal, Sequence

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BaseHubParams(Config):
    assembler_object: str = "assembler"
    corner_generator: str = "generator_red"
    spawn_symbol: str = "agent.agent"
    include_inner_wall: bool = True
    # Order: top-left, top-right, bottom-left, bottom-right.
    corner_objects: list[str] | None = None
    layout: Literal["default", "tight"] = "default"
    charger_object: str = "charger"


class BaseHub(Scene[BaseHubParams]):
    """
    Build a symmetric 11x11 base:
    - Center cell: assembler with charger two cells above
    - Four corner generators with one empty cell of clearance on all sides
    - Symmetric L-shaped empty corridors at each corner to form 4 exits
    - Spawn pads around center with empty clearance
    """

    def render(self) -> None:
        grid = self.grid
        h, w = self.height, self.width

        # Fill with empty to start
        grid[:] = "empty"

        cx, cy = w // 2, h // 2

        # Optional inner wall ring around the border of the base area
        if self.params.include_inner_wall and h >= 3 and w >= 3:
            grid[0, :] = "wall"
            grid[-1, :] = "wall"
            grid[:, 0] = "wall"
            grid[:, -1] = "wall"

            # Deterministic 3-wide gates at midpoints of each side
            gate_half = 1
            # top gate centered at cx
            grid[0, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[1, cx - gate_half : cx + gate_half + 1] = "empty"
            # bottom gate
            grid[h - 1, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[h - 2, cx - gate_half : cx + gate_half + 1] = "empty"
            # left gate
            grid[cy - gate_half : cy + gate_half + 1, 0] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, 1] = "empty"
            # right gate
            grid[cy - gate_half : cy + gate_half + 1, w - 1] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, w - 2] = "empty"

        if self.params.layout == "tight" and min(h, w) >= 7:
            self._render_tight_layout(cx, cy)
        else:
            self._render_default_layout(cx, cy)

    def _place_spawn_pads(self, positions: Sequence[tuple[int, int]]) -> None:
        grid = self.grid
        h, w = self.height, self.width

        for x, y in positions:
            if 1 <= x < w - 1 and 1 <= y < h - 1 and grid[y, x] == "empty":
                grid[y, x] = self.params.spawn_symbol

    def _resolve_corner_names(self) -> list[str]:
        if self.params.corner_objects and len(self.params.corner_objects) == 4:
            return list(self.params.corner_objects)
        return [self.params.corner_generator] * 4

    def _render_default_layout(self, cx: int, cy: int) -> None:
        grid = self.grid
        h, w = self.height, self.width

        corridor_width = 3
        half = corridor_width // 2

        # Carve plus-shaped corridors that meet each gate with 3-tile width
        x0 = max(1, cx - half)
        x1 = min(w - 1, cx + half + 1)
        y0 = max(1, cy - half)
        y1 = min(h - 1, cy + half + 1)

        grid[1 : h - 1, x0:x1] = "empty"
        grid[y0:y1, 1 : w - 1] = "empty"

        # Place central altar, charger, and chest after carving so they persist
        if 1 <= cx < w - 1 and 1 <= cy < h - 1:
            grid[cy, cx] = self.params.assembler_object

            charger_y = cy - 3
            if 1 <= charger_y < h - 1:
                grid[charger_y, cx] = self.params.charger_object

            chest_y = cy + 3
            if 1 <= chest_y < h - 1:
                grid[chest_y, cx] = "chest"

        # Spawn pads in plus-shape around center with clearance
        self._place_spawn_pads(
            [
                (cx, cy - 2),
                (cx + 2, cy),
                (cx, cy + 2),
                (cx - 2, cy),
            ]
        )

        # Place corner objects symmetrically
        corner_positions = [
            (2, 2),
            (w - 3, 2),
            (2, h - 3),
            (w - 3, h - 3),
        ]

        for (x, y), name in zip(corner_positions, self._resolve_corner_names(), strict=False):
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                grid[y, x] = name

    def _render_tight_layout(self, cx: int, cy: int) -> None:
        grid = self.grid
        h, w = self.height, self.width

        # Carve L exits first to keep ingress paths consistent with default layout
        self._carve_L(1, 1, orientation="right-down")
        self._carve_L(w - 4, 1, orientation="left-down")
        self._carve_L(1, h - 4, orientation="right-up")
        self._carve_L(w - 4, h - 4, orientation="left-up")

        core_radius = 3
        x0 = max(0, cx - core_radius)
        x1 = min(w, cx + core_radius + 1)
        y0 = max(0, cy - core_radius)
        y1 = min(h, cy + core_radius + 1)
        grid[y0:y1, x0:x1] = "empty"

        building_positions: list[tuple[int, int]] = []

        def place_building(x: int, y: int, name: str) -> None:
            if not (1 <= x < w - 1 and 1 <= y < h - 1):
                return
            if grid[y, x] != "empty":
                return
            grid[y, x] = name
            building_positions.append((x, y))

        if 1 <= cx < w - 1 and 1 <= cy < h - 1:
            place_building(cx, cy, self.params.assembler_object)

        charger_y = cy - 2
        if 1 <= cx < w - 1 and 1 <= charger_y < h - 1:
            place_building(cx, charger_y, self.params.charger_object)

        chest_y = cy + 2
        if 1 <= cx < w - 1 and 1 <= chest_y < h - 1:
            place_building(cx, chest_y, "chest")

        corner_positions = [
            (cx - 2, cy - 2),
            (cx + 2, cy - 2),
            (cx - 2, cy + 2),
            (cx + 2, cy + 2),
        ]

        for (x, y), name in zip(corner_positions, self._resolve_corner_names(), strict=False):
            place_building(x, y, name)

        self._ensure_clearance(building_positions)

        perimeter_radius = core_radius + 1
        self._build_tight_perimeter(cx, cy, perimeter_radius, gate_half=1)

        spawn_distance = perimeter_radius + 1
        spawn_candidates = [
            (cx, cy - spawn_distance),
            (cx + spawn_distance, cy),
            (cx, cy + spawn_distance),
            (cx - spawn_distance, cy),
        ]

        self._place_spawn_pads(
            [(sx, sy) for sx, sy in spawn_candidates if 0 <= sx < w and 0 <= sy < h and grid[sy, sx] == "empty"]
        )

    def _ensure_clearance(self, positions: Sequence[tuple[int, int]]) -> None:
        grid = self.grid
        h, w = self.height, self.width

        for x, y in positions:
            for nx in range(x - 1, x + 2):
                if not (0 <= nx < w):
                    continue
                for ny in range(y - 1, y + 2):
                    if not (0 <= ny < h):
                        continue
                    if (nx, ny) == (x, y):
                        continue
                    grid[ny, nx] = "empty"

    def _build_tight_perimeter(self, cx: int, cy: int, radius: int, gate_half: int) -> None:
        if radius <= 0:
            return

        grid = self.grid
        h, w = self.height, self.width

        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                if not (0 <= x < w and 0 <= y < h):
                    continue

                on_perimeter = (abs(x - cx) == radius and abs(y - cy) <= radius) or (
                    abs(y - cy) == radius and abs(x - cx) <= radius
                )
                if not on_perimeter:
                    continue

                on_gate = (abs(x - cx) <= gate_half and abs(y - cy) == radius) or (
                    abs(y - cy) <= gate_half and abs(x - cx) == radius
                )

                if on_gate:
                    continue

                grid[y, x] = "wall"

    def _carve_L(self, x: int, y: int, orientation: Literal["right-down", "left-down", "right-up", "left-up"]):
        grid = self.grid
        h, w = self.height, self.width

        width = 3  # corridor thickness
        leg = max(3, min(h, w) // 3)  # leg length based on base size

        def carve_rect(x0: int, y0: int, cw: int, ch: int):
            x1 = max(0, x0)
            y1 = max(0, y0)
            x2 = min(w, x0 + cw)
            y2 = min(h, y0 + ch)
            if x2 > x1 and y2 > y1:
                grid[y1:y2, x1:x2] = "empty"

        if orientation == "right-down":
            # horizontal then vertical
            carve_rect(x, y, leg, width)
            carve_rect(x + leg - width, y, width, leg)
            # open top border
            carve_rect(x, 0, width, 1)
        elif orientation == "left-down":
            carve_rect(x - leg + width, y, leg, width)
            carve_rect(x - leg + width, y, width, leg)
            # open top border
            carve_rect(x - width + 1, 0, width, 1)
        elif orientation == "right-up":
            carve_rect(x, y, leg, width)
            carve_rect(x + leg - width, y - leg + width, width, leg)
            # open left border
            carve_rect(0, y - width + 1, width, width)
        elif orientation == "left-up":
            carve_rect(x - leg + width, y, leg, width)
            carve_rect(x - leg + width, y - leg + width, width, leg)
            # open bottom border
            carve_rect(x - width + 1, h - 1, width, 1)
