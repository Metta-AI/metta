from typing import Literal, Sequence

from mettagrid.mapgen.scene import Scene, SceneConfig

DEFAULT_CORNER_CHESTS: tuple[str, str, str, str] = (
    "chest_carbon",
    "chest_oxygen",
    "chest_germanium",
    "chest_silicon",
)

DEFAULT_EXTRACTORS: tuple[str, str, str, str] = (
    "carbon_extractor",
    "oxygen_extractor",
    "germanium_extractor",
    "silicon_extractor",
)


class BaseHubConfig(SceneConfig):
    assembler_object: str = "assembler"
    corner_generator: str | None = None
    spawn_symbol: str = "agent.agent"
    # If set, place at least this many spawn pads (best-effort) in the hub
    spawn_count: int | None = None
    hub_width: int = 21
    hub_height: int = 21
    include_inner_wall: bool = True
    # Order: top-left, top-right, bottom-left, bottom-right.
    corner_objects: list[str] | None = None
    corner_bundle: Literal["chests", "extractors", "none", "custom"] = "chests"
    cross_objects: list[str] | None = None
    cross_bundle: Literal["none", "chests", "extractors", "custom"] = "none"
    cross_distance: int = 4
    layout: Literal["default", "tight"] = "default"
    charger_object: str = "charger"
    heart_chest_object: str = "chest"


class BaseHub(Scene[BaseHubConfig]):
    """
    Build a symmetric 11x11 base:
    - Center cell: assembler with charger two cells above
    - Four corner generators with one empty cell of clearance on all sides
    - Symmetric L-shaped empty corridors at each corner to form 4 exits
    - Spawn pads around center with empty clearance
    """

    def render(self) -> None:
        full_grid = self.grid
        full_h, full_w = self.height, self.width
        cfg = self.config

        # Compute centered hub region
        hub_w = max(7, min(cfg.hub_width, full_w))
        hub_h = max(7, min(cfg.hub_height, full_h))
        x0 = (full_w - hub_w) // 2
        y0 = (full_h - hub_h) // 2
        x1 = x0 + hub_w
        y1 = y0 + hub_h

        grid = full_grid[y0:y1, x0:x1]
        h, w = hub_h, hub_w
        cx, cy = w // 2, h // 2

        # Clear hub region
        grid[:] = "empty"

        # Optional inner wall around hub region
        if cfg.include_inner_wall and h >= 3 and w >= 3:
            grid[0, :] = "wall"
            grid[-1, :] = "wall"
            grid[:, 0] = "wall"
            grid[:, -1] = "wall"

            gate_half = 1
            # top/bottom gates
            grid[0, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[1, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[h - 1, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[h - 2, cx - gate_half : cx + gate_half + 1] = "empty"
            # left/right gates
            grid[cy - gate_half : cy + gate_half + 1, 0] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, 1] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, w - 1] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, w - 2] = "empty"

        original_grid = self.grid
        original_height = self.height
        original_width = self.width

        try:
            self.grid = grid
            self.height = h
            self.width = w

            if cfg.layout == "tight":
                self._render_tight_layout(cx, cy)
            else:
                self._render_default_layout(cx, cy)
        finally:
            self.grid = original_grid
            self.height = original_height
            self.width = original_width

    def _place_spawn_pads(self, positions: Sequence[tuple[int, int]]) -> None:
        grid = self.grid
        h, w = self.height, self.width

        for x, y in positions:
            if 1 <= x < w - 1 and 1 <= y < h - 1 and grid[y, x] == "empty":
                grid[y, x] = self.config.spawn_symbol

    def _resolve_corner_names(self) -> list[str]:
        cfg = self.config
        if cfg.corner_objects and len(cfg.corner_objects) == 4:
            return list(cfg.corner_objects)
        if cfg.corner_generator:
            return [cfg.corner_generator] * 4
        if cfg.corner_bundle == "extractors":
            return list(DEFAULT_EXTRACTORS)
        if cfg.corner_bundle == "none":
            return []
        return list(DEFAULT_CORNER_CHESTS)

    def _resolve_cross_names(self) -> list[str]:
        cfg = self.config
        if cfg.cross_objects and len(cfg.cross_objects) == 4:
            return list(cfg.cross_objects)
        if cfg.cross_bundle == "chests":
            return list(DEFAULT_CORNER_CHESTS)
        if cfg.cross_bundle == "extractors":
            return list(DEFAULT_EXTRACTORS)
        return []

    def _cross_positions(self, cx: int, cy: int, distance: int) -> list[tuple[int, int]]:
        dist = max(1, distance)
        return [
            (cx, cy - dist),
            (cx + dist, cy),
            (cx, cy + dist),
            (cx - dist, cy),
        ]

    def _place_named_objects(self, positions: Sequence[tuple[int, int]], names: Sequence[str]) -> None:
        grid = self.grid
        h, w = self.height, self.width

        for (x, y), name in zip(positions, names, strict=False):
            if not name:
                continue
            if 0 <= x < w and 0 <= y < h:
                grid[y, x] = name

    def _render_default_layout(self, cx: int, cy: int) -> None:
        grid = self.grid
        h, w = self.height, self.width
        cfg = self.config

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
            grid[cy, cx] = cfg.assembler_object

            charger_y = cy - 3
            if 1 <= charger_y < h - 1:
                grid[charger_y, cx] = cfg.charger_object

            chest_y = cy + 3
            if 1 <= chest_y < h - 1:
                grid[chest_y, cx] = cfg.heart_chest_object

        # Spawn pads: ensure at least spawn_count if provided, otherwise place 4
        desired = max(0, int(cfg.spawn_count)) if cfg.spawn_count is not None else 4
        base_positions = [(cx, cy - 2), (cx + 2, cy), (cx, cy + 2), (cx - 2, cy)]
        valid_positions: list[tuple[int, int]] = []
        for sx, sy in base_positions:
            if len(valid_positions) >= desired:
                break
            if 0 <= sx < w and 0 <= sy < h and grid[sy, sx] == "empty":
                valid_positions.append((sx, sy))

        # If more spawns are needed, expand rings until we have enough empty tiles
        radius = 3
        max_radius = max(h, w)
        while len(valid_positions) < desired and radius < max_radius:
            candidates = [
                (cx + radius, cy),
                (cx - radius, cy),
                (cx, cy + radius),
                (cx, cy - radius),
                (cx + radius, cy + radius),
                (cx + radius, cy - radius),
                (cx - radius, cy + radius),
                (cx - radius, cy - radius),
            ]
            for sx, sy in candidates:
                if len(valid_positions) >= desired:
                    break
                if 0 <= sx < w and 0 <= sy < h and grid[sy, sx] == "empty":
                    valid_positions.append((sx, sy))
            radius += 1

        self._place_spawn_pads(valid_positions[:desired])

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

        cross_names = self._resolve_cross_names()
        if cross_names:
            cross_positions = self._cross_positions(cx, cy, cfg.cross_distance)
            self._place_named_objects(cross_positions, cross_names)

    def _render_tight_layout(self, cx: int, cy: int) -> None:
        grid = self.grid
        h, w = self.height, self.width
        cfg = self.config

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
            place_building(cx, cy, cfg.assembler_object)

        charger_y = cy - 2
        if 1 <= cx < w - 1 and 1 <= charger_y < h - 1:
            place_building(cx, charger_y, cfg.charger_object)

        chest_y = cy + 2
        if 1 <= cx < w - 1 and 1 <= chest_y < h - 1:
            place_building(cx, chest_y, cfg.heart_chest_object)

        corner_positions = [
            (cx - 2, cy - 2),
            (cx + 2, cy - 2),
            (cx - 2, cy + 2),
            (cx + 2, cy + 2),
        ]

        for (x, y), name in zip(corner_positions, self._resolve_corner_names(), strict=False):
            place_building(x, y, name)

        cross_names = self._resolve_cross_names()
        if cross_names:
            cross_positions = self._cross_positions(cx, cy, cfg.cross_distance)
            self._place_named_objects(cross_positions, cross_names)

        self._ensure_clearance(building_positions)

        perimeter_radius = core_radius + 1
        self._build_tight_perimeter(cx, cy, perimeter_radius, gate_half=1)

        # Spawn pads: ensure at least spawn_count if provided, otherwise place 4 near the perimeter
        desired = max(0, int(cfg.spawn_count)) if cfg.spawn_count is not None else 4
        spawn_distance = perimeter_radius + 1
        positions: list[tuple[int, int]] = [
            (cx, cy - spawn_distance),
            (cx + spawn_distance, cy),
            (cx, cy + spawn_distance),
            (cx - spawn_distance, cy),
        ]
        # If more spawns needed, distribute more around the perimeter ring
        step = max(1, (2 * perimeter_radius + 1) // 4)
        if len(positions) < desired:
            for dx in range(-perimeter_radius, perimeter_radius + 1, step):
                if len(positions) >= desired:
                    break
                positions.append((cx + dx, cy - spawn_distance))
                if len(positions) >= desired:
                    break
                positions.append((cx + dx, cy + spawn_distance))
            for dy in range(-perimeter_radius, perimeter_radius + 1, step):
                if len(positions) >= desired:
                    break
                positions.append((cx - spawn_distance, cy + dy))
                if len(positions) >= desired:
                    break
                positions.append((cx + spawn_distance, cy + dy))

        valid_positions = [
            (sx, sy) for sx, sy in positions[:desired] if 0 <= sx < w and 0 <= sy < h and grid[sy, sx] == "empty"
        ]
        self._place_spawn_pads(valid_positions)

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
