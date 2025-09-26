from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class AssemblerMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["AssemblerMapBuilder"]):
        seed: Optional[int] = None

        width: int = 10
        height: int = 10
        objects: dict[str, int] = {}
        agents: int | dict[str, int] = 0
        border_width: int = 0
        border_object: str = "wall"

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

    def build(self):
        # Reset RNG to ensure deterministic builds across multiple calls
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)

        height = self._config.height
        width = self._config.width

        # Create empty grid
        grid = np.full((height, width), "empty", dtype="<U50")

        # Draw border first if needed
        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        # Calculate inner area where objects can be placed
        if self._config.border_width > 0:
            inner_height = max(0, height - 2 * self._config.border_width)
            inner_width = max(0, width - 2 * self._config.border_width)
        else:
            inner_height = height
            inner_width = width

        # If inner area is too small for a 1-cell padding around objects, return as is
        if inner_height < 3 or inner_width < 3:
            return GameMap(grid)

        # Prepare agent symbols (placed after objects)
        if isinstance(self._config.agents, int):
            agent_symbols = ["agent.agent"] * self._config.agents
        elif isinstance(self._config.agents, dict):
            agent_symbols = ["agent." + agent for agent, na in self._config.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents configuration: {self._config.agents}")

        # Prepare object symbols
        object_symbols = []
        for obj_name, count in self._config.objects.items():
            object_symbols.extend([obj_name] * count)

        # Compute valid placement bounds that guarantee a 1-cell padding from any border
        top = self._config.border_width + 1
        left = self._config.border_width + 1
        bottom = height - self._config.border_width - 2
        right = width - self._config.border_width - 2

        if bottom < top or right < left:
            return GameMap(grid)

        # Keep a mask of reserved cells (objects and their padding)
        reserved = np.zeros((height, width), dtype=bool)

        # Helper to mark a 3x3 neighborhood as reserved around (i, j)
        def reserve_with_padding(i: int, j: int):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < height and 0 <= jj < width:
                        reserved[ii, jj] = True

        # Generate all candidate centers that satisfy border padding
        cand_rows = np.arange(top, bottom + 1)
        cand_cols = np.arange(left, right + 1)
        candidates = [(i, j) for i in cand_rows for j in cand_cols]
        self._rng.shuffle(candidates)

        # Place objects greedily with padding constraint
        for symbol in object_symbols:
            placed = False
            for idx in range(len(candidates)):
                i, j = candidates[idx]
                # Check 3x3 neighborhood is unreserved
                if reserved[i - 1 : i + 2, j - 1 : j + 2].any():
                    continue
                # Place object
                grid[i, j] = symbol
                reserve_with_padding(i, j)
                # Remove this candidate to avoid reusing exact cell
                candidates.pop(idx)
                placed = True
                break
            if not placed:
                # No valid spot left; stop placing remaining objects
                break

        # Now place agents in remaining empty cells (no special padding required)
        if agent_symbols:
            empty_mask = grid == "empty"
            empty_indices = np.argwhere(empty_mask)
            if len(empty_indices) > 0:
                self._rng.shuffle(empty_indices)
                num_placeable = min(len(agent_symbols), len(empty_indices))
                for k in range(num_placeable):
                    i, j = empty_indices[k]
                    grid[i, j] = agent_symbols[k]

        return GameMap(grid)


class RegionAssemblerMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["RegionAssemblerMapBuilder"]):
        seed: Optional[int] = None

        width: int = 12
        height: int = 12
        agents: int | dict[str, int] = 1
        border_width: int = 1
        border_object: str = "wall"

        # Central assemblers to place with 3x3 padding
        num_assemblers: int = 1

        # Resource placement with regional bias
        # List of dicts: {"name": str, "count": int, "region": str}
        resource_regions: list[dict] = []

        # Separation mode: "strict" = only in designated region; "soft" uses region_bias
        separation_mode: str = "strict"
        region_bias: float = 0.8

    def __init__(self, config: "RegionAssemblerMapBuilder.Config"):
        self._config: RegionAssemblerMapBuilder.Config = config
        self._rng = np.random.default_rng(self._config.seed)

    def build(self) -> GameMap:
        # Reset RNG if seed is provided
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)
        rng = self._rng

        h, w = self._config.height, self._config.width
        grid = np.full((h, w), "empty", dtype="<U50")

        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        def in_bounds(i: int, j: int) -> bool:
            bw = self._config.border_width
            return bw <= i < h - bw and bw <= j < w - bw

        reserved = np.zeros((h, w), dtype=bool)

        def reserve_3x3(i: int, j: int):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < h and 0 <= jj < w:
                        reserved[ii, jj] = True

        # Place assemblers around center
        centers: list[tuple[int, int]] = []
        cx, cy = h // 2, w // 2
        grid[cx, cy] = "altar"
        reserve_3x3(cx, cy)
        centers.append((cx, cy))

        extra = max(0, self._config.num_assemblers - 1)
        attempts = 0
        while extra > 0 and attempts < 1000:
            attempts += 1
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.integers(low=3, high=max(4, min(h, w) // 3))
            i = int(cx + dist * np.sin(ang))
            j = int(cy + dist * np.cos(ang))
            if not in_bounds(i, j):
                continue
            if reserved[max(0, i - 1) : min(h, i + 2), max(0, j - 1) : min(w, j + 2)].any():
                continue
            grid[i, j] = "altar"
            reserve_3x3(i, j)
            centers.append((i, j))
            extra -= 1

        # Region bounds helper
        def get_region_bounds(region: str) -> tuple[int, int, int, int]:
            bw = self._config.border_width
            mid_i, mid_j = h // 2, w // 2
            regions = {
                "north": (bw, mid_i, bw, w - bw),
                "south": (mid_i, h - bw, bw, w - bw),
                "east": (bw, h - bw, mid_j, w - bw),
                "west": (bw, h - bw, bw, mid_j),
                "northeast": (bw, mid_i, mid_j, w - bw),
                "northwest": (bw, mid_i, bw, mid_j),
                "southeast": (mid_i, h - bw, mid_j, w - bw),
                "southwest": (mid_i, h - bw, bw, mid_j),
                "center": (max(bw, mid_i - 2), min(h - bw, mid_i + 3), max(bw, mid_j - 2), min(w - bw, mid_j + 3)),
            }
            return regions.get(region, (bw, h - bw, bw, w - bw))

        # Place resources by regional specification
        for spec in self._config.resource_regions:
            name = str(spec.get("name", "mine_red"))
            count = int(spec.get("count", 3))
            region = str(spec.get("region", "north"))

            placed = 0
            tries = 0
            while placed < count and tries < count * 30:
                tries += 1

                # Choose target region based on separation mode
                if self._config.separation_mode == "strict" or rng.random() < self._config.region_bias:
                    target_region = region
                else:
                    all_regions = [
                        "north",
                        "south",
                        "east",
                        "west",
                        "northeast",
                        "northwest",
                        "southeast",
                        "southwest",
                    ]
                    others = [r for r in all_regions if r != region]
                    target_region = rng.choice(others)

                min_i, max_i, min_j, max_j = get_region_bounds(target_region)
                i = int(rng.integers(min_i, max_i))
                j = int(rng.integers(min_j, max_j))
                if not in_bounds(i, j):
                    continue
                if reserved[max(0, i - 1) : min(h, i + 2), max(0, j - 1) : min(w, j + 2)].any():
                    continue
                grid[i, j] = name
                reserve_3x3(i, j)
                placed += 1

        # Place agents in remaining empty cells
        if isinstance(self._config.agents, int):
            num_agents = self._config.agents
        else:
            num_agents = sum(self._config.agents.values())

        empties = np.argwhere((grid == "empty") & (~reserved))
        rng.shuffle(empties)
        for k in range(min(num_agents, len(empties))):
            i, j = map(int, empties[k])
            grid[i, j] = "agent.agent"

        return GameMap(grid)
