from typing import Tuple, List, Dict, Any
import numpy as np
from omegaconf import DictConfig

def rectangles_overlap(rect1: Tuple[int, int, int, int],
                       rect2: Tuple[int, int, int, int]) -> bool:
    """
    Checks if two rectangles overlap.
    Each rectangle is a tuple (x0, y0, x1, y1) where (x0, y0) is the top-left and (x1, y1) is the bottom-right.
    """
    x0, y0, x1, y1 = rect1
    a0, b0, a1, b1 = rect2
    return not (x1 < a0 or a1 < x0 or y1 < b0 or b1 < y0)

def build_many_cylinder(
    width: int,
    height: int,
    num_agents: int = 1,
    num_cylinders: int = 10,
    border_width: int = 1,
    cylinder_params: DictConfig = None,
    seed: Any = None,
    **kwargs,
) -> np.ndarray:
    """
    Builds a grid environment with many cylinders randomly placed.
    
    Each cylinder is placed with a random orientation (horizontal or vertical). Its position is chosen so that it fits
    inside the overall grid (and attempts are made to avoid overlapping already-placed cylinders). One end is sealed,
    with the door (an opening) on the opposite side. Inside each cylinder a single special resource is placed – one
    randomly chosen from ["heart altar", "generator", "convertor"].
    
    Finally, agents are placed in random empty cells.
    """
    if cylinder_params is None:
        cylinder_params = {"length": 15, "cylinder_width": 10}

    rng = np.random.default_rng(seed)
    grid = np.full((height, width), "empty", dtype='<U50')
    placed_rectangles: List[Tuple[int, int, int, int]] = []
    resources = ["heart altar", "generator", "convertor"]

    for _ in range(num_cylinders):
        # Choose orientation.
        horizontal = rng.choice([True, False])
        if horizontal:
            cyl_len = cylinder_params["length"]
            cyl_wid = cylinder_params["cylinder_width"]
        else:
            # For vertical cylinders, swap the dimensions.
            cyl_len = cylinder_params["cylinder_width"]
            cyl_wid = cylinder_params["length"]

        # Attempt to find a placement that doesn't overlap.
        placed = False
        for _ in range(10):
            max_x = width - cyl_len
            max_y = height - cyl_wid
            if max_x < 0 or max_y < 0:
                # Cylinder dimensions are larger than the grid.
                return grid
            cx = rng.integers(0, max_x + 1)
            cy = rng.integers(0, max_y + 1)
            new_rect = (cx, cy, cx + cyl_len - 1, cy + cyl_wid - 1)
            if not any(rectangles_overlap(new_rect, r) for r in placed_rectangles):
                placed_rectangles.append(new_rect)
                placed = True
                break
        if not placed:
            # Skip this cylinder if no suitable placement was found.
            continue

        # Randomly choose a sealed end.
        if horizontal:
            sealed_end = rng.choice(["left", "right"])
        else:
            sealed_end = rng.choice(["top", "bottom"])

        # Determine door coordinates (door is on the side opposite to the sealed end).
        if horizontal:
            if sealed_end == "left":
                door_x = cx + cyl_len - border_width  # door on right side
            else:
                door_x = cx  # door on left side
            door_y = cy + cyl_wid // 2
        else:
            if sealed_end == "top":
                door_y = cy + cyl_wid - border_width  # door on bottom side
            else:
                door_y = cy  # door on top side
            door_x = cx + cyl_len // 2

        # Draw the cylinder border.
        for y in range(cy, cy + cyl_wid):
            for x in range(cx, cx + cyl_len):
                if (x < cx + border_width or 
                    x > cx + cyl_len - border_width - 1 or 
                    y < cy + border_width or 
                    y > cy + cyl_wid - border_width - 1):
                    # Leave door cell open.
                    if (x, y) == (door_x, door_y):
                        grid[y, x] = "door"
                    else:
                        grid[y, x] = "wall"
        # Determine interior cells.
        interior = []
        for y in range(cy + border_width, cy + cyl_wid - border_width):
            for x in range(cx + border_width, cx + cyl_len - border_width):
                if grid[y, x] == "empty":
                    interior.append((x, y))
        # Place one special resource randomly inside the cylinder.
        if interior:
            chosen_cell = rng.choice(interior)
            resource = rng.choice(resources)
            grid[chosen_cell[1], chosen_cell[0]] = resource

    # --- Place agents ---
    empty_cells = [(x, y) for y in range(height) for x in range(width) if grid[y, x] == "empty"]
    if empty_cells:
        num_to_place = min(num_agents, len(empty_cells))
        chosen_indices = rng.choice(len(empty_cells), size=num_to_place, replace=False)
        for idx in chosen_indices:
            pos = empty_cells[idx]
            grid[pos[1], pos[0]] = "agent.agent"

    return grid

class ManyCylinder:
    """
    A lightweight map builder for a single-room environment populated with many randomly oriented and placed cylinders.
    Each cylinder contains exactly one special resource.
    """
    def __init__(self,
                 width: int,
                 height: int,
                 num_agents: int = 1,
                 num_cylinders: int = 10,
                 border_width: int = 1,
                 cylinder_params: DictConfig = None,
                 seed: Any = None,
                 **kwargs):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_cylinders = num_cylinders
        self.border_width = border_width
        self.cylinder_params = cylinder_params if cylinder_params is not None else {"length": 15, "cylinder_width": 10}
        self.seed = seed
        self.kwargs = kwargs

    def build(self) -> np.ndarray:
        return build_many_cylinder(
            width=self.width,
            height=self.height,
            num_agents=self.num_agents,
            num_cylinders=self.num_cylinders,
            border_width=self.border_width,
            cylinder_params=self.cylinder_params,
            seed=self.seed,
            **self.kwargs,
        )
