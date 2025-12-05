"""Rigorous mission difficulty estimation using actual map generation.

Generates the map and computes ground truth metrics via pathfinding.
Use this to validate and calibrate the lightweight estimator.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import Mission
    from mettagrid.mapgen.types import MapGrid


@dataclass
class ObjectPositions:
    """Positions of all relevant objects on the map."""

    spawns: list[tuple[int, int]] = field(default_factory=list)
    assembler: tuple[int, int] | None = None
    chest: tuple[int, int] | None = None
    chargers: list[tuple[int, int]] = field(default_factory=list)
    carbon_extractors: list[tuple[int, int]] = field(default_factory=list)
    oxygen_extractors: list[tuple[int, int]] = field(default_factory=list)
    germanium_extractors: list[tuple[int, int]] = field(default_factory=list)
    silicon_extractors: list[tuple[int, int]] = field(default_factory=list)
    walls: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class PathfindingResult:
    """Result of pathfinding from a source to targets."""

    distances: dict[tuple[int, int], int]  # position -> shortest distance
    unreachable: list[tuple[int, int]]  # positions that cannot be reached


@dataclass
class ChargerCoverage:
    """Analysis of charger coverage on the map."""

    coverage_map: np.ndarray  # bool array: True if tile can reach charger in time
    gap_tiles: int  # number of tiles where agent could get stranded
    coverage_ratio: float  # fraction of walkable tiles that are safe
    max_gap_distance: int  # farthest any tile is from nearest charger


@dataclass
class OptimalPath:
    """Optimal path to collect resources and produce a heart."""

    total_distance: int  # total movement steps
    path_sequence: list[str]  # sequence of locations visited
    charger_visits: int  # number of charger visits needed
    feasible: bool  # whether the path is actually achievable


@dataclass
class RigorousReport:
    """Ground truth difficulty report from actual map generation."""

    # Map metadata
    map_width: int
    map_height: int
    seed: int | None

    # Object positions
    objects: ObjectPositions

    # Accessibility
    all_extractors_reachable: bool
    assembler_reachable: bool
    unreachable_objects: list[str]

    # Distance metrics (from spawn)
    spawn_to_assembler: int | None
    spawn_to_chest: int | None
    spawn_to_charger: int | None
    spawn_to_extractors: dict[str, int]  # resource type -> min distance
    avg_extractor_distance: float

    # Charger coverage (for zero-regen scenarios)
    charger_coverage: ChargerCoverage | None

    # Optimal path analysis
    optimal_path: OptimalPath | None

    # Comparison metrics
    estimated_total_steps: int  # ground truth total steps
    estimated_difficulty: float  # ground truth difficulty score

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Rigorous Difficulty Report ===",
            f"Map: {self.map_width}x{self.map_height}",
            f"",
            f"Object Counts:",
            f"  Spawns: {len(self.objects.spawns)}",
            f"  Chargers: {len(self.objects.chargers)}",
            f"  Carbon Extractors: {len(self.objects.carbon_extractors)}",
            f"  Oxygen Extractors: {len(self.objects.oxygen_extractors)}",
            f"  Germanium Extractors: {len(self.objects.germanium_extractors)}",
            f"  Silicon Extractors: {len(self.objects.silicon_extractors)}",
            f"",
            f"Distances from Spawn:",
            f"  To Assembler: {self.spawn_to_assembler}",
            f"  To Charger: {self.spawn_to_charger}",
            f"  To Extractors: {self.spawn_to_extractors}",
            f"  Avg Extractor Distance: {self.avg_extractor_distance:.1f}",
            f"",
            f"Accessibility:",
            f"  All Extractors Reachable: {self.all_extractors_reachable}",
            f"  Assembler Reachable: {self.assembler_reachable}",
        ]

        if self.unreachable_objects:
            lines.append(f"  ⚠️  Unreachable: {', '.join(self.unreachable_objects)}")

        if self.charger_coverage:
            cc = self.charger_coverage
            lines.extend([
                f"",
                f"Charger Coverage:",
                f"  Coverage Ratio: {cc.coverage_ratio:.1%}",
                f"  Gap Tiles: {cc.gap_tiles}",
                f"  Max Gap Distance: {cc.max_gap_distance}",
            ])

        if self.optimal_path:
            op = self.optimal_path
            lines.extend([
                f"",
                f"Optimal Path:",
                f"  Total Distance: {op.total_distance}",
                f"  Charger Visits: {op.charger_visits}",
                f"  Feasible: {op.feasible}",
                f"  Sequence: {' → '.join(op.path_sequence[:10])}{'...' if len(op.path_sequence) > 10 else ''}",
            ])

        lines.extend([
            f"",
            f"Ground Truth:",
            f"  Estimated Steps: {self.estimated_total_steps}",
            f"  Difficulty Score: {self.estimated_difficulty:.3f}",
        ])

        return "\n".join(lines)


def estimate_difficulty_rigorous(mission: Mission, seed: int | None = None) -> RigorousReport:
    """Estimate difficulty by actually generating the map.

    Args:
        mission: Mission instance (with variants applied)
        seed: Random seed for map generation (None for random)

    Returns:
        RigorousReport with ground truth metrics
    """
    # Generate the actual map
    env_config = mission.make_env()
    map_builder = env_config.game.map_builder

    # Set seed if provided
    if seed is not None and hasattr(map_builder, "seed"):
        map_builder.seed = seed

    # Build the map
    builder = map_builder.create()
    num_agents = mission.num_cogs if mission.num_cogs else mission.site.min_cogs
    game_map = builder.build_for_num_agents(num_agents)
    grid = game_map.grid

    height, width = grid.shape

    # Extract object positions
    objects = _extract_object_positions(grid)

    # Find primary spawn (first one)
    spawn = objects.spawns[0] if objects.spawns else None

    # Compute distances from spawn
    if spawn:
        distances = _bfs_all_distances(grid, spawn, objects.walls)
    else:
        distances = {}

    # Calculate specific distances
    spawn_to_assembler = distances.get(objects.assembler) if objects.assembler else None
    spawn_to_chest = distances.get(objects.chest) if objects.chest else None
    spawn_to_charger = min(
        (distances.get(c, float("inf")) for c in objects.chargers),
        default=None
    )
    if spawn_to_charger == float("inf"):
        spawn_to_charger = None

    spawn_to_extractors = {}
    unreachable = []

    for resource, extractors in [
        ("carbon", objects.carbon_extractors),
        ("oxygen", objects.oxygen_extractors),
        ("germanium", objects.germanium_extractors),
        ("silicon", objects.silicon_extractors),
    ]:
        if extractors:
            min_dist = min(distances.get(e, float("inf")) for e in extractors)
            if min_dist == float("inf"):
                unreachable.append(f"{resource}_extractor")
                spawn_to_extractors[resource] = -1
            else:
                spawn_to_extractors[resource] = int(min_dist)
        else:
            unreachable.append(f"{resource}_extractor (none placed)")
            spawn_to_extractors[resource] = -1

    # Check accessibility
    all_extractors_reachable = len([d for d in spawn_to_extractors.values() if d >= 0]) == 4
    assembler_reachable = spawn_to_assembler is not None and spawn_to_assembler != float("inf")

    if not assembler_reachable:
        unreachable.append("assembler")

    # Calculate average extractor distance
    valid_distances = [d for d in spawn_to_extractors.values() if d >= 0]
    avg_extractor_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0.0

    # Charger coverage analysis (for zero-regen scenarios)
    energy_analysis = _analyze_energy_for_coverage(mission)
    charger_coverage = None
    if energy_analysis["regen"] == 0 and objects.chargers:
        charger_coverage = _compute_charger_coverage(
            grid, objects, energy_analysis["capacity"], energy_analysis["move_cost"]
        )

    # Optimal path analysis
    optimal_path = _compute_optimal_path(
        mission, grid, objects, distances, spawn, energy_analysis
    )

    # Compute ground truth difficulty
    estimated_total_steps = optimal_path.total_distance if optimal_path else float("inf")
    estimated_difficulty = estimated_total_steps / 1000 if estimated_total_steps != float("inf") else float("inf")

    return RigorousReport(
        map_width=width,
        map_height=height,
        seed=seed,
        objects=objects,
        all_extractors_reachable=all_extractors_reachable,
        assembler_reachable=assembler_reachable,
        unreachable_objects=unreachable,
        spawn_to_assembler=spawn_to_assembler,
        spawn_to_chest=spawn_to_chest,
        spawn_to_charger=int(spawn_to_charger) if spawn_to_charger else None,
        spawn_to_extractors=spawn_to_extractors,
        avg_extractor_distance=avg_extractor_distance,
        charger_coverage=charger_coverage,
        optimal_path=optimal_path,
        estimated_total_steps=int(estimated_total_steps) if estimated_total_steps != float("inf") else -1,
        estimated_difficulty=estimated_difficulty,
    )


def _extract_object_positions(grid: MapGrid) -> ObjectPositions:
    """Extract positions of all relevant objects from the grid."""
    objects = ObjectPositions()
    height, width = grid.shape

    for y in range(height):
        for x in range(width):
            cell = grid[y, x]
            pos = (y, x)

            if cell.startswith("agent"):
                objects.spawns.append(pos)
            elif cell == "wall":
                objects.walls.add(pos)
            elif "assembler" in cell:
                objects.assembler = pos
            elif "chest" in cell:
                objects.chest = pos
            elif "charger" in cell:
                objects.chargers.append(pos)
            elif "carbon" in cell and "extractor" in cell:
                objects.carbon_extractors.append(pos)
            elif "oxygen" in cell and "extractor" in cell:
                objects.oxygen_extractors.append(pos)
            elif "germanium" in cell and "extractor" in cell:
                objects.germanium_extractors.append(pos)
            elif "silicon" in cell and "extractor" in cell:
                objects.silicon_extractors.append(pos)

    return objects


def _bfs_all_distances(
    grid: MapGrid,
    start: tuple[int, int],
    walls: set[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    """BFS to compute distances from start to all reachable positions."""
    height, width = grid.shape
    distances = {start: 0}
    queue = deque([start])

    while queue:
        y, x = queue.popleft()
        current_dist = distances[(y, x)]

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                pos = (ny, nx)
                if pos not in distances and pos not in walls:
                    distances[pos] = current_dist + 1
                    queue.append(pos)

    return distances


def _analyze_energy_for_coverage(mission: Mission) -> dict:
    """Extract energy parameters from mission."""
    return {
        "capacity": mission.energy_capacity,
        "regen": mission.energy_regen_amount,
        "move_cost": mission.move_energy_cost,
        "charger_efficiency": mission.charger.efficiency,
    }


def _compute_charger_coverage(
    grid: MapGrid,
    objects: ObjectPositions,
    capacity: int,
    move_cost: int
) -> ChargerCoverage:
    """Compute which tiles can safely reach a charger before energy death."""
    height, width = grid.shape

    # Max steps before energy death (with zero regen)
    max_steps = capacity // move_cost

    # BFS from each charger to find tiles within range
    coverage_distances = np.full((height, width), float("inf"))

    for charger_pos in objects.chargers:
        charger_distances = _bfs_all_distances(grid, charger_pos, objects.walls)
        for pos, dist in charger_distances.items():
            y, x = pos
            coverage_distances[y, x] = min(coverage_distances[y, x], dist)

    # A tile is "safe" if it can reach a charger within max_steps
    coverage_map = coverage_distances <= max_steps

    # Count walkable tiles (not walls)
    walkable = np.ones((height, width), dtype=bool)
    for wall in objects.walls:
        walkable[wall] = False

    # Gap tiles: walkable but not covered
    gap_mask = walkable & ~coverage_map
    gap_tiles = np.sum(gap_mask)

    # Coverage ratio
    walkable_count = np.sum(walkable)
    coverage_ratio = np.sum(coverage_map & walkable) / walkable_count if walkable_count > 0 else 0.0

    # Max gap distance
    max_gap_distance = int(np.max(coverage_distances[walkable])) if walkable_count > 0 else 0

    return ChargerCoverage(
        coverage_map=coverage_map,
        gap_tiles=int(gap_tiles),
        coverage_ratio=float(coverage_ratio),
        max_gap_distance=max_gap_distance,
    )


def _compute_optimal_path(
    mission: Mission,
    grid: MapGrid,
    objects: ObjectPositions,
    spawn_distances: dict[tuple[int, int], int],
    spawn: tuple[int, int] | None,
    energy: dict
) -> OptimalPath | None:
    """Compute optimal path to collect resources and produce one heart.

    Uses a greedy TSP-like approach: visit nearest unvisited extractor,
    account for charger visits when needed.
    """
    if not spawn or not objects.assembler:
        return None

    # Get resource requirements from mission
    from cogames.cogs_vs_clips.difficulty_estimator import _analyze_resources

    resources = _analyze_resources(mission)

    # Determine how many visits to each extractor type
    visits_needed = {}
    for resource, need in resources.heart_cost.items():
        output = resources.extractor_output.get(resource, 1)
        visits_needed[resource] = math.ceil(need / output) if output > 0 else 0

    # Get extractor positions by type
    extractor_positions = {
        "carbon": objects.carbon_extractors,
        "oxygen": objects.oxygen_extractors,
        "germanium": objects.germanium_extractors,
        "silicon": objects.silicon_extractors,
    }

    # Build distance cache between all relevant points
    relevant_points = {spawn, objects.assembler}
    for extractors in extractor_positions.values():
        relevant_points.update(extractors)
    if objects.chargers:
        relevant_points.update(objects.chargers)
    if objects.chest:
        relevant_points.add(objects.chest)

    # Compute pairwise distances
    pairwise_distances = {}
    for point in relevant_points:
        if point:
            distances = _bfs_all_distances(grid, point, objects.walls)
            pairwise_distances[point] = distances

    # Greedy path construction
    current_pos = spawn
    total_distance = 0
    path_sequence = ["spawn"]
    visits_remaining = visits_needed.copy()

    # Energy tracking
    capacity = energy["capacity"]
    regen = energy["regen"]
    move_cost = energy["move_cost"]
    charger_output = 50 * energy["charger_efficiency"] // 100
    current_energy = capacity
    charger_visits = 0

    def get_distance(from_pos, to_pos):
        if from_pos in pairwise_distances:
            return pairwise_distances[from_pos].get(to_pos, float("inf"))
        return float("inf")

    def find_nearest_charger(from_pos):
        if not objects.chargers:
            return None, float("inf")
        nearest = None
        nearest_dist = float("inf")
        for charger in objects.chargers:
            dist = get_distance(from_pos, charger)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = charger
        return nearest, nearest_dist

    # Visit extractors (greedy nearest neighbor)
    while any(v > 0 for v in visits_remaining.values()):
        # Find nearest extractor that still needs visiting
        best_extractor = None
        best_resource = None
        best_distance = float("inf")

        for resource, remaining in visits_remaining.items():
            if remaining > 0:
                for ext_pos in extractor_positions[resource]:
                    dist = get_distance(current_pos, ext_pos)
                    if dist < best_distance:
                        best_distance = dist
                        best_extractor = ext_pos
                        best_resource = resource

        if best_extractor is None or best_distance == float("inf"):
            # Can't reach remaining extractors
            return OptimalPath(
                total_distance=total_distance,
                path_sequence=path_sequence,
                charger_visits=charger_visits,
                feasible=False,
            )

        # Check if we have enough energy to reach it
        energy_needed = int(best_distance) * move_cost
        energy_after_move = current_energy - energy_needed + int(best_distance) * regen

        if energy_after_move < 0 and regen == 0:
            # Need charger visit first
            charger, charger_dist = find_nearest_charger(current_pos)
            if charger is None:
                return OptimalPath(
                    total_distance=total_distance,
                    path_sequence=path_sequence,
                    charger_visits=charger_visits,
                    feasible=False,
                )

            # Go to charger
            total_distance += int(charger_dist)
            current_pos = charger
            current_energy = capacity  # Recharged
            charger_visits += 1
            path_sequence.append("charger")

            # Recalculate distance to extractor
            best_distance = get_distance(current_pos, best_extractor)

        # Move to extractor
        total_distance += int(best_distance)
        current_energy = current_energy - int(best_distance) * move_cost + int(best_distance) * regen
        current_energy = min(current_energy, capacity)
        current_pos = best_extractor
        visits_remaining[best_resource] -= 1
        path_sequence.append(f"{best_resource}")

    # Go to chest (if exists) to deposit, then assembler
    if objects.chest:
        chest_dist = get_distance(current_pos, objects.chest)
        if chest_dist != float("inf"):
            total_distance += int(chest_dist)
            current_pos = objects.chest
            path_sequence.append("chest")

    # Go to assembler
    assembler_dist = get_distance(current_pos, objects.assembler)
    if assembler_dist == float("inf"):
        return OptimalPath(
            total_distance=total_distance,
            path_sequence=path_sequence,
            charger_visits=charger_visits,
            feasible=False,
        )

    total_distance += int(assembler_dist)
    path_sequence.append("assembler")

    # Add waiting time if energy-constrained with positive regen
    if regen > 0 and not (regen >= move_cost):
        # Calculate total energy needed vs budget
        total_energy_needed = total_distance * move_cost + resources.extraction_energy_cost
        energy_from_regen = total_distance * regen
        deficit = total_energy_needed - capacity - energy_from_regen
        if deficit > 0:
            wait_steps = math.ceil(deficit / regen)
            total_distance += wait_steps
            path_sequence.append(f"wait({wait_steps})")

    return OptimalPath(
        total_distance=total_distance,
        path_sequence=path_sequence,
        charger_visits=charger_visits,
        feasible=True,
    )


def compare_estimators(mission: Mission, seed: int | None = None) -> dict:
    """Compare lightweight vs rigorous estimators.

    Returns:
        Dictionary with both reports and comparison metrics.
    """
    from cogames.cogs_vs_clips.difficulty_estimator import estimate_difficulty

    lightweight = estimate_difficulty(mission)
    rigorous = estimate_difficulty_rigorous(mission, seed=seed)

    # Comparison metrics
    lightweight_steps = lightweight.steady_state_steps
    rigorous_steps = rigorous.estimated_total_steps

    if rigorous_steps > 0:
        step_ratio = lightweight_steps / rigorous_steps
        difficulty_ratio = lightweight.difficulty_score / rigorous.estimated_difficulty
    else:
        step_ratio = float("inf")
        difficulty_ratio = float("inf")

    return {
        "lightweight": lightweight,
        "rigorous": rigorous,
        "comparison": {
            "lightweight_steps": lightweight_steps,
            "rigorous_steps": rigorous_steps,
            "step_ratio": step_ratio,  # <1 = lightweight underestimates, >1 = overestimates
            "lightweight_difficulty": lightweight.difficulty_score,
            "rigorous_difficulty": rigorous.estimated_difficulty,
            "difficulty_ratio": difficulty_ratio,
            "avg_distance_lightweight": lightweight.spatial.estimated_avg_distance if lightweight.spatial else 0,
            "avg_distance_rigorous": rigorous.avg_extractor_distance,
        }
    }

