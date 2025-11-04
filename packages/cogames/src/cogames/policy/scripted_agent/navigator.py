"""Clean navigation system for scripted agent."""

import logging
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NavigationResult:
    """Result of a navigation attempt."""

    next_step: Optional[Tuple[int, int]]  # Next cell to move to (or None if stuck)
    is_adjacent: bool  # True if target is adjacent (ready to use)
    method: str  # "adjacent", "astar", "bfs", "greedy", "stuck"
    path: Optional[Tuple[Tuple[int, int], ...]] = None  # Full path including start and destination


class Navigator:
    """Handles all pathfinding and navigation logic."""

    # Occupancy states
    OCC_UNKNOWN = 0
    OCC_FREE = 1
    OCC_OBSTACLE = 2  # Any unwalkable object (walls, stations, etc)

    def __init__(self, map_height: int, map_width: int):
        self.height = map_height
        self.width = map_width

    def navigate_to(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        occupancy_map: list,
        optimistic: bool = True,
        use_astar: bool = True,
        astar_threshold: int = 15,
    ) -> NavigationResult:
        """Find next step to move from start toward target.

        Args:
            start: Current position (row, col)
            target: Target position (row, col)
            occupancy_map: 2D list of OCC_UNKNOWN/FREE/OBSTACLE
            optimistic: Treat unknown cells as free when pathfinding
            use_astar: Use A* for long distances (faster)
            astar_threshold: Distance threshold to switch from BFS to A*

        Returns:
            NavigationResult with next step and metadata
        """
        sr, sc = start
        tr, tc = target

        # Calculate Manhattan distance
        dist = abs(tr - sr) + abs(tc - sc)

        # Adjacent - ready to move into target (extractors/stations require moving INTO them)
        if dist == 1:
            logger.info(f"[Navigator] Distance=1: {start}→{target}, returning is_adjacent=True")
            return NavigationResult(
                next_step=target,
                is_adjacent=True,
                method="adjacent",
                path=(start, target),
            )

        # If target is an obstacle (station/extractor/wall), pathfind to adjacent cells instead
        # This is crucial because BFS/A* can't pathfind TO an obstacle
        target_is_obstacle = not self._is_walkable(tr, tc, occupancy_map, optimistic)

        if target_is_obstacle:
            # Find walkable adjacent cells to the target
            adjacent_cells = [
                (nr, nc) for nr, nc in self._neighbors4(tr, tc) if self._is_walkable(nr, nc, occupancy_map, optimistic)
            ]

            if not adjacent_cells:
                logger.warning(f"[Navigator] Target {target} has no walkable adjacent cells")
                return NavigationResult(next_step=None, is_adjacent=False, method="stuck")

            # Check if we're ALREADY at one of the adjacent cells - if so, we're adjacent!
            if start in adjacent_cells:
                logger.debug(f"[Navigator] Already adjacent to obstacle target {target}, can use it")
                return NavigationResult(next_step=target, is_adjacent=True, method="adjacent")

            # Sort by distance from start - try closest first
            adjacent_cells.sort(key=lambda p: abs(p[0] - sr) + abs(p[1] - sc))

            # Try pathfinding to each adjacent cell
            for adj_target in adjacent_cells:
                if use_astar and dist >= astar_threshold:
                    logger.debug(f"[Navigator] Using A* for {start}→{adj_target} (adjacent to {target})")
                    path = self._astar(start, adj_target, occupancy_map, optimistic)
                    if path:
                        return NavigationResult(
                            next_step=path[1] if len(path) > 1 else None,
                            is_adjacent=False,
                            method="astar",
                            path=tuple(path),
                        )
                    logger.debug(f"[Navigator] A* failed for {start}→{adj_target}")
                else:
                    logger.debug(f"[Navigator] Using BFS for {start}→{adj_target} (adjacent to {target})")
                    path = self._bfs(start, adj_target, occupancy_map, optimistic)
                    if path:
                        return NavigationResult(
                            next_step=path[1] if len(path) > 1 else None,
                            is_adjacent=False,
                            method="bfs",
                            path=tuple(path),
                        )
                    logger.debug(f"[Navigator] BFS failed for {start}→{adj_target}")

            logger.warning(
                f"[Navigator] BFS/A* failed to all {len(adjacent_cells)} adjacent cells of {target}, trying greedy"
            )
        else:
            # Target is walkable - path directly to it
            if use_astar and dist >= astar_threshold:
                logger.debug(f"[Navigator] Using A* for {start}→{target} (dist={dist} >= {astar_threshold})")
                path = self._astar(start, target, occupancy_map, optimistic)
                if path:
                    return NavigationResult(
                        next_step=path[1] if len(path) > 1 else None,
                        is_adjacent=False,
                        method="astar",
                        path=tuple(path),
                    )
                logger.debug("[Navigator] A* failed, trying greedy")
            else:
                logger.debug(f"[Navigator] Using BFS for {start}→{target} (dist={dist} < {astar_threshold})")
                path = self._bfs(start, target, occupancy_map, optimistic)
                if path:
                    return NavigationResult(
                        next_step=path[1] if len(path) > 1 else None,
                        is_adjacent=False,
                        method="bfs",
                        path=tuple(path),
                    )
                logger.debug("[Navigator] BFS failed, trying greedy")

        # Pathfinding failed - try greedy movement
        next_step = self._greedy_step(start, target, occupancy_map, optimistic)
        if next_step:
            logger.debug(f"[Navigator] Greedy succeeded: {start}→{next_step} toward {target}")
            return NavigationResult(
                next_step=next_step,
                is_adjacent=False,
                method="greedy",
                path=(start, next_step),
            )

        # Completely stuck
        logger.warning(f"[Navigator] STUCK at {start}, cannot reach {target} (dist={dist})")
        return NavigationResult(next_step=None, is_adjacent=False, method="stuck", path=None)

    def _is_walkable(self, r: int, c: int, occupancy_map: list, optimistic: bool) -> bool:
        """Check if a cell is walkable (not OBSTACLE)."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        cell = occupancy_map[r][c]
        if cell == self.OCC_FREE:
            return True
        if cell == self.OCC_UNKNOWN and optimistic:
            return True
        # OBSTACLE is not walkable
        return False

    def _neighbors4(self, r: int, c: int) -> list[Tuple[int, int]]:
        """Get 4-connected neighbors."""
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

    def _bfs(
        self, start: Tuple[int, int], goal: Tuple[int, int], occupancy_map: list, optimistic: bool
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS pathfinding - returns complete path from start to goal."""
        if start == goal:
            return [start]

        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        queue = deque([start])
        max_iterations = self.height * self.width  # Safety limit
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            r, c = queue.popleft()

            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in parent:
                    continue
                if not self._is_walkable(nr, nc, occupancy_map, optimistic):
                    continue

                parent[(nr, nc)] = (r, c)

                if (nr, nc) == goal:
                    path = self._reconstruct_path(parent, goal)
                    logger.debug(f"[BFS] Found path {start}→{goal}: length={len(path)}, visited={len(parent)} cells")
                    return path

                queue.append((nr, nc))

        # No path found
        if iterations >= max_iterations:
            logger.warning(f"[BFS] Hit iteration limit searching {start}→{goal}")
        else:
            logger.debug(
                f"[BFS] No path {start}→{goal}: visited={len(parent)} cells, "
                f"opt={optimistic}, queue_empty={len(queue) == 0}"
            )
        return None

    def _astar(
        self, start: Tuple[int, int], goal: Tuple[int, int], occupancy_map: list, optimistic: bool
    ) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding - returns complete path from start to goal."""
        if start == goal:
            return [start]

        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        heap = [(heuristic(start), 0, start)]  # (f_score, g_score, position)
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        max_iterations = self.height * self.width
        iterations = 0

        while heap and iterations < max_iterations:
            iterations += 1
            f_score, g, current = heappop(heap)

            if current == goal:
                path = self._reconstruct_path(parent, goal)
                logger.debug(f"[A*] Found path {start}→{goal}: length={len(path)}, visited={len(parent)} cells")
                return path

            if g > g_score.get(current, float("inf")):
                continue

            r, c = current
            for nr, nc in self._neighbors4(r, c):
                if not self._is_walkable(nr, nc, occupancy_map, optimistic):
                    continue

                tentative_g = g + 1
                if tentative_g < g_score.get((nr, nc), float("inf")):
                    parent[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    heappush(heap, (tentative_g + heuristic((nr, nc)), tentative_g, (nr, nc)))

        # No path found
        if iterations >= max_iterations:
            logger.warning(f"[A*] Hit iteration limit searching {start}→{goal}")
        else:
            logger.debug(
                f"[A*] No path {start}→{goal}: visited={len(parent)} cells, "
                f"opt={optimistic}, heap_empty={len(heap) == 0}"
            )
        return None

    def _greedy_step(
        self, start: Tuple[int, int], goal: Tuple[int, int], occupancy_map: list, optimistic: bool
    ) -> Optional[Tuple[int, int]]:
        """Greedy step - move to adjacent cell closest to goal.

        Args:
            optimistic: If True, treat unknown cells as walkable (same as free)

        Returns:
            Next cell to move to, or None if completely blocked
        """
        sr, sc = start
        gr, gc = goal
        current_dist = abs(gr - sr) + abs(gc - sc)

        best_step = None
        best_dist = current_dist
        fallback_step = None  # Best walkable neighbor regardless of distance
        fallback_dist = float("inf")

        candidates = []
        for nr, nc in self._neighbors4(sr, sc):
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue

            cell = occupancy_map[nr][nc]
            new_dist = abs(gr - nr) + abs(gc - nc)
            cell_name = ["UNK", "FREE", "OBSTACLE"][cell] if cell <= 2 else f"?{cell}"

            # When optimistic, treat unknown cells as walkable (same as free)
            is_walkable = cell == self.OCC_FREE or (cell == self.OCC_UNKNOWN and optimistic)

            candidates.append((nr, nc, new_dist, cell_name, is_walkable))

            if is_walkable:
                # Track best improving move
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_step = (nr, nc)

                # Track best walkable neighbor (for fallback when stuck)
                if new_dist < fallback_dist:
                    fallback_dist = new_dist
                    fallback_step = (nr, nc)

        # If no improving move, take best walkable neighbor (allows escaping local minima)
        if best_step is None and fallback_step is not None:
            logger.debug(
                f"[Navigator] Greedy stuck at {start}, taking sideways/back move to {fallback_step} "
                f"(dist {current_dist}→{fallback_dist})"
            )
            return fallback_step

        if best_step is None and candidates:
            logger.warning(
                f"[Navigator] Greedy failed at {start} → {goal} (dist={current_dist}, opt={optimistic}): "
                f"candidates={candidates}"
            )

        return best_step

    def _reconstruct_path(
        self, parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        path = [goal]
        node = goal
        while True:
            prev = parent.get(node)
            if prev is None:
                break
            path.append(prev)
            node = prev
        path.reverse()
        return path
