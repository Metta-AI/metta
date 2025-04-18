# This code is a Python translation of https://github.com/mxgmn/WaveFunctionCollapse, with some modifications.
#
# Original work Copyright (c) 2016 Maxim Gumin
#
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import heapq
import logging
import time
from typing import Any, List, Literal, Optional

import numpy as np

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene
from mettagrid.map.utils.pattern import Symmetry, ascii_to_patterns_with_counts
from mettagrid.map.utils.random import MaybeSeed

logger = logging.getLogger(__name__)

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

NextNodeHeuristic = Literal["scanline", "mrv", "entropy"]


def opposite_direction(d: int) -> int:
    return (d + 2) % 4


class WFC(Scene):
    def __init__(
        self,
        pattern: str,
        pattern_size: int = 3,
        next_node_heuristic: NextNodeHeuristic = "entropy",
        periodic_input: bool = True,
        symmetry: Symmetry = "all",
        attempts: int = 3,
        seed: MaybeSeed = None,
        children: Optional[List[Any]] = None,
    ):
        super().__init__(children=children)

        self._ascii_pattern = pattern
        self._pattern_size = pattern_size
        self._rng = np.random.default_rng(seed)
        patterns_with_counts = ascii_to_patterns_with_counts(
            self._ascii_pattern, pattern_size, periodic=periodic_input, symmetry=symmetry
        )

        self._weights = np.array([p[1] for p in patterns_with_counts], dtype=np.float_)
        self._patterns = [p[0] for p in patterns_with_counts]
        self._pattern_count = len(self._weights)

        self._sum_of_weights = np.sum(self._weights)
        self._sum_of_weight_log_weights = np.sum(self._weights * np.log(self._weights))

        self._starting_entropy = np.log(self._sum_of_weights) - self._sum_of_weight_log_weights / self._sum_of_weights

        self._next_node_heuristic = next_node_heuristic
        self._attempts = attempts

        self._fill_propagator()

    def _fill_propagator(self):
        """
        Here we pre-calculate pattern compatibility, and we'll use this to check the compatibility of potential
        patterns going forward, when we exclude some possibility from the wave function and want to propagate
        the change.
        """
        self._propagator = []  # no point in using numpy arrays here (I tested it, it's slower)

        self._propagator_lengths = np.zeros((4, self._pattern_count), dtype=np.int_)

        for d in range(4):
            d_propagator = []
            for t in range(self._pattern_count):
                compatibles = []
                for t2 in range(self._pattern_count):
                    if self._patterns[t].is_compatible(self._patterns[t2], dx[d], dy[d]):
                        compatibles.append(t2)
                d_propagator.append(compatibles)
                self._propagator_lengths[d, t] = len(compatibles)
            self._propagator.append(d_propagator)

    def _render(self, node: Node):
        WFCRenderSession(self, node).run()


class WFCRenderSession:
    def __init__(self, scene: WFC, node: Node):
        self.scene = scene
        self.node = node

        self.width = self.node.width
        self.height = self.node.height
        self.weights = self.scene._weights
        self.pattern_count = len(self.weights)
        self.rng = self.scene._rng

        self.reset()

    def reset(self):
        start = time.time()
        self.wave = np.full((self.height, self.width, self.pattern_count), True, dtype=np.bool_)

        self.compatible = np.zeros((self.height, self.width, 4, self.pattern_count), dtype=np.int_)

        for y in range(self.height):
            for x in range(self.width):
                for d in range(4):
                    self.compatible[y, x, d, :] = self.scene._propagator_lengths[opposite_direction(d), :]

        self.sums_of_ones = np.full((self.height, self.width), len(self.weights), dtype=np.int_)
        self.sums_of_weights = np.full((self.height, self.width), self.scene._sum_of_weights, dtype=np.float_)
        self.sums_of_weight_log_weights = np.full(
            (self.height, self.width),
            self.scene._sum_of_weight_log_weights,
            dtype=np.float_,
        )
        self.observed = 0
        self.queue = []
        for y in range(self.height):
            for x in range(self.width):
                heapq.heappush(self.queue, (self.cell_score(x, y), x, y))

        self.stack = np.zeros((self.width * self.height * self.pattern_count, 3), dtype=np.int_)
        self.stacksize = 0
        logger.debug(f"reset() time: {time.time() - start}")

        self._pick_next_time = 0
        self._pick_next_count = 0
        self._observe_time = 0
        self._propagate_time = 0

    def cell_score(self, x: int, y: int) -> float:
        if self.scene._next_node_heuristic == "mrv":
            return self.sums_of_ones[y, x]
        else:
            return self.scene._starting_entropy

    def attempt_run(self):
        while True:
            cell = self.pick_next_node()
            if cell is None:
                return True

            self.observe(cell)
            start = time.time()
            if not self.propagate():
                return False
            self._propagate_time += time.time() - start

    def run(self):
        ok = False
        for i in range(self.scene._attempts):
            logger.info(f"Attempt {i + 1} of {self.scene._attempts}, pattern:\n{self.scene._ascii_pattern}")
            start_time = time.time()
            self.reset()
            ok = self.attempt_run()
            attempt_time = time.time() - start_time
            logger.info(
                f"Attempt {i + 1} time: {attempt_time:.3f}s "
                f"(pick_next_node: {self._pick_next_time:.3f}s, "
                f"observe: {self._observe_time:.3f}s, "
                f"propagate: {self._propagate_time:.3f}s, "
                f"pick_next_count: {self._pick_next_count})"
            )
            if ok:
                break
            else:
                logger.info(f"Attempt {i + 1} failed")

        if not ok:
            raise Exception(f"Failed to generate map with pattern:\n{self.scene._ascii_pattern}")

        for y in range(self.height):
            for x in range(self.width):
                for t in range(self.pattern_count):
                    if self.wave[y, x, t]:
                        self.node.grid[y, x] = "wall" if self.scene._patterns[t].data[0][0] else "empty"

    def pick_next_node(self):
        # non-periodic
        self._pick_next_count += 1
        start = time.time()
        used_width = self.width - self.scene._pattern_size + 1
        used_height = self.height - self.scene._pattern_size + 1

        if self.scene._next_node_heuristic == "scanline":
            for i in range(self.observed, used_width * used_height):
                y = i // used_width
                x = i % used_width
                if self.sums_of_ones[y, x] > 1:
                    self.observed = i + 1
                    return (y, x)
            return None
        else:  # entropy or mrv
            while len(self.queue) > 0:
                _, x, y = heapq.heappop(self.queue)
                if self.sums_of_ones[y, x] > 1:
                    self._pick_next_time += time.time() - start
                    return (y, x)
            return None

    def observe(self, cell: tuple[int, int]):
        start = time.time()
        y, x = cell
        distribution = self.wave[y, x] * self.weights
        distribution /= np.sum(distribution)
        r = self.rng.choice(range(self.pattern_count), p=distribution)
        for t in range(self.pattern_count):
            if t != r and self.wave[y, x, t]:
                self.ban(y, x, t)
        self._observe_time += time.time() - start

    def ban(self, y: int, x: int, t: int) -> bool:
        self.wave[y, x, t] = False
        self.compatible[y, x, :, t] = 0

        self.stack[self.stacksize] = (y, x, t)
        self.stacksize += 1

        self.sums_of_ones[y, x] -= 1
        if self.sums_of_ones[y, x] == 0:
            return False

        self.sums_of_weights[y, x] -= self.weights[t]
        self.sums_of_weight_log_weights[y, x] -= self.scene._weights[t] * np.log(self.scene._weights[t])

        sum = self.sums_of_weights[y, x]
        if sum > 0:
            heapq.heappush(self.queue, (self.cell_score(x, y), x, y))

        return True

    def propagate(self):
        while self.stacksize > 0:
            y1, x1, t1 = self.stack[self.stacksize - 1]
            self.stacksize -= 1

            for d in range(4):
                y2 = y1 + dy[d]
                x2 = x1 + dx[d]
                if y2 < 0 or y2 >= self.height or x2 < 0 or x2 >= self.width:
                    continue

                dt_propagator = self.scene._propagator[d][t1]
                compat = self.compatible[y2, x2, d]
                for t2 in dt_propagator:
                    compat[t2] -= 1
                    if compat[t2] == 0:
                        ok = self.ban(y2, x2, t2)
                        if not ok:
                            return False

        return True
