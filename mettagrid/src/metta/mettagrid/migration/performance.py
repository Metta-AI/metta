"""
Performance benchmarking utilities for map format comparison.
"""

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from metta.mettagrid.map_builder.map_builder import GameMap
from metta.mettagrid.mettagrid_config import GameConfig
from metta.mettagrid.migration.map_format_converter import MapFormatConverter


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    execution_time: float
    memory_peak: Optional[int] = None
    memory_delta: Optional[int] = None
    iterations: int = 1
    error: Optional[str] = None

    @property
    def time_per_iteration(self) -> float:
        return self.execution_time / self.iterations if self.iterations > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "execution_time": self.execution_time,
            "time_per_iteration": self.time_per_iteration,
            "memory_peak": self.memory_peak,
            "memory_delta": self.memory_delta,
            "iterations": self.iterations,
            "error": self.error,
        }


class PerformanceBenchmark:
    """
    Performance benchmarking utilities for map format operations.

    This class provides tools to measure and compare the performance
    of legacy vs int-based map formats across various operations.
    """

    def __init__(self, game_config: Optional[GameConfig] = None):
        """
        Initialize the benchmarker.

        Args:
            game_config: Optional GameConfig for validation
        """
        self.game_config = game_config
        self.converter = MapFormatConverter(game_config)
        self.results: List[BenchmarkResult] = []

    @contextmanager
    def _measure_performance(self, name: str, iterations: int = 1):
        """Context manager for measuring execution time and memory usage."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        memory_before = process.memory_info().rss

        # Start timing
        start_time = time.perf_counter()

        try:
            yield
            error = None
        except Exception as e:
            error = str(e)

        # End timing
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # End memory tracking
        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_peak=peak,
            memory_delta=memory_delta,
            iterations=iterations,
            error=error,
        )

        self.results.append(result)

    def benchmark_map_creation(
        self, heights: List[int] = None, widths: List[int] = None, iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark map creation performance for different sizes.

        Args:
            heights: List of map heights to test
            widths: List of map widths to test
            iterations: Number of iterations per test

        Returns:
            Dictionary with benchmark results
        """
        if heights is None:
            heights = [10, 50, 100, 200]
        if widths is None:
            widths = [10, 50, 100, 200]

        results = {"creation": {}}

        for height in heights:
            for width in widths:
                size_key = f"{height}x{width}"
                results["creation"][size_key] = {}

                # Benchmark legacy creation
                with self._measure_performance(f"legacy_creation_{size_key}", iterations):
                    for _ in range(iterations):
                        legacy_grid = np.full((height, width), "empty", dtype=np.str_)
                        # Add some variety
                        legacy_grid[0, :] = "wall"
                        legacy_grid[-1, :] = "wall"
                        legacy_grid[:, 0] = "wall"
                        legacy_grid[:, -1] = "wall"
                        if height > 5 and width > 5:
                            legacy_grid[height // 2, width // 2] = "agent"
                        # game_map = GameMap(grid=legacy_grid)  # Not used in benchmark

                # Benchmark int creation
                with self._measure_performance(f"int_creation_{size_key}", iterations):
                    for _ in range(iterations):
                        int_grid = np.zeros((height, width), dtype=np.uint8)
                        # Add some variety
                        int_grid[0, :] = 1  # wall
                        int_grid[-1, :] = 1  # wall
                        int_grid[:, 0] = 1  # wall
                        int_grid[:, -1] = 1  # wall
                        if height > 5 and width > 5:
                            int_grid[height // 2, width // 2] = 10  # agent
                        # decoder_key = ["empty", "wall"] + [f"object_{i}" for i in range(2, 11)] + ["agent"]
                        # Not used in benchmark
                        # game_map = GameMap(grid=int_grid, decoder_key=decoder_key)  # Not used in benchmark

        return results

    def benchmark_format_conversion(self, test_maps: List[GameMap], iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark conversion performance between formats.

        Args:
            test_maps: List of GameMap instances to test
            iterations: Number of iterations per test

        Returns:
            Dictionary with benchmark results
        """
        results = {"conversion": {}}

        for i, game_map in enumerate(test_maps):
            map_key = f"map_{i}"
            results["conversion"][map_key] = {
                "size": game_map.grid.shape,
                "format": "legacy" if game_map.is_legacy() else "int",
            }

            if game_map.is_legacy():
                # Benchmark legacy -> int conversion
                with self._measure_performance(f"legacy_to_int_{map_key}", iterations):
                    for _ in range(iterations):
                        self.converter.convert_game_map_to_int(game_map)

            else:  # int-based
                # Benchmark int -> legacy conversion
                with self._measure_performance(f"int_to_legacy_{map_key}", iterations):
                    for _ in range(iterations):
                        self.converter.convert_game_map_to_legacy(game_map)

        return results

    def benchmark_memory_usage(self, map_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Compare memory usage between legacy and int-based maps.

        Args:
            map_sizes: List of (height, width) tuples to test

        Returns:
            Dictionary with memory comparison results
        """
        if map_sizes is None:
            map_sizes = [(10, 10), (50, 50), (100, 100), (200, 200)]

        results = {"memory": {}}

        for height, width in map_sizes:
            size_key = f"{height}x{width}"
            results["memory"][size_key] = {}

            # Measure legacy map memory
            legacy_grid = np.full((height, width), "empty", dtype=np.str_)
            legacy_size = legacy_grid.nbytes
            legacy_item_size = legacy_grid.itemsize

            # Measure int map memory
            int_grid = np.zeros((height, width), dtype=np.uint8)
            decoder_key = ["empty", "wall", "agent"]
            int_size = int_grid.nbytes
            int_item_size = int_grid.itemsize
            decoder_size = sum(len(s.encode("utf-8")) for s in decoder_key)
            total_int_size = int_size + decoder_size

            results["memory"][size_key] = {
                "legacy": {
                    "grid_bytes": legacy_size,
                    "item_size": legacy_item_size,
                    "total_bytes": legacy_size,
                },
                "int": {
                    "grid_bytes": int_size,
                    "item_size": int_item_size,
                    "decoder_bytes": decoder_size,
                    "total_bytes": total_int_size,
                },
                "savings": {
                    "absolute": legacy_size - total_int_size,
                    "percentage": ((legacy_size - total_int_size) / legacy_size) * 100,
                },
            }

        return results

    def benchmark_access_patterns(self, game_maps: List[GameMap], iterations: int = 1000) -> Dict[str, Any]:
        """
        Benchmark different access patterns on maps.

        Args:
            game_maps: List of GameMap instances to test
            iterations: Number of access operations per test

        Returns:
            Dictionary with access pattern benchmark results
        """
        results = {"access_patterns": {}}

        for i, game_map in enumerate(game_maps):
            map_key = f"map_{i}"
            height, width = game_map.grid.shape

            results["access_patterns"][map_key] = {
                "size": (height, width),
                "format": "legacy" if game_map.is_legacy() else "int",
            }

            # Random access pattern
            positions = [(np.random.randint(0, height), np.random.randint(0, width)) for _ in range(iterations)]

            if game_map.is_legacy():
                legacy_grid = game_map.get_legacy_grid()

                # Benchmark random access
                with self._measure_performance(f"legacy_random_access_{map_key}", iterations):
                    for r, c in positions:
                        _ = legacy_grid[r, c]

                # Benchmark sequential access
                with self._measure_performance(f"legacy_sequential_access_{map_key}", 1):
                    for r in range(height):
                        for c in range(width):
                            _ = legacy_grid[r, c]

            else:  # int-based
                int_grid = game_map.grid

                # Benchmark random access
                with self._measure_performance(f"int_random_access_{map_key}", iterations):
                    for r, c in positions:
                        _ = int_grid[r, c]

                # Benchmark sequential access
                with self._measure_performance(f"int_sequential_access_{map_key}", 1):
                    for r in range(height):
                        for c in range(width):
                            _ = int_grid[r, c]

        return results

    def benchmark_operations(
        self, game_maps: List[GameMap], operations: List[str] = None, iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark common map operations.

        Args:
            game_maps: List of GameMap instances to test
            operations: List of operations to benchmark
            iterations: Number of iterations per operation

        Returns:
            Dictionary with operation benchmark results
        """
        if operations is None:
            operations = ["copy", "flatten", "count_objects", "find_objects"]

        results = {"operations": {}}

        for i, game_map in enumerate(game_maps):
            map_key = f"map_{i}"
            results["operations"][map_key] = {}

            for operation in operations:
                op_name = f"{operation}_{map_key}"

                if operation == "copy":
                    with self._measure_performance(op_name, iterations):
                        for _ in range(iterations):
                            _ = game_map.grid.copy()

                elif operation == "flatten":
                    with self._measure_performance(op_name, iterations):
                        for _ in range(iterations):
                            _ = game_map.grid.flatten()

                elif operation == "count_objects":
                    with self._measure_performance(op_name, iterations):
                        for _ in range(iterations):
                            unique, counts = np.unique(game_map.grid, return_counts=True)

                elif operation == "find_objects":
                    # Find specific object type
                    target = "empty" if game_map.is_legacy() else 0
                    with self._measure_performance(op_name, iterations):
                        for _ in range(iterations):
                            _ = np.where(game_map.grid == target)

        return results

    def run_comprehensive_benchmark(self, test_maps: List[GameMap] = None) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark suite.

        Args:
            test_maps: Optional list of test maps. If None, generates test maps.

        Returns:
            Dictionary with all benchmark results
        """
        if test_maps is None:
            test_maps = self._generate_test_maps()

        comprehensive_results = {
            "timestamp": time.time(),
            "test_maps_count": len(test_maps),
            "system_info": self._get_system_info(),
        }

        # Run all benchmark suites
        comprehensive_results.update(self.benchmark_map_creation())
        comprehensive_results.update(self.benchmark_format_conversion(test_maps))
        comprehensive_results.update(self.benchmark_memory_usage())
        comprehensive_results.update(self.benchmark_access_patterns(test_maps))
        comprehensive_results.update(self.benchmark_operations(test_maps))

        return comprehensive_results

    def _generate_test_maps(self) -> List[GameMap]:
        """Generate a variety of test maps for benchmarking."""
        test_maps = []

        sizes = [(10, 10), (50, 50), (100, 100)]

        for height, width in sizes:
            # Legacy map
            legacy_grid = np.full((height, width), "empty", dtype=np.str_)
            legacy_grid[0, :] = "wall"
            legacy_grid[-1, :] = "wall"
            legacy_grid[:, 0] = "wall"
            legacy_grid[:, -1] = "wall"
            if height > 5 and width > 5:
                legacy_grid[height // 2, width // 2] = "agent"
            test_maps.append(GameMap(grid=legacy_grid))

            # Int map
            int_grid = np.zeros((height, width), dtype=np.uint8)
            int_grid[0, :] = 1  # wall
            int_grid[-1, :] = 1  # wall
            int_grid[:, 0] = 1  # wall
            int_grid[:, -1] = 1  # wall
            if height > 5 and width > 5:
                int_grid[height // 2, width // 2] = 10  # agent
            decoder_key = ["empty", "wall"] + [f"object_{i}" for i in range(2, 10)] + ["agent"]
            test_maps.append(GameMap(grid=int_grid, decoder_key=decoder_key))

        return test_maps

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "numpy_version": np.__version__,
        }

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmark results."""
        if not self.results:
            return {"error": "No benchmark results available"}

        summary = {
            "total_benchmarks": len(self.results),
            "successful": len([r for r in self.results if r.error is None]),
            "failed": len([r for r in self.results if r.error is not None]),
            "total_time": sum(r.execution_time for r in self.results),
            "average_time": np.mean([r.execution_time for r in self.results]),
            "results": [r.to_dict() for r in self.results],
        }

        return summary

    def clear_results(self):
        """Clear all stored benchmark results."""
        self.results.clear()
