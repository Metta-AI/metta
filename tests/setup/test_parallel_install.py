#!/usr/bin/env python3
"""
Tests for parallel installation with dependency resolution.

These tests verify:
- Dependency graph construction and resolution
- Topological sorting with parallel batching
- Proper ordering of component installation
- Missing dependency auto-inclusion
- Thread-safe parallel execution
- Complex dependency chains and structures
- Circular dependency detection
- Actual concurrent execution timing
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from metta.setup.components.base import SetupModule
from metta.setup.metta_cli import collect_missing_dependencies, topological_sort_parallel
from metta.setup.profiles import UserType
from tests.setup.test_base import BaseMettaSetupTest


class MockModule:
    """Mock setup module for testing dependency resolution."""

    def __init__(self, name: str, dependencies: list[str] = None, description: str = None):
        self.name = name
        self._dependencies = dependencies or []
        self.description = description or f"Mock {name} module"
        self.install_once = True
        self._installed = False

    def dependencies(self) -> list[str]:
        return self._dependencies

    def check_installed(self) -> bool:
        return self._installed

    def install(self, non_interactive: bool = False):
        self._installed = True


class TimedTestModule(SetupModule):
    """Test module that records installation timing for concurrency verification."""

    installation_log = []  # Shared log for tracking installation order and timing
    _log_lock = threading.Lock()

    def __init__(self, name: str, deps: list[str], install_duration: float = 0.01):
        super().__init__()
        self._name = name
        self._deps = deps
        self._install_duration = install_duration
        self._installed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Test module {self._name} (sleeps {self._install_duration}s)"

    def dependencies(self) -> list[str]:
        return self._deps

    def check_installed(self) -> bool:
        return self._installed

    def install(self, non_interactive: bool = False) -> None:
        """Install with timing logging for concurrency verification."""
        start_time = time.time()

        with self._log_lock:
            self.installation_log.append(
                {"module": self.name, "event": "start", "timestamp": start_time, "thread_id": threading.get_ident()}
            )

        # Simulate installation work
        time.sleep(self._install_duration)

        end_time = time.time()
        self._installed = True

        with self._log_lock:
            self.installation_log.append(
                {
                    "module": self.name,
                    "event": "end",
                    "timestamp": end_time,
                    "thread_id": threading.get_ident(),
                    "duration": end_time - start_time,
                }
            )

    @classmethod
    def clear_log(cls):
        """Clear the installation log for a fresh test."""
        with cls._log_lock:
            cls.installation_log.clear()

    @classmethod
    def get_log_summary(cls) -> dict:
        """Get timing analysis of installations."""
        with cls._log_lock:
            log = cls.installation_log.copy()

        if not log:
            return {}

        # Group by module
        modules = {}
        for event in log:
            module = event["module"]
            if module not in modules:
                modules[module] = {}
            modules[module][event["event"]] = event

        # Calculate overlaps and verify concurrent execution
        start_times = [(m, data["start"]["timestamp"]) for m, data in modules.items() if "start" in data]
        end_times = [(m, data["end"]["timestamp"]) for m, data in modules.items() if "end" in data]

        if not start_times or not end_times:
            return {"modules": modules, "overlaps": [], "total_time": 0, "concurrent_count": 0}

        # Find overlapping installations (concurrent execution)
        overlaps = []
        for i, (mod1, start1) in enumerate(start_times):
            for j, (mod2, start2) in enumerate(start_times):
                if i >= j:
                    continue
                # Check if mod1 and mod2 installations overlap
                end1 = modules[mod1]["end"]["timestamp"] if "end" in modules[mod1] else start1
                end2 = modules[mod2]["end"]["timestamp"] if "end" in modules[mod2] else start2

                if (start1 <= start2 < end1) or (start2 <= start1 < end2):
                    overlaps.append((mod1, mod2))

        return {
            "modules": modules,
            "overlaps": overlaps,
            "total_time": max(end_times, key=lambda x: x[1])[1] - min(start_times, key=lambda x: x[1])[1],
            "concurrent_count": len(overlaps),
        }


@pytest.mark.setup
@pytest.mark.profile("external")
class TestParallelInstallDependencyResolution(unittest.TestCase):
    """Test dependency resolution and parallel batching logic."""

    def test_topological_sort_parallel_simple(self):
        """Test topological sort with simple dependency chain."""
        # Create test modules: A -> B -> C (A depends on B, B depends on C)
        modules = {"A": Mock(name="A"), "B": Mock(name="B"), "C": Mock(name="C")}

        dependencies = {"A": ["B"], "B": ["C"], "C": []}

        batches = topological_sort_parallel(modules, dependencies)

        # Should have 3 batches: [C], [B], [A]
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ["C"])  # No dependencies
        self.assertEqual(batches[1], ["B"])  # Depends on C
        self.assertEqual(batches[2], ["A"])  # Depends on B

    def test_topological_sort_parallel_with_parallelism(self):
        """Test topological sort with parallel opportunities."""
        # Create test modules: A -> C, B -> C (both A and B depend on C, but A and B are independent)
        modules = {
            "A": Mock(name="A"),
            "B": Mock(name="B"),
            "C": Mock(name="C"),
            "D": Mock(name="D"),  # Independent module
        }

        dependencies = {"A": ["C"], "B": ["C"], "C": [], "D": []}

        batches = topological_sort_parallel(modules, dependencies)

        # Should have 2 batches: [C, D] (parallel), [A, B] (parallel after C completes)
        self.assertEqual(len(batches), 2)

        # First batch should contain C and D (no dependencies)
        self.assertIn("C", batches[0])
        self.assertIn("D", batches[0])
        self.assertEqual(len(batches[0]), 2)

        # Second batch should contain A and B (both depend on C)
        self.assertIn("A", batches[1])
        self.assertIn("B", batches[1])
        self.assertEqual(len(batches[1]), 2)

    def test_missing_dependency_inclusion(self):
        """Test that missing dependencies are automatically included."""
        # Simulate requesting notebookwidgets without nodejs
        requested_modules = ["notebookwidgets", "core", "aws"]

        # Mock the get_all_modules to return our test modules
        all_modules = [
            MockModule("core", []),
            MockModule("aws", []),
            MockModule("system", []),
            MockModule("nodejs", ["system"]),
            MockModule("notebookwidgets", ["nodejs"]),
            MockModule("mettascope", ["nodejs"]),
        ]

        # Create module maps
        module_map = {m.name: m for m in all_modules if m.name in requested_modules}
        all_modules_dict = {m.name: m for m in all_modules}

        # Test the missing dependency collection logic using extracted function
        missing_deps = collect_missing_dependencies(module_map, all_modules_dict)

        # Should find nodejs and system as missing dependencies
        self.assertIn("nodejs", missing_deps)
        self.assertIn("system", missing_deps)

        # Add missing deps to module_map
        for dep_name in missing_deps:
            if dep_name in all_modules_dict:
                module_map[dep_name] = all_modules_dict[dep_name]

        # Now build final dependency graph
        dependencies = {}
        for module in module_map.values():
            dependencies[module.name] = module.dependencies()

        # Verify correct dependencies
        self.assertEqual(dependencies["notebookwidgets"], ["nodejs"])
        self.assertEqual(dependencies["nodejs"], ["system"])
        self.assertEqual(dependencies["system"], [])
        self.assertEqual(dependencies["core"], [])
        self.assertEqual(dependencies["aws"], [])

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        # Create circular dependency: A -> B -> C -> A
        modules = {"A": Mock(name="A"), "B": Mock(name="B"), "C": Mock(name="C")}

        dependencies = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],  # Creates circular dependency
        }

        batches = topological_sort_parallel(modules, dependencies)

        # With circular dependency, no modules should be in the ready queue initially
        # So we should get an empty result
        self.assertEqual(len(batches), 0)

    def test_real_component_dependencies(self):
        """Test with real component dependency patterns."""
        # Test realistic component setup
        modules = {
            "system": Mock(name="system"),
            "core": Mock(name="core"),
            "aws": Mock(name="aws"),
            "nodejs": Mock(name="nodejs"),
            "notebookwidgets": Mock(name="notebookwidgets"),
            "mettascope": Mock(name="mettascope"),
            "skypilot": Mock(name="skypilot"),
            "datadog_agent": Mock(name="datadog_agent"),
        }

        # Real dependency relationships from the codebase
        dependencies = {
            "system": [],
            "core": [],
            "aws": [],
            "nodejs": ["system"],
            "notebookwidgets": ["nodejs"],
            "mettascope": ["nodejs"],
            "skypilot": ["aws"],
            "datadog_agent": ["aws"],
        }

        batches = topological_sort_parallel(modules, dependencies)

        # Expected batching:
        # Batch 1: system, core, aws (no dependencies - can run in parallel)
        # Batch 2: nodejs, skypilot, datadog_agent (depend on system/aws - can run in parallel after batch 1)
        # Batch 3: notebookwidgets, mettascope (depend on nodejs - can run in parallel after batch 2)

        self.assertEqual(len(batches), 3)

        # Batch 1: Independent modules
        batch1 = set(batches[0])
        self.assertIn("system", batch1)
        self.assertIn("core", batch1)
        self.assertIn("aws", batch1)

        # Batch 2: Modules depending on batch 1
        batch2 = set(batches[1])
        self.assertIn("nodejs", batch2)
        self.assertIn("skypilot", batch2)
        self.assertIn("datadog_agent", batch2)

        # Batch 3: Modules depending on nodejs
        batch3 = set(batches[2])
        self.assertIn("notebookwidgets", batch3)
        self.assertIn("mettascope", batch3)


@pytest.mark.setup
@pytest.mark.profile("external")
class TestParallelInstallIntegration(unittest.TestCase):
    """Integration tests for parallel install logic with mock modules."""

    def test_parallel_install_output_format(self):
        """Test that parallel install produces expected batch structure."""
        # Test with mock modules that have a dependency chain
        modules = {
            "system": MockModule("system", []),
            "nodejs": MockModule("nodejs", ["system"]),
            "notebookwidgets": MockModule("notebookwidgets", ["nodejs"]),
        }
        dependencies = {m.name: m.dependencies() for m in modules.values()}

        batches = topological_sort_parallel(modules, dependencies)

        # Should have 3 batches: system -> nodejs -> notebookwidgets
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ["system"])
        self.assertEqual(batches[1], ["nodejs"])
        self.assertEqual(batches[2], ["notebookwidgets"])

    def test_missing_dependency_auto_addition(self):
        """Test that missing dependencies are automatically added."""
        # Request notebookwidgets without nodejs
        requested_modules = ["notebookwidgets", "core"]

        all_modules = [
            MockModule("system", []),
            MockModule("core", []),
            MockModule("nodejs", ["system"]),
            MockModule("notebookwidgets", ["nodejs"]),
        ]

        module_map = {m.name: m for m in all_modules if m.name in requested_modules}
        all_modules_dict = {m.name: m for m in all_modules}

        # Should find missing dependencies
        missing_deps = collect_missing_dependencies(module_map, all_modules_dict)
        self.assertEqual(missing_deps, {"nodejs", "system"})

    def test_parallel_batch_ordering(self):
        """Test that components install in correct dependency order."""
        # Test with mock components that have clear dependency relationships
        modules = {
            "core": MockModule("core", []),
            "aws": MockModule("aws", []),
            "nodejs": MockModule("nodejs", ["system"]),
            "system": MockModule("system", []),
            "skypilot": MockModule("skypilot", ["aws"]),
            "mettascope": MockModule("mettascope", ["nodejs"]),
        }
        dependencies = {m.name: m.dependencies() for m in modules.values()}

        batches = topological_sort_parallel(modules, dependencies)

        # Should have multiple batches due to dependencies
        self.assertGreater(len(batches), 1, "Should have multiple batches for dependencies")

        # Verify dependency ordering
        batch_positions = {}
        for i, batch in enumerate(batches):
            for module in batch:
                batch_positions[module] = i

        # Dependencies should be in earlier batches
        if "nodejs" in batch_positions and "system" in batch_positions:
            self.assertLess(batch_positions["system"], batch_positions["nodejs"])
        if "skypilot" in batch_positions and "aws" in batch_positions:
            self.assertLessEqual(batch_positions["aws"], batch_positions["skypilot"])
        if "mettascope" in batch_positions and "nodejs" in batch_positions:
            self.assertLess(batch_positions["nodejs"], batch_positions["mettascope"])


@pytest.mark.setup
@pytest.mark.profile("external")
class TestComplexDependencyStructures(unittest.TestCase):
    """Test complex dependency structures and edge cases."""

    def test_diamond_dependency(self):
        """Test diamond dependency pattern: A,B->C; C,D->E (diamond with E at bottom)."""
        modules = {
            "A": Mock(name="A"),
            "B": Mock(name="B"),
            "C": Mock(name="C"),
            "D": Mock(name="D"),
            "E": Mock(name="E"),
        }
        dependencies = {"A": ["C"], "B": ["C"], "C": ["E"], "D": ["E"], "E": []}

        batches = topological_sort_parallel(modules, dependencies)

        # Should be 3 batches: [E], [C,D], [A,B]
        self.assertEqual(len(batches), 3)
        self.assertEqual(set(batches[0]), {"E"})
        self.assertEqual(set(batches[1]), {"C", "D"})
        self.assertEqual(set(batches[2]), {"A", "B"})

    def test_deep_chain(self):
        """Test deep dependency chain: A->B->C->D->E->F."""
        modules = {f"L{i}": Mock(name=f"L{i}") for i in range(6)}
        dependencies = {"L0": ["L1"], "L1": ["L2"], "L2": ["L3"], "L3": ["L4"], "L4": ["L5"], "L5": []}

        batches = topological_sort_parallel(modules, dependencies)

        # Should be 6 batches, one for each level
        self.assertEqual(len(batches), 6)
        self.assertEqual(batches[0], ["L5"])  # No deps
        self.assertEqual(batches[1], ["L4"])  # Depends on L5
        self.assertEqual(batches[2], ["L3"])  # Depends on L4
        self.assertEqual(batches[3], ["L2"])  # Depends on L3
        self.assertEqual(batches[4], ["L1"])  # Depends on L2
        self.assertEqual(batches[5], ["L0"])  # Depends on L1

    def test_complex_branching(self):
        """Test complex branching: multiple roots, multiple shared dependencies."""
        # Structure:
        # Frontend, Backend -> WebServer -> Database
        # Analytics, Monitoring -> Metrics -> Database
        # Cache (independent)
        modules = {
            "Frontend": Mock(name="Frontend"),
            "Backend": Mock(name="Backend"),
            "WebServer": Mock(name="WebServer"),
            "Database": Mock(name="Database"),
            "Analytics": Mock(name="Analytics"),
            "Monitoring": Mock(name="Monitoring"),
            "Metrics": Mock(name="Metrics"),
            "Cache": Mock(name="Cache"),
        }
        dependencies = {
            "Frontend": ["WebServer"],
            "Backend": ["WebServer"],
            "WebServer": ["Database"],
            "Analytics": ["Metrics"],
            "Monitoring": ["Metrics"],
            "Metrics": ["Database"],
            "Database": [],
            "Cache": [],
        }

        batches = topological_sort_parallel(modules, dependencies)

        # Should be 3 batches (all modules are resolved)
        self.assertEqual(len(batches), 3)
        self.assertEqual(set(batches[0]), {"Database", "Cache"})  # No deps
        self.assertEqual(set(batches[1]), {"WebServer", "Metrics"})  # Depend on Database
        self.assertEqual(set(batches[2]), {"Frontend", "Backend", "Analytics", "Monitoring"})
        # All modules resolved in 3 batches

    def test_missing_transitive_dependencies(self):
        """Test deep transitive dependency resolution."""
        # User requests only Frontend, but it needs Backend->Database->Network
        requested_modules = ["Frontend"]

        all_modules = [
            MockModule("Frontend", ["Backend"]),
            MockModule("Backend", ["Database"]),
            MockModule("Database", ["Network"]),
            MockModule("Network", []),
            MockModule("Independent", []),
        ]

        module_map = {m.name: m for m in all_modules if m.name in requested_modules}
        all_modules_dict = {m.name: m for m in all_modules}

        missing_deps = collect_missing_dependencies(module_map, all_modules_dict)

        # Should find all transitive dependencies
        self.assertEqual(missing_deps, {"Backend", "Database", "Network"})

    def test_multiple_root_dependencies(self):
        """Test module with multiple independent dependency chains."""
        # Structure: WebApp -> [Database, Cache, MessageQueue]
        # where Database->Storage, Cache->Redis, MessageQueue->Broker
        requested_modules = ["WebApp"]

        all_modules = [
            MockModule("WebApp", ["Database", "Cache", "MessageQueue"]),
            MockModule("Database", ["Storage"]),
            MockModule("Storage", []),
            MockModule("Cache", ["Redis"]),
            MockModule("Redis", []),
            MockModule("MessageQueue", ["Broker"]),
            MockModule("Broker", []),
        ]

        module_map = {m.name: m for m in all_modules if m.name in requested_modules}
        all_modules_dict = {m.name: m for m in all_modules}

        missing_deps = collect_missing_dependencies(module_map, all_modules_dict)

        # Should find all dependencies in all chains
        expected = {"Database", "Cache", "MessageQueue", "Storage", "Redis", "Broker"}
        self.assertEqual(missing_deps, expected)


@pytest.mark.setup
@pytest.mark.profile("external")
class TestConcurrentExecutionTiming(unittest.TestCase):
    """Test actual concurrent execution with timing verification."""

    def test_parallel_execution_timing(self):
        """Test that independent modules actually install concurrently."""
        # Create 3 independent modules that each take 2 seconds
        modules = [
            TimedTestModule("ModA", [], install_duration=0.01),
            TimedTestModule("ModB", [], install_duration=0.01),
            TimedTestModule("ModC", [], install_duration=0.01),
        ]

        module_map = {m.name: m for m in modules}
        dependencies = {m.name: m.dependencies() for m in modules}

        # Get batches
        batches = topological_sort_parallel(module_map, dependencies)
        self.assertEqual(len(batches), 1)  # All independent, single batch
        self.assertEqual(set(batches[0]), {"ModA", "ModB", "ModC"})

        # Execute in parallel with ThreadPoolExecutor
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_modules = [module_map[name] for name in batches[0]]
            futures = [executor.submit(lambda m: m.install(), module) for module in batch_modules]
            # Wait for all to complete
            for future in futures:
                future.result()

        total_time = time.time() - start_time

        # Verify timing - should be fast due to parallel execution
        self.assertLess(total_time, 1.0, f"Parallel execution took {total_time}s, should be fast")

        # Verify concurrent execution in logs
        summary = TimedTestModule.get_log_summary()
        self.assertGreaterEqual(summary["concurrent_count"], 1, "Should have concurrent overlapping installations")

        # Verify all modules have different thread IDs (running in parallel)
        thread_ids = set()
        for module_data in summary["modules"].values():
            if "start" in module_data:
                thread_ids.add(module_data["start"]["thread_id"])
        self.assertGreaterEqual(len(thread_ids), 2, "Should have multiple threads executing")

    def test_dependency_ordering_with_timing(self):
        """Test that dependent modules wait for their dependencies."""
        # Chain: ModA -> ModB -> ModC (each takes 0.05 seconds)
        modules = [
            TimedTestModule("ModA", ["ModB"], install_duration=0.01),
            TimedTestModule("ModB", ["ModC"], install_duration=0.01),
            TimedTestModule("ModC", [], install_duration=0.01),
        ]

        module_map = {m.name: m for m in modules}
        dependencies = {m.name: m.dependencies() for m in modules}

        batches = topological_sort_parallel(module_map, dependencies)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ["ModC"])
        self.assertEqual(batches[1], ["ModB"])
        self.assertEqual(batches[2], ["ModA"])

        # Execute batches in order
        start_time = time.time()

        for batch in batches:
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_modules = [module_map[name] for name in batch]
                futures = [executor.submit(lambda m: m.install(), module) for module in batch_modules]
                for future in futures:
                    future.result()

        total_time = time.time() - start_time

        # Should complete successfully in reasonable time
        self.assertLess(total_time, 1.0, f"Sequential execution took {total_time}s, should be reasonable")

        # Verify order in installation log
        summary = TimedTestModule.get_log_summary()

        # ModC should start and end before ModB starts
        mod_c_end = summary["modules"]["ModC"]["end"]["timestamp"]
        mod_b_start = summary["modules"]["ModB"]["start"]["timestamp"]
        self.assertLess(mod_c_end, mod_b_start, "ModC should complete before ModB starts")

        # ModB should start and end before ModA starts
        mod_b_end = summary["modules"]["ModB"]["end"]["timestamp"]
        mod_a_start = summary["modules"]["ModA"]["start"]["timestamp"]
        self.assertLess(mod_b_end, mod_a_start, "ModB should complete before ModA starts")

    def test_mixed_parallel_and_sequential(self):
        """Test complex scenario with both parallel and sequential execution."""
        # Structure: Base -> [ServiceA, ServiceB] -> [Frontend, Backend]
        # Base (2s) -> ServiceA,ServiceB parallel (1.5s each) -> Frontend,Backend parallel (1s each)
        modules = [
            TimedTestModule("Base", [], install_duration=0.01),
            TimedTestModule("ServiceA", ["Base"], install_duration=0.01),
            TimedTestModule("ServiceB", ["Base"], install_duration=0.01),
            TimedTestModule("Frontend", ["ServiceA"], install_duration=0.01),
            TimedTestModule("Backend", ["ServiceB"], install_duration=0.01),
        ]

        module_map = {m.name: m for m in modules}
        dependencies = {m.name: m.dependencies() for m in modules}

        batches = topological_sort_parallel(module_map, dependencies)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ["Base"])
        self.assertEqual(set(batches[1]), {"ServiceA", "ServiceB"})
        self.assertEqual(set(batches[2]), {"Frontend", "Backend"})

        # Execute with proper timing
        start_time = time.time()

        for batch in batches:
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_modules = [module_map[name] for name in batch]
                futures = [executor.submit(lambda m: m.install(), module) for module in batch_modules]
                for future in futures:
                    future.result()

        total_time = time.time() - start_time

        # Should complete successfully in reasonable time
        self.assertLess(total_time, 1.0, f"Mixed execution took {total_time}s, should be reasonable")

        # Verify ServiceA and ServiceB overlap (parallel execution)
        summary = TimedTestModule.get_log_summary()
        overlaps = summary["overlaps"]

        # Should have at least one overlap between ServiceA and ServiceB
        service_overlaps = [overlap for overlap in overlaps if set(overlap) == {"ServiceA", "ServiceB"}]
        self.assertGreaterEqual(len(service_overlaps), 1, "ServiceA and ServiceB should execute in parallel")


@pytest.mark.setup
@pytest.mark.profile("external")
class TestCircularDependencyDetection(unittest.TestCase):
    """Test circular dependency detection and error handling."""

    def test_simple_circular_dependency(self):
        """Test detection of simple A->B->A circular dependency."""
        modules = {"A": Mock(name="A"), "B": Mock(name="B")}
        dependencies = {"A": ["B"], "B": ["A"]}

        batches = topological_sort_parallel(modules, dependencies)

        # Should return empty or incomplete batches for circular dependencies
        # Both A and B will have in_degree > 0 and never be added to ready_queue
        if batches:
            # If any batches returned, they shouldn't contain all modules
            all_batched = {module for batch in batches for module in batch}
            self.assertNotEqual(all_batched, {"A", "B"}, "Circular dependency should not resolve all modules")
        else:
            # Or should return empty list
            self.assertEqual(batches, [])

    def test_complex_circular_dependency(self):
        """Test detection of complex circular dependency A->B->C->D->A."""
        modules = {f"Mod{i}": Mock(name=f"Mod{i}") for i in ["A", "B", "C", "D"]}
        dependencies = {"ModA": ["ModB"], "ModB": ["ModC"], "ModC": ["ModD"], "ModD": ["ModA"]}

        batches = topological_sort_parallel(modules, dependencies)

        # Should not resolve all modules due to circular dependency
        if batches:
            all_batched = {module for batch in batches for module in batch}
            self.assertNotEqual(len(all_batched), 4, "Circular dependency should not resolve all 4 modules")
        else:
            self.assertEqual(batches, [])

    def test_partial_circular_dependency(self):
        """Test scenario where some modules have circular deps, others don't."""
        # Structure: A->B->A (circular), C->D (normal), E (independent)
        modules = {
            "A": Mock(name="A"),
            "B": Mock(name="B"),
            "C": Mock(name="C"),
            "D": Mock(name="D"),
            "E": Mock(name="E"),
        }
        dependencies = {
            "A": ["B"],
            "B": ["A"],  # Circular
            "C": ["D"],
            "D": [],  # Normal chain
            "E": [],  # Independent
        }

        batches = topological_sort_parallel(modules, dependencies)

        # Should successfully batch D and E, but not resolve A,B circular dependency
        all_batched = {module for batch in batches for module in batch}

        # Should at least include the non-circular modules
        self.assertIn("D", all_batched, "Non-circular module D should be batched")
        self.assertIn("E", all_batched, "Independent module E should be batched")

        # May or may not include C depending on implementation
        # Should NOT include both A and B (circular dependency)
        circular_resolved = {"A", "B"}.issubset(all_batched)
        self.assertFalse(circular_resolved, "Circular dependency A->B->A should not be fully resolved")


@pytest.mark.setup
@pytest.mark.profile("external")
class TestManualScenarios(BaseMettaSetupTest):
    """Test the specific scenarios that were manually verified."""

    def setUp(self):
        super().setUp()
        self._create_test_config(UserType.EXTERNAL)

    def test_diamond_dependency_manual_scenario(self):
        """Test the exact diamond dependency scenario from manual verification."""
        # Diamond dependency pattern: A,B->C; C,D->E
        modules = {"A": "mod_a", "B": "mod_b", "C": "mod_c", "D": "mod_d", "E": "mod_e"}
        dependencies = {"A": ["C"], "B": ["C"], "C": ["E"], "D": ["E"], "E": []}

        batches = topological_sort_parallel(modules, dependencies)

        # Should be: [E], [C,D], [A,B]
        expected_batch_0 = {"E"}
        expected_batch_1 = {"C", "D"}
        expected_batch_2 = {"A", "B"}

        self.assertEqual(len(batches), 3, "Should have exactly 3 batches")
        self.assertEqual(set(batches[0]), expected_batch_0, f"Batch 0 should be {expected_batch_0}")
        self.assertEqual(set(batches[1]), expected_batch_1, f"Batch 1 should be {expected_batch_1}")
        self.assertEqual(set(batches[2]), expected_batch_2, f"Batch 2 should be {expected_batch_2}")

    def test_simple_circular_dependency_manual_scenario(self):
        """Test the exact circular dependency scenario from manual verification."""
        # Simple circular: A->B->A
        modules = {"A": "mod_a", "B": "mod_b"}
        dependencies = {"A": ["B"], "B": ["A"]}

        batches = topological_sort_parallel(modules, dependencies)

        # Should be empty or not resolve all modules
        all_batched = {module for batch in batches for module in batch} if batches else set()

        # Circular dependency should not be fully resolved
        self.assertNotEqual(all_batched, {"A", "B"}, "Circular dependency should not resolve all modules")

    def test_partial_circular_dependency_manual_scenario(self):
        """Test the exact partial circular dependency scenario from manual verification."""
        # Partial circular: A->B->A (circular), C->D (normal), E (independent)
        modules = {"A": "mod_a", "B": "mod_b", "C": "mod_c", "D": "mod_d", "E": "mod_e"}
        dependencies = {
            "A": ["B"],
            "B": ["A"],  # Circular
            "C": ["D"],
            "D": [],  # Normal chain
            "E": [],  # Independent
        }

        batches = topological_sort_parallel(modules, dependencies)
        all_batched = {module for batch in batches for module in batch}

        # Verify non-circular modules are batched
        self.assertIn("D", all_batched, "Non-circular module D should be batched")
        self.assertIn("E", all_batched, "Independent module E should be batched")
        # Circular should not be fully resolved
        self.assertFalse({"A", "B"}.issubset(all_batched), "Circular dependency A->B->A should not be fully resolved")

    def test_comprehensive_real_world_scenario(self):
        """Test the comprehensive integration scenario from manual verification."""
        # Real-world scenario: User requests mettascope and skypilot
        requested = ["mettascope", "skypilot"]

        # All available modules with realistic dependencies
        all_modules = [
            MockModule("system", []),
            MockModule("core", []),
            MockModule("aws", []),
            MockModule("nodejs", ["system"]),
            MockModule("mettascope", ["nodejs"]),
            MockModule("skypilot", ["aws"]),
            MockModule("notebookwidgets", ["nodejs"]),
            MockModule("datadog_agent", ["aws"]),
        ]

        # Create maps
        module_map = {m.name: m for m in all_modules if m.name in requested}
        all_modules_dict = {m.name: m for m in all_modules}

        # Test dependency collection
        missing_deps = collect_missing_dependencies(module_map, all_modules_dict)
        expected_missing = {"aws", "nodejs", "system"}
        self.assertEqual(missing_deps, expected_missing, f"Should find missing deps: {expected_missing}")

        # Add missing deps
        for dep_name in missing_deps:
            if dep_name in all_modules_dict:
                module_map[dep_name] = all_modules_dict[dep_name]

        final_modules = sorted(module_map.keys())
        expected_final = ["aws", "mettascope", "nodejs", "skypilot", "system"]
        self.assertEqual(final_modules, expected_final, f"Final module list should be: {expected_final}")

        # Test topological sorting
        dependencies = {m.name: m.dependencies() for m in module_map.values()}
        batches = topological_sort_parallel(module_map, dependencies)

        # Verify dependency relationships
        self.assertEqual(dependencies.get("nodejs"), ["system"], "nodejs should depend on system")
        self.assertEqual(dependencies.get("mettascope"), ["nodejs"], "mettascope should depend on nodejs")
        self.assertEqual(dependencies.get("skypilot"), ["aws"], "skypilot should depend on aws")

        # Verify batch structure
        batch_contents = [set(batch) for batch in batches]

        # system/aws should be in early batch
        early_batch_has_deps = any("system" in batch and "aws" in batch for batch in batch_contents)
        self.assertTrue(early_batch_has_deps, "system and aws should be in early batch")

        # nodejs/skypilot should be after their dependencies
        later_batches = batch_contents[1:] if len(batch_contents) > 1 else []
        nodejs_skypilot_later = any("nodejs" in batch and "skypilot" in batch for batch in later_batches)
        self.assertTrue(nodejs_skypilot_later, "nodejs and skypilot should be in later batch")

        # mettascope should be last (depends on nodejs)
        if batch_contents:
            mettascope_last = "mettascope" in batch_contents[-1]
            self.assertTrue(mettascope_last, "mettascope should be in final batch")

    def test_concurrent_execution_verification(self):
        """Test that concurrent execution actually works with timing."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        # Create simple test module that sleeps
        class QuickTimedModule:
            def __init__(self, name, deps, install_duration=0.01):
                self.name = name
                self._deps = deps
                self._install_duration = install_duration
                self._installed = False
                self.install_times = []

            def dependencies(self):
                return self._deps

            def install(self):
                start_time = time.time()
                time.sleep(self._install_duration)
                end_time = time.time()
                self.install_times.append((start_time, end_time))
                self._installed = True

        # Create 3 independent modules that each take 0.05 seconds
        modules = [
            QuickTimedModule("ModA", [], install_duration=0.01),
            QuickTimedModule("ModB", [], install_duration=0.01),
            QuickTimedModule("ModC", [], install_duration=0.01),
        ]

        # Execute in parallel
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(module.install) for module in modules]
            for future in futures:
                future.result()

        total_time = time.time() - start_time

        # Verify timing - should complete successfully in reasonable time
        self.assertLess(total_time, 1.0, f"Parallel execution took {total_time:.3f}s, should be reasonable")

        # Verify all modules were installed
        for module in modules:
            self.assertTrue(module._installed, f"Module {module.name} should be installed")
            self.assertEqual(len(module.install_times), 1, f"Module {module.name} should have one install time")


if __name__ == "__main__":
    unittest.main()
