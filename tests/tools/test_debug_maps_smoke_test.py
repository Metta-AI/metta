"""
Test suite for debug_maps_smoke_test.py tool.

This module tests the smoke test infrastructure for debug maps.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.debug_maps_smoke_test import DebugMapSmokeTestResults


class TestDebugMapSmokeTestResults:
    """Test suite for DebugMapSmokeTestResults class."""

    def test_init(self):
        """Test DebugMapSmokeTestResults initialization."""
        results = DebugMapSmokeTestResults()
        assert results.map_results == {}
        assert results.overall_success is True
        assert results.errors == []

    def test_add_map_result_success(self):
        """Test adding a successful map result."""
        results = DebugMapSmokeTestResults()
        results.add_map_result(
            map_name="test_map", success=True, avg_reward=0.5, avg_steps=50, completion_rate=0.8, errors=[]
        )

        assert "test_map" in results.map_results
        assert results.map_results["test_map"]["success"] is True
        assert results.map_results["test_map"]["avg_reward"] == 0.5
        assert results.map_results["test_map"]["avg_steps"] == 50
        assert results.map_results["test_map"]["completion_rate"] == 0.8
        assert results.map_results["test_map"]["errors"] == []
        assert results.overall_success is True
        assert results.errors == []

    def test_add_map_result_failure(self):
        """Test adding a failed map result."""
        results = DebugMapSmokeTestResults()
        test_errors = ["Low reward", "Timeout"]

        results.add_map_result(
            map_name="test_map", success=False, avg_reward=0.05, avg_steps=250, completion_rate=0.0, errors=test_errors
        )

        assert "test_map" in results.map_results
        assert results.map_results["test_map"]["success"] is False
        assert results.overall_success is False
        assert results.errors == test_errors

    def test_multiple_map_results(self):
        """Test adding multiple map results."""
        results = DebugMapSmokeTestResults()

        # Add successful result
        results.add_map_result("map1", True, 0.5, 50, 0.8, [])

        # Add failed result
        results.add_map_result("map2", False, 0.05, 250, 0.0, ["Error"])

        assert len(results.map_results) == 2
        assert results.overall_success is False
        assert "Error" in results.errors

    @patch("builtins.print")
    def test_print_summary_success(self, mock_print):
        """Test print_summary with successful results."""
        results = DebugMapSmokeTestResults()
        results.add_map_result("test_map", True, 0.5, 50, 0.8, [])

        results.print_summary()

        # Verify that print was called with success messages
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("✅ ALL TESTS PASSED" in call for call in print_calls)
        assert any("test_map" in call for call in print_calls)

    @patch("builtins.print")
    def test_print_summary_failure(self, mock_print):
        """Test print_summary with failed results."""
        results = DebugMapSmokeTestResults()
        results.add_map_result("test_map", False, 0.05, 250, 0.0, ["Low reward"])

        results.print_summary()

        # Verify that print was called with failure messages
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("❌ SOME TESTS FAILED" in call for call in print_calls)
        assert any("Low reward" in call for call in print_calls)


class TestDebugMapSmokeTestScript:
    """Test suite for the smoke test script functionality."""

    def test_script_file_exists(self):
        """Test that the smoke test script file exists."""
        script_path = Path("tools/debug_maps_smoke_test.py")
        assert script_path.exists(), "Smoke test script should exist"
        assert script_path.is_file(), "Smoke test script should be a file"

    def test_script_has_main_function(self):
        """Test that the script has a main function."""
        # Import the module to check it has the required functions
        import tools.debug_maps_smoke_test as smoke_test_module

        assert hasattr(smoke_test_module, "main"), "Script should have a main function"
        assert hasattr(smoke_test_module, "run_debug_maps_smoke_test"), (
            "Script should have run_debug_maps_smoke_test function"
        )
        assert hasattr(smoke_test_module, "test_single_map"), "Script should have test_single_map function"

    def test_debug_map_definitions(self):
        """Test that debug maps are properly defined in the script."""

        # Read the file content to check debug maps definition
        script_path = Path("tools/debug_maps_smoke_test.py")
        with open(script_path, "r") as f:
            content = f.read()

        # Check that the new debug paths are used
        expected_paths = [
            "env/mettagrid/debug/evals/debug_mixed_objects",
            "env/mettagrid/debug/evals/debug_resource_collection",
            "env/mettagrid/debug/evals/debug_simple_obstacles",
            "env/mettagrid/debug/evals/debug_tiny_two_altars",
        ]

        for path in expected_paths:
            assert path in content, f"Script should contain path {path}"

    def test_script_imports(self):
        """Test that the script imports are working correctly."""
        # Try importing the module to verify imports work
        try:
            import tools.debug_maps_smoke_test
        except ImportError as e:
            pytest.fail(f"Failed to import smoke test script: {e}")

    def test_success_criteria_constants(self):
        """Test that success criteria are properly defined."""

        script_path = Path("tools/debug_maps_smoke_test.py")
        with open(script_path, "r") as f:
            content = f.read()

        # Check that success criteria are defined
        expected_criteria = ["min_reward", "max_avg_steps", "min_completion_rate", "max_execution_time"]

        for criteria in expected_criteria:
            assert criteria in content, f"Script should define {criteria} criteria"
