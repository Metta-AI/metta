import os
import sys
import time
from pathlib import Path

import pytest

from metta.common.tests.fixtures import docker_client_fixture

print(f"\n===== applying conftest from {Path(__file__)} =====")

# Add dependencies to sys.path if not already present
base_dir = Path(__file__).resolve().parent

print("\n===== DEBUG: Python sys.path =====")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
print("===== END DEBUG: Python sys.path =====\n")


class TestDurationMonitor:
    """Simple test duration monitor that warns about slow tests"""

    def __init__(self):
        self.slow_threshold = float(os.environ.get("PYTEST_SLOW_THRESHOLD", "5.0"))
        self.test_durations = {}
        self.slow_unmarked_tests = []
        self._start_times = {}

    def test_started(self, nodeid):
        """Record test start time"""
        self._start_times[nodeid] = time.time()

    def test_finished(self, nodeid, item):
        """Record test duration and check if properly marked"""
        if nodeid not in self._start_times:
            return

        duration = time.time() - self._start_times.pop(nodeid)
        self.test_durations[nodeid] = duration

        # Check if test is marked as slow
        is_marked_slow = any(marker.name == "slow" for marker in item.iter_markers())

        # Warn if test is slow but not marked
        if duration > self.slow_threshold and not is_marked_slow:
            self.slow_unmarked_tests.append((nodeid, duration))

    def print_warnings(self):
        """Print warnings about slow unmarked tests"""
        print("\n" + "=" * 80)
        print("📊 Test Duration Monitor Summary")
        print("=" * 80)

        # Always print stats
        total_tests = len(self.test_durations)
        slow_tests = len([d for d in self.test_durations.values() if d > self.slow_threshold])

        print(f"Total tests monitored: {total_tests}")
        print(f"Slow threshold: {self.slow_threshold}s")
        print(f"Tests exceeding threshold: {slow_tests}")

        if not self.slow_unmarked_tests:
            print("\n✅ All slow tests are properly marked with @pytest.mark.slow")
            print("=" * 80 + "\n")
            return

        print(f"\n⚠️  WARNING: Found {len(self.slow_unmarked_tests)} unmarked slow test(s)")
        print("=" * 80)

        # Sort by duration (slowest first)
        self.slow_unmarked_tests.sort(key=lambda x: x[1], reverse=True)

        for node_id, duration in self.slow_unmarked_tests:
            print(f"  {duration:6.2f}s - {node_id}")

            # GitHub Actions annotation if in CI
            if os.environ.get("GITHUB_ACTIONS"):
                # Extract file path from nodeid
                file_path = node_id.split("::")[0] if "::" in node_id else node_id
                print(
                    f"::warning file={file_path}::Test '{node_id}' took {duration:.2f}s "
                    + "but is not marked with @pytest.mark.slow"
                )

        print("\nTo fix: Add @pytest.mark.slow decorator to these tests")
        print("=" * 80 + "\n")


# Global instance
_duration_monitor = TestDurationMonitor()


def pytest_configure(config):
    # Add multiple markers correctly
    config.addinivalue_line("markers", "benchmark: mark a test as a benchmark test")
    config.addinivalue_line("markers", "verbose: mark a test to display verbose output")
    config.addinivalue_line("markers", "slow: mark a test as slow (runs in second phase)")


@pytest.fixture
def verbose(request):
    """Fixture that can be used in tests to check if verbose mode is enabled."""
    marker = request.node.get_closest_marker("verbose")
    return marker is not None


# Properly handle output capture for verbose tests
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()

    # Only process after the call phase (actual test execution)
    if report.when == "call" and item.get_closest_marker("verbose"):
        capman = item.config.pluginmanager.get_plugin("capturemanager")
        if capman and hasattr(report, "capstdout") and hasattr(report, "capstderr"):
            # Print the captured output with formatting
            print(f"\n\n===== VERBOSE OUTPUT FOR: {item.name} =====\n")
            if report.capstdout:
                print("--- STDOUT ---")
                print(report.capstdout)
            if report.capstderr:
                print("--- STDERR ---")
                print(report.capstderr)
            print(f"===== END VERBOSE OUTPUT FOR: {item.name} =====\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # noqa: ARG001
    """Monitor test execution time"""
    node_id = item.nodeid
    _duration_monitor.test_started(node_id)
    _outcome = yield
    _duration_monitor.test_finished(node_id, item)


@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Print duration warnings at end of test session"""
    _duration_monitor.print_warnings()


docker_client = docker_client_fixture()
