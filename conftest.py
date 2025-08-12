import pytest

from metta.common.test_support import docker_client_fixture


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
def pytest_runtest_makereport(item, call):
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


docker_client = docker_client_fixture()
