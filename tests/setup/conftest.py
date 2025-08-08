"""
Pytest configuration for setup integration tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from metta.common.util.fs import get_repo_root
from tests.setup.test_base import BaseMettaSetupTest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options to control profile-aware tests.

    Options can also be provided via environment variables:
    - METTA_TEST_SETUP: whether we want setup tests to run or not (default: "", set to "1" to run tests)
    - METTA_TEST_PROFILE: active profile name (e.g., softmax, cloud, external, softmax-docker, custom)
    """
    parser.addoption(
        "--metta-setup",
        action="store",
        default=os.environ.get("METTA_TEST_SETUP"),
        help="Whether pytest knows to run setup integration tests or not.",
    )
    parser.addoption(
        "--metta-profile",
        action="store",
        default=os.environ.get("METTA_TEST_PROFILE"),
        help="Active Metta profile for tests (from METTA_TEST_PROFILE).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the setup tests."""
    config.addinivalue_line("markers", "profile(name): test requires the given profile to be active")
    config.addinivalue_line("markers", "setup: mark a setup test/class to be executed")

    # Propagate active profile into BaseMettaSetupTest for convenience
    active_profile = config.getoption("--metta-profile") or os.environ.get("METTA_TEST_PROFILE")
    BaseMettaSetupTest.active_profile_name = active_profile
    try:
        from metta.setup.profiles import UserType

        BaseMettaSetupTest.active_user_type = (
            UserType(active_profile) if active_profile in {u.value for u in UserType} else None
        )
    except Exception:
        BaseMettaSetupTest.active_user_type = None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests based on active profile.

    - Tests marked with @pytest.mark.setup are skipped unless --metta-setup is set
    - Tests marked with @pytest.mark.profile("<name>") are skipped unless --metta-profile matches
    """
    active_profile = config.getoption("--metta-profile") or os.environ.get("METTA_TEST_PROFILE")
    active_setup = config.getoption("--metta-setup") or os.environ.get("METTA_TEST_SETUP")
    setup_dir = Path(__file__).parent.resolve()

    def skip_profile_marker(required: str) -> pytest.MarkDecorator:
        return pytest.mark.skip(
            reason=f"requires profile '{required}'; run with --metta-profile={required} "
            "or METTA_TEST_PROFILE={required}"
        )

    for item in items:
        # Apply only to tests under tests/setup/**
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        try:
            in_setup = item_path.is_relative_to(setup_dir)
        except Exception:
            in_setup = str(item_path).startswith(str(setup_dir))

        if not in_setup:
            continue

        # Only run tests explicitly marked as setup
        if not item.get_closest_marker("setup"):
            item.add_marker(pytest.mark.skip(reason="requires @pytest.mark.setup"))
            continue

        profile_marker = item.get_closest_marker("profile")

        if profile_marker and profile_marker.args:
            required_profile = str(profile_marker.args[0])
            if active_setup is None or (active_profile is None or required_profile != active_profile):
                item.add_marker(skip_profile_marker(required_profile))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Get the repository root directory."""
    return get_repo_root()


@pytest.fixture(scope="function")
def temp_test_env() -> Generator[Path, None, None]:
    """Create a temporary test environment."""
    temp_dir = tempfile.mkdtemp(prefix="metta_test_")
    test_home = Path(temp_dir) / "home"
    test_home.mkdir(parents=True, exist_ok=True)

    # Store original environment variables
    original_home = os.environ.get("HOME")
    original_config_path = os.environ.get("METTA_CONFIG_PATH")
    original_zdotdir = os.environ.get("ZDOTDIR")
    original_test_env = os.environ.get("METTA_TEST_ENV")

    # Set up test environment
    os.environ["HOME"] = str(test_home)
    os.environ["METTA_TEST_ENV"] = "1"

    # Create shell config files for testing
    zshrc = test_home / ".zshrc"
    bashrc = test_home / ".bashrc"
    zshrc.write_text("# Test zshrc\n")
    bashrc.write_text("# Test bashrc\n")

    yield test_home

    # Clean up
    if original_home:
        os.environ["HOME"] = original_home
    if original_config_path:
        os.environ["METTA_CONFIG_PATH"] = original_config_path
    else:
        os.environ.pop("METTA_CONFIG_PATH", None)
    if original_zdotdir:
        os.environ["ZDOTDIR"] = original_zdotdir
    else:
        os.environ.pop("ZDOTDIR", None)
    if original_test_env:
        os.environ["METTA_TEST_ENV"] = original_test_env
    else:
        os.environ.pop("METTA_TEST_ENV", None)

    # Remove temporary directory
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_config(test_env: Path) -> Path:
    """Create a clean configuration directory."""
    config_dir = test_env / ".metta"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
