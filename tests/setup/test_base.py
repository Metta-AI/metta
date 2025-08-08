import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from metta.setup.profiles import UserType


class BaseMettaSetupTest(unittest.TestCase):
    """Base class for metta setup installer component tests.

    Provides common setup and teardown logic for testing metta setup components
    in an isolated environment with temporary home directory and configuration.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for the entire test class."""
        super().setUpClass()

        # Create temporary directory for the class
        cls.temp_dir = tempfile.mkdtemp(prefix="metta_test_class_")

        # Store original environment variables
        cls.original_home = os.environ.get("HOME")
        cls.original_config_path = os.environ.get("METTA_CONFIG_PATH")
        cls.original_zdotdir = os.environ.get("ZDOTDIR")

        # Set up test home directory
        cls.test_home = Path(cls.temp_dir) / "home"
        cls.test_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(cls.test_home)

        # Set test environment flag to prevent browser login
        os.environ["METTA_TEST_ENV"] = "1"

        # Clear ZDOTDIR for tests to use default location
        os.environ.pop("ZDOTDIR", None)

        # Set up test config directory
        cls.test_config_dir = cls.test_home / ".metta"
        cls.test_config_dir.mkdir(parents=True, exist_ok=True)

        # Create shell config files for testing
        cls.zshrc = cls.test_home / ".zshrc"
        cls.bashrc = cls.test_home / ".bashrc"
        cls.zshrc.write_text("# Test zshrc\n")
        cls.bashrc.write_text("# Test bashrc\n")

        # Get repository root
        cls.repo_root = Path(__file__).parent.parent.parent

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after the entire test class."""
        if cls.original_home:
            os.environ["HOME"] = cls.original_home
        if cls.original_config_path:
            os.environ["METTA_CONFIG_PATH"] = cls.original_config_path
        else:
            os.environ.pop("METTA_CONFIG_PATH", None)
        if cls.original_zdotdir:
            os.environ["ZDOTDIR"] = cls.original_zdotdir
        else:
            os.environ.pop("ZDOTDIR", None)
        # Clean up test environment variable
        os.environ.pop("METTA_TEST_ENV", None)

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

        super().tearDownClass()

    def setUp(self):
        """Set up for each individual test method."""
        super().setUp()
        # Clear any previous test state
        self._clear_test_state()

    def tearDown(self):
        """Clean up after each individual test method."""
        self._clear_test_state()
        super().tearDown()

    def _clear_test_state(self):
        """Clear test state between individual tests."""
        # Clear any config files created by previous tests
        if hasattr(self, "test_config_dir") and self.test_config_dir.exists():
            for config_file in self.test_config_dir.glob("*.yaml"):
                config_file.unlink(missing_ok=True)

        # Clear shell config files and recreate them
        if hasattr(self, "zshrc"):
            self.zshrc.write_text("# Test zshrc\n")
        if hasattr(self, "bashrc"):
            self.bashrc.write_text("# Test bashrc\n")

        # Clear AWS config if it exists
        aws_dir = self.test_home / ".aws"
        if aws_dir.exists():
            import shutil

            shutil.rmtree(aws_dir, ignore_errors=True)

    def _create_test_config(self, user_type: UserType, custom_config: bool = False) -> None:
        """Create a test configuration file.

        Args:
            user_type: The user type to configure
            custom_config: Whether to use custom configuration mode
        """
        config_data = {
            "user_type": user_type.value,
            "custom_config": custom_config,
            "config_version": 1,
        }

        if custom_config:
            # Add component configurations
            from metta.setup.profiles import PROFILE_DEFINITIONS

            profile_config = PROFILE_DEFINITIONS.get(user_type, {})
            config_data["components"] = profile_config.get("components", {})

        config_file = self.test_config_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

    def _run_metta_command(self, args: list[str]) -> "subprocess.CompletedProcess[str]":
        """Run metta command and return result.

        Args:
            args: Command arguments to pass to metta CLI

        Returns:
            CompletedProcess with stdout, stderr, and returncode
        """
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "metta.setup.metta_cli"] + args
        return subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

    def _get_zshrc_path(self) -> Path:
        """Get the correct path for .zshrc based on ZDOTDIR."""
        if os.environ.get("ZDOTDIR"):
            zdotdir = Path(os.environ["ZDOTDIR"])
            return zdotdir / ".zshrc"
        else:
            return self.test_home / ".zshrc"

    def _get_bashrc_path(self) -> Path:
        """Get the correct path for .bashrc."""
        return self.test_home / ".bashrc"

    def _check_shell_config_contains(self, shell_file: Path, expected_line: str) -> bool:
        """Check if shell config file contains expected line.

        Args:
            shell_file: Path to shell config file
            expected_line: Line to search for

        Returns:
            True if line is found, False otherwise
        """
        if not shell_file.exists():
            return False
        content = shell_file.read_text()
        return expected_line in content

    def _set_env_var(self, var_name: str, value: str) -> None:
        """Set an environment variable for testing.

        Args:
            var_name: Name of the environment variable to set
            value: Value to set for the environment variable
        """
        os.environ[var_name] = value
