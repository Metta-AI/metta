"""Tests for constants in metta.common.util.constants."""

import os
import pathlib

import metta.common.util.constants


class TestConstants:
    """Test that constants are properly configured."""

    def test_skypilot_launch_path_exists(self):
        """Test that SKYPILOT_LAUNCH_PATH points to an existing file."""
        # Convert to Path for easier testing
        launch_path = pathlib.Path(metta.common.util.constants.SKYPILOT_LAUNCH_PATH)

        # Check that the path exists
        assert launch_path.exists(), (
            f"SKYPILOT_LAUNCH_PATH does not exist: {metta.common.util.constants.SKYPILOT_LAUNCH_PATH}"
        )

        # Check that it's a file (not a directory)
        assert launch_path.is_file(), (
            f"SKYPILOT_LAUNCH_PATH is not a file: {metta.common.util.constants.SKYPILOT_LAUNCH_PATH}"
        )

        # Check that the file is readable
        assert os.access(launch_path, os.R_OK), (
            f"SKYPILOT_LAUNCH_PATH is not readable: {metta.common.util.constants.SKYPILOT_LAUNCH_PATH}"
        )

    def test_skypilot_launch_path_is_absolute(self):
        """Test that SKYPILOT_LAUNCH_PATH is an absolute path."""
        launch_path = pathlib.Path(metta.common.util.constants.SKYPILOT_LAUNCH_PATH)
        assert launch_path.is_absolute(), (
            f"SKYPILOT_LAUNCH_PATH should be absolute: {metta.common.util.constants.SKYPILOT_LAUNCH_PATH}"
        )

    def test_skypilot_launch_path_is_python_file(self):
        """Test that SKYPILOT_LAUNCH_PATH points to a Python file."""
        launch_path = pathlib.Path(metta.common.util.constants.SKYPILOT_LAUNCH_PATH)
        assert launch_path.suffix == ".py", (
            f"SKYPILOT_LAUNCH_PATH should be a .py file: {metta.common.util.constants.SKYPILOT_LAUNCH_PATH}"
        )
