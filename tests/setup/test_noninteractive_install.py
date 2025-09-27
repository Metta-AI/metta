#!/usr/bin/env python3
"""
Tests for non-interactive behavior in automated environments.

These tests verify specific non-interactive behaviors that aren't covered by the
automatic --non-interactive flag in test_base.py:
- Output content verification (no interactive prompts in output)
- Environment simulation (Docker/CI conditions)
- Component exclusion behaviors in test environments
- Individual component testing per profile
- Error handling edge cases
"""

import os
import subprocess
import sys
import unittest

import pytest

from softmax.cli.profiles import PROFILE_DEFINITIONS, UserType
from tests.setup.test_base import BaseMettaSetupTest


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveOutputVerification(BaseMettaSetupTest):
    """Test that non-interactive mode produces expected output without interactive prompts."""

    def test_install_non_interactive_no_config_fails(self):
        """Test non-interactive install fails cleanly without config."""
        result = self._run_metta_command(["install"])

        # Should fail but not hang
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Must specify a profile",
            result.stderr + result.stdout,
        )

    def test_output_contains_no_interactive_prompts(self):
        """Test that installation output contains no interactive prompts."""
        self._create_test_config(UserType.EXTERNAL)

        result = self._run_metta_command(["install", "core"])
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")

        # Should not contain interactive prompts in output
        self.assertNotIn("Enter your choice", result.stdout)
        self.assertNotIn("(y/n)", result.stdout)
        self.assertNotIn("Do you have your API key ready", result.stdout)


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveEnvironmentHandling(BaseMettaSetupTest):
    """Test non-interactive mode handles different environment conditions."""

    def test_no_tty_environment(self):
        """Test installation works without TTY (Docker condition)."""
        env = os.environ.copy()
        env["CI"] = "1"  # Simulate CI environment
        env.pop("TERM", None)  # Remove terminal info

        self._create_test_config(UserType.EXTERNAL)

        cmd = [sys.executable, "-m", "softmax.cli.metta_cli", "install", "--non-interactive", "core"]
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,  # No stdin available
        )

        # Should complete without hanging
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveComponentExclusions(BaseMettaSetupTest):
    """Test component exclusion behaviors in test environments."""

    def test_authentication_components_skip_auth_in_test_env(self):
        """Test that authentication components handle test environment correctly."""
        self._create_test_config(UserType.EXTERNAL)

        result = self._run_metta_command(["install", "wandb"])
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")
        self.assertNotIn("Do you have your API key ready", result.stdout)


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveIndividualComponents(BaseMettaSetupTest):
    """Test that each component from profile configurations can be installed individually."""

    def test_external_profile_individual_components(self):
        """Test that each component in external profile can be installed individually."""
        self._create_test_config(UserType.EXTERNAL)

        # Get all enabled components for external profile
        profile_config = PROFILE_DEFINITIONS[UserType.EXTERNAL]
        enabled_components = [name for name, config in profile_config["components"].items() if config["enabled"]]

        self.assertGreater(len(enabled_components), 0, "External profile should have enabled components")

        # Test each component individually
        for component in enabled_components:
            with self.subTest(component=component):
                result = self._run_metta_command(["install", component])

                # Should complete without hanging and return success
                self.assertEqual(result.returncode, 0, f"Component '{component}' failed to install: {result.stderr}")


@pytest.mark.setup
@pytest.mark.profile("cloud")
class TestNonInteractiveIndividualComponentsCloud(BaseMettaSetupTest):
    """Test that each component from cloud profile can be installed individually."""

    def test_cloud_profile_individual_components(self):
        """Test that each component in cloud profile can be installed individually."""
        self._create_test_config(UserType.CLOUD)

        # Get all enabled components for cloud profile
        profile_config = PROFILE_DEFINITIONS[UserType.CLOUD]
        enabled_components = [name for name, config in profile_config["components"].items() if config["enabled"]]

        self.assertGreater(len(enabled_components), 0, "Cloud profile should have enabled components")

        # Test each component individually
        for component in enabled_components:
            with self.subTest(component=component):
                result = self._run_metta_command(["install", component])

                # Should complete without hanging and return success
                self.assertEqual(result.returncode, 0, f"Component '{component}' failed to install: {result.stderr}")


@pytest.mark.setup
@pytest.mark.profile("softmax")
class TestNonInteractiveIndividualComponentsSoftmax(BaseMettaSetupTest):
    """Test that each component from softmax profile can be installed individually."""

    def test_softmax_profile_individual_components(self):
        """Test that each component in softmax profile can be installed individually."""
        self._create_test_config(UserType.SOFTMAX)

        # Get all enabled components for softmax profile
        profile_config = PROFILE_DEFINITIONS[UserType.SOFTMAX]
        enabled_components = [name for name, config in profile_config["components"].items() if config["enabled"]]

        self.assertGreater(len(enabled_components), 0, "Softmax profile should have enabled components")

        # Test each component individually
        for component in enabled_components:
            with self.subTest(component=component):
                result = self._run_metta_command(["install", component])

                # Should complete without hanging and return success
                self.assertEqual(result.returncode, 0, f"Component '{component}' failed to install: {result.stderr}")


@pytest.mark.setup
@pytest.mark.profile("softmax-docker")
class TestNonInteractiveIndividualComponentsSoftmaxDocker(BaseMettaSetupTest):
    """Test that each component from softmax-docker profile can be installed individually."""

    def test_softmax_docker_profile_individual_components(self):
        """Test that each component in softmax-docker profile can be installed individually."""
        self._create_test_config(UserType.SOFTMAX_DOCKER)

        # Get all enabled components for softmax-docker profile
        profile_config = PROFILE_DEFINITIONS[UserType.SOFTMAX_DOCKER]
        enabled_components = [name for name, config in profile_config["components"].items() if config["enabled"]]

        self.assertGreater(len(enabled_components), 0, "Softmax-docker profile should have enabled components")

        # Test each component individually
        for component in enabled_components:
            with self.subTest(component=component):
                result = self._run_metta_command(["install", component])

                # Should complete without hanging and return success
                self.assertEqual(result.returncode, 0, f"Component '{component}' failed to install: {result.stderr}")


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveErrorHandling(BaseMettaSetupTest):
    """Test non-interactive mode error handling and edge cases."""

    def test_configure_invalid_profile_non_interactive(self):
        """Test configure with invalid profile fails cleanly."""
        result = self._run_metta_command(["configure", "--profile=invalid"])

        # Should fail cleanly without hanging
        self.assertNotEqual(result.returncode, 0)

    def test_configure_component_non_interactive(self):
        """Test configuring individual components in non-interactive mode."""
        self._create_test_config(UserType.EXTERNAL)

        # Components that support configuration should handle non-interactive mode
        result = self._run_metta_command(["configure", "githooks"])

        # Should complete without hanging (may succeed or fail depending on component)
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
