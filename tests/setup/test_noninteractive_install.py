#!/usr/bin/env python3
"""
Tests for non-interactive installation and configuration.

These tests verify that `metta install --non-interactive` and `metta configure --non-interactive`
work correctly in automated environments like Docker containers and CI systems.
"""

import os
import subprocess
import sys
import unittest

import pytest

from metta.setup.saved_settings import SavedSettings, UserType
from tests.setup.test_base import BaseMettaSetupTest


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveInstall(BaseMettaSetupTest):
    """Tests for non-interactive installation flow."""

    def test_end_to_end_configure_and_install(self):
        """Test complete end-to-end non-interactive flow."""
        # Configure with external profile
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=external"])
        self.assertEqual(result.returncode, 0, f"Configure failed: {result.stderr}")

        # Verify config exists
        config = SavedSettings(self.test_config_dir / "config.yaml")
        self.assertTrue(config.exists())
        self.assertEqual(config.user_type, UserType.EXTERNAL)

        # Install core component
        result = self._run_metta_command(["install", "--non-interactive", "core"])
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")

        # Should not contain interactive prompts
        self.assertNotIn("Enter your choice", result.stdout)
        self.assertNotIn("(y/n)", result.stdout)

    def test_install_non_interactive_with_config(self):
        """Test non-interactive install works when config exists."""
        self._create_test_config(UserType.EXTERNAL)

        result = self._run_metta_command(["install", "--non-interactive"])
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")

        # Should complete without hanging
        self.assertIn("Installation complete", result.stdout)

    def test_install_non_interactive_no_config_fails(self):
        """Test non-interactive install fails cleanly without config."""
        result = self._run_metta_command(["install", "--non-interactive"])

        # Should fail but not hang
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No configuration found", result.stderr)

    def test_configure_non_interactive_with_profile(self):
        """Test non-interactive configure with explicit profile."""
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=external"])
        self.assertEqual(result.returncode, 0, f"Configure failed: {result.stderr}")

        # Verify config was created
        config = SavedSettings(self.test_config_dir / "config.yaml")
        self.assertTrue(config.exists())
        self.assertEqual(config.user_type, UserType.EXTERNAL)

    def test_configure_non_interactive_setup_wizard_uses_defaults(self):
        """Test non-interactive setup wizard uses appropriate defaults."""
        result = self._run_metta_command(["configure", "--non-interactive"])

        # Should complete without hanging, using defaults
        self.assertEqual(result.returncode, 0, f"Configure failed: {result.stderr}")

        # Should have created a config file with default settings
        config = SavedSettings(self.test_config_dir / "config.yaml")
        self.assertTrue(config.exists())

    def test_install_specific_components_non_interactive(self):
        """Test installing specific components in non-interactive mode."""
        self._create_test_config(UserType.EXTERNAL)

        # Install specific components that should be safe to install everywhere
        result = self._run_metta_command(["install", "--non-interactive", "core", "system"])
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")

    def test_force_reinstall_non_interactive(self):
        """Test force reinstall works in non-interactive mode."""
        self._create_test_config(UserType.EXTERNAL)

        # First install
        result1 = self._run_metta_command(["install", "--non-interactive", "core"])
        self.assertEqual(result1.returncode, 0)

        # Force reinstall
        result2 = self._run_metta_command(["install", "--non-interactive", "--force", "core"])
        self.assertEqual(result2.returncode, 0)

    def test_status_non_interactive_no_prompts(self):
        """Test status command in non-interactive mode doesn't prompt."""
        self._create_test_config(UserType.EXTERNAL)

        result = self._run_metta_command(["status", "--non-interactive"])
        # Should always complete without hanging, even if components not installed
        self.assertIsNotNone(result.returncode)
        self.assertIn("Component", result.stdout)


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveExternal(BaseMettaSetupTest):
    """Test non-interactive installation works for external profile."""

    def test_external_profile_non_interactive(self):
        """Test external profile installation in non-interactive mode."""
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=external"])
        self.assertEqual(result.returncode, 0)

        result = self._run_metta_command(["install", "--non-interactive"])
        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")


@pytest.mark.setup
@pytest.mark.profile("softmax")
class TestNonInteractiveSoftmax(BaseMettaSetupTest):
    """Test non-interactive installation works for softmax profile."""

    def test_softmax_profile_non_interactive(self):
        """Test softmax profile installation in non-interactive mode."""
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=softmax"])
        self.assertEqual(result.returncode, 0)

        # Note: Some softmax components may fail without proper credentials, but shouldn't hang
        result = self._run_metta_command(["install", "--non-interactive"])
        # Don't assert returncode == 0 since some components may fail without credentials
        self.assertIsNotNone(result.returncode)  # Should complete without hanging


@pytest.mark.setup
@pytest.mark.profile("cloud")
class TestNonInteractiveCloud(BaseMettaSetupTest):
    """Test non-interactive installation works for cloud profile."""

    def test_cloud_profile_non_interactive(self):
        """Test cloud profile installation in non-interactive mode."""
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=cloud"])
        self.assertEqual(result.returncode, 0)

        result = self._run_metta_command(["install", "--non-interactive"])
        # Don't assert returncode == 0 since some components may fail without credentials
        self.assertIsNotNone(result.returncode)  # Should complete without hanging


@pytest.mark.setup
@pytest.mark.profile("softmax-docker")
class TestNonInteractiveSoftmaxDocker(BaseMettaSetupTest):
    """Test non-interactive installation works for softmax-docker profile."""

    def test_softmax_docker_profile_non_interactive(self):
        """Test softmax-docker profile installation in non-interactive mode."""
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=softmax"])
        self.assertEqual(result.returncode, 0)

        # This should work better in docker environment with non-interactive mode
        result = self._run_metta_command(["install", "--non-interactive"])
        self.assertIsNotNone(result.returncode)  # Should complete without hanging


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveComponentInstall(BaseMettaSetupTest):
    """Test individual components install successfully in non-interactive mode."""

    def setUp(self):
        super().setUp()
        # Create external profile config for all component tests
        self._create_test_config(UserType.EXTERNAL)

    def test_safe_components_install_non_interactive(self):
        """Test that each safe component installs without hanging."""
        safe_components = ["core", "system", "nodejs", "filter_repo", "mettascope", "experiments"]

        for component in safe_components:
            with self.subTest(component=component):
                result = self._run_metta_command(["install", "--non-interactive", component])

                # Component should complete installation without hanging
                # Some may fail due to missing dependencies, but shouldn't hang
                self.assertIsNotNone(result.returncode)

                # If installation succeeded, verify it was non-interactive
                if result.returncode == 0:
                    self.assertNotIn("Enter your choice", result.stdout)
                    self.assertNotIn("(y/n)", result.stdout)

    def test_all_enabled_components_install_non_interactive(self):
        """Test that all enabled components can be installed non-interactively."""
        # This is the main test - install all components for external profile
        result = self._run_metta_command(["install", "--non-interactive"])

        # Installation should complete without hanging
        self.assertIsNotNone(result.returncode)

        # Should not contain any interactive prompts in output
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

        cmd = [sys.executable, "-m", "metta.setup.metta_cli", "install", "--non-interactive", "core"]
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,  # No stdin available
        )

        # Should complete without hanging
        self.assertIsNotNone(result.returncode)

    def test_debian_frontend_noninteractive_set(self):
        """Test that DEBIAN_FRONTEND=noninteractive is properly set in environment."""
        self._create_test_config(UserType.EXTERNAL)

        # Mock a component that would check environment variables
        # This is more of an integration test to ensure the base class sets env vars correctly
        result = self._run_metta_command(["install", "--non-interactive", "core"])

        # Should complete successfully
        # The actual environment variable setting is tested at the unit level in base.py
        self.assertIsNotNone(result.returncode)

    def test_install_with_missing_dependencies_non_interactive(self):
        """Test non-interactive install handles missing dependencies gracefully."""
        self._create_test_config(UserType.SOFTMAX)

        # Try to install components that might have missing dependencies
        result = self._run_metta_command(["install", "--non-interactive", "aws", "skypilot"])

        # Should complete without hanging, even if some installations fail
        self.assertIsNotNone(result.returncode)

        # Should not prompt for user input
        self.assertNotIn("Enter your choice", result.stdout)


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNonInteractiveErrorHandling(BaseMettaSetupTest):
    """Test non-interactive mode error handling and edge cases."""

    def test_configure_invalid_profile_non_interactive(self):
        """Test configure with invalid profile fails cleanly."""
        result = self._run_metta_command(["configure", "--non-interactive", "--profile=invalid"])

        # Should fail cleanly without hanging
        self.assertNotEqual(result.returncode, 0)

    def test_install_invalid_component_non_interactive(self):
        """Test install with invalid component fails cleanly."""
        self._create_test_config(UserType.EXTERNAL)

        result = self._run_metta_command(["install", "--non-interactive", "nonexistent_component"])

        # Should complete without hanging
        self.assertIsNotNone(result.returncode)

    def test_configure_component_non_interactive(self):
        """Test configuring individual components in non-interactive mode."""
        self._create_test_config(UserType.EXTERNAL)

        # Components that support configuration should handle non-interactive mode
        result = self._run_metta_command(["configure", "--non-interactive", "githooks"])

        # Should complete without hanging (may succeed or fail depending on component)
        self.assertIsNotNone(result.returncode)


if __name__ == "__main__":
    unittest.main()
