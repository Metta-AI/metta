#!/usr/bin/env python3
"""
Tests for nodejs component setup functionality.

These tests verify:
- Correct dependencies are declared (system component)
- pnpm setup works correctly
- Global packages are installed and available
- Project dependencies are installed
"""

import os
import shutil
import subprocess
import unittest

import pytest

from softmax.cli.components.nodejs import NodejsSetup
from softmax.cli.saved_settings import UserType
from tests.setup.test_base import BaseMettaSetupTest


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNodejsComponentDependencies(BaseMettaSetupTest):
    """Test nodejs component has correct dependencies."""

    def test_nodejs_requires_system_dependency(self):
        """Test that nodejs component declares system as dependency."""
        nodejs_setup = NodejsSetup()
        dependencies = nodejs_setup.dependencies()

        self.assertIn(
            "system",
            dependencies,
            "nodejs component must depend on system component to ensure Node.js/corepack is available",
        )

    def test_nodejs_component_name(self):
        """Test nodejs component has correct name."""
        nodejs_setup = NodejsSetup()
        self.assertEqual(nodejs_setup.name, "nodejs")


@pytest.mark.setup
@pytest.mark.profile("external")
class TestNodejsInstallationFlow(BaseMettaSetupTest):
    """Test the complete nodejs installation flow."""

    def test_nodejs_component_enabled_in_external_profile(self):
        """Test that nodejs component is enabled in external profile."""
        self._create_test_config(UserType.EXTERNAL)

        nodejs_setup = NodejsSetup()
        self.assertTrue(nodejs_setup.is_enabled(), "nodejs should be enabled for external profile")

    def test_nodejs_binaries_installation_flow(self):
        """Test that nodejs installation provides all required binaries."""
        # Make sure the base is there
        self._create_test_config(UserType.EXTERNAL)
        self._run_metta_command(["install"])

        def check_binary_available(binary_name):
            """Check if a binary is available in PATH."""
            return shutil.which(binary_name) is not None

        def safely_remove_global_package(package_name):
            """Remove a global pnpm package if it exists."""
            try:
                # Check if pnpm is available first
                if not check_binary_available("pnpm"):
                    return

                # Try to uninstall the global package
                subprocess.run(
                    ["pnpm", "remove", "--global", package_name],
                    capture_output=True,
                    check=False,  # Don't fail if package wasn't installed
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass  # Ignore errors - package might not have been installed

        # Step 1: Remove turbo if it exists (pnpm and corepack come with Node.js)
        safely_remove_global_package("turbo")

        # Step 2: Run nodejs installation
        nodejs_setup = NodejsSetup()

        # First ensure system dependency is met (Node.js should be installed)
        node_available = check_binary_available("node")
        self.assertTrue(node_available, "Node.js should be installed by system component dependency")

        # Corepack should be available (comes with Node.js)
        corepack_available = check_binary_available("corepack")
        self.assertTrue(corepack_available, "corepack should be available with Node.js installation")

        # Run the nodejs setup
        try:
            nodejs_setup.install()
        except Exception as e:
            self.fail(f"nodejs installation failed: {e}")

        # Step 3: Verify all binaries are now available
        pnpm_available_after = check_binary_available("pnpm")
        self.assertTrue(pnpm_available_after, "pnpm should be available after nodejs installation")

        turbo_available_after = check_binary_available("turbo")
        self.assertTrue(turbo_available_after, "turbo should be installed globally by nodejs setup")

        # Step 4: Verify pnpm works
        try:
            result = subprocess.run(["pnpm", "--version"], capture_output=True, text=True, check=True)
            self.assertIsNotNone(result.stdout.strip(), "pnpm should return version")
        except subprocess.CalledProcessError as e:
            self.fail(f"pnpm --version failed: {e}")

        # Step 5: Verify turbo works
        try:
            result = subprocess.run(["turbo", "--version"], capture_output=True, text=True, check=True)
            self.assertIsNotNone(result.stdout.strip(), "turbo should return version")
        except subprocess.CalledProcessError as e:
            self.fail(f"turbo --version failed: {e}")

        # Step 6: Verify project dependencies were installed
        node_modules_path = self.repo_root / "node_modules"
        self.assertTrue(node_modules_path.exists(), "node_modules should exist after pnpm install")
        self.assertTrue(node_modules_path.is_dir(), "node_modules should be a directory")

        # Verify some expected dependencies exist (from package.json devDependencies)
        expected_deps = ["prettier", "turbo"]  # Known deps from root package.json
        for dep in expected_deps:
            dep_path = node_modules_path / dep
            if dep_path.exists():
                # At least one expected dependency was found, confirming pnpm install worked
                break
        else:
            self.fail(
                f"None of the expected dependencies {expected_deps} were found in node_modules. "
                f"This suggests 'pnpm install --frozen-lockfile' did not run properly."
            )

        # Step 7: Verify pnpm setup worked by checking that PNPM_HOME is set in current process
        # (The nodejs component should have set this after running pnpm setup)
        pnpm_home_env = os.environ.get("PNPM_HOME")
        if pnpm_home_env:
            self.assertTrue(os.path.exists(pnpm_home_env), f"PNPM_HOME directory should exist at {pnpm_home_env}")
        # If PNPM_HOME isn't set, check the standard location
        else:
            pnpm_home = os.path.expanduser("~/.local/share/pnpm")
            # This might not exist in test environment, so just warn if missing
            if not os.path.exists(pnpm_home):
                print(f"Warning: pnpm setup did not create {pnpm_home}, but pnpm is still working")

        # Step 8: Verify global packages are accessible
        pnpm_path = shutil.which("pnpm")
        self.assertIsNotNone(pnpm_path, "pnpm must be available in PATH after nodejs installation")

        turbo_path = shutil.which("turbo")
        self.assertIsNotNone(turbo_path, "turbo must be available in PATH after nodejs installation")


if __name__ == "__main__":
    unittest.main()
