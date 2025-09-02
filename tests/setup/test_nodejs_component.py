#!/usr/bin/env python3
"""
Tests for nodejs component setup functionality.

These tests verify:
- Correct dependencies are declared (system component)
- PNPM_HOME and PATH configuration works properly
- Shell profile updates work correctly
- Fresh installation scenarios work as expected
"""

import os
import tempfile
import unittest

import pytest

from metta.setup.components.nodejs import NodejsSetup
from metta.setup.saved_settings import UserType
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
class TestPnpmPathConfiguration(BaseMettaSetupTest):
    """Test PNPM_HOME and PATH configuration logic."""

    def setUp(self):
        super().setUp()
        self.nodejs_setup = NodejsSetup()

    def test_shell_config_paths_respect_zdotdir(self):
        """Test that shell config paths respect ZDOTDIR environment variable."""
        # Test without ZDOTDIR
        os.environ.pop("ZDOTDIR", None)
        paths = self.nodejs_setup._get_shell_config_paths()

        zsh_path = None
        bash_path = None
        for shell_name, config_path in paths:
            if shell_name == "zsh":
                zsh_path = config_path
            elif shell_name == "bash":
                bash_path = config_path

        self.assertIsNotNone(zsh_path)
        self.assertIsNotNone(bash_path)
        self.assertTrue(zsh_path.endswith("/.zshrc"))
        self.assertTrue(bash_path.endswith("/.bashrc"))

        # Test with custom ZDOTDIR
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_zdotdir = os.path.join(temp_dir, "custom_zsh")
            os.environ["ZDOTDIR"] = custom_zdotdir

            paths = self.nodejs_setup._get_shell_config_paths()
            zsh_path = None
            for shell_name, config_path in paths:
                if shell_name == "zsh":
                    zsh_path = config_path
                    break

            self.assertIsNotNone(zsh_path)
            self.assertEqual(zsh_path, os.path.join(custom_zdotdir, ".zshrc"))

    def test_pnpm_config_detection_various_formats(self):
        """Test that PNPM configuration detection handles different formats."""
        pnpm_home = "/Users/testuser/.local/share/pnpm"

        # Test with expanded path format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(f"""# pnpm
export PNPM_HOME="{pnpm_home}"
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac
# pnpm end
""")
            f.flush()

            result = self.nodejs_setup._check_pnpm_config_in_file(f.name, pnpm_home)
            self.assertTrue(result, "Should detect correctly configured pnpm with expanded path")

            os.unlink(f.name)

        # Test with $HOME format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("""# pnpm
export PNPM_HOME="$HOME/.local/share/pnpm"
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac
# pnpm end
""")
            f.flush()

            result = self.nodejs_setup._check_pnpm_config_in_file(f.name, pnpm_home)
            self.assertTrue(result, "Should detect correctly configured pnpm with $HOME format")

            os.unlink(f.name)

        # Test with missing pnpm section
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write('# shell config without pnpm\nexport PATH="/some/path:$PATH"\n')
            f.flush()

            result = self.nodejs_setup._check_pnpm_config_in_file(f.name, pnpm_home)
            self.assertFalse(result, "Should not detect pnpm config when missing")

            os.unlink(f.name)

    def test_pnpm_environment_setup(self):
        """Test that PNPM environment setup works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pnpm_home = os.path.join(temp_dir, "pnpm")

            # Save original environment
            original_pnpm_home = os.environ.get("PNPM_HOME")
            original_path = os.environ.get("PATH", "")

            try:
                # Test setup
                self.nodejs_setup._setup_pnpm_environment(pnpm_home)

                # Verify directory was created
                self.assertTrue(os.path.exists(pnpm_home), "PNPM_HOME directory should be created")

                # Verify environment variables
                self.assertEqual(os.environ.get("PNPM_HOME"), pnpm_home, "PNPM_HOME should be set")
                self.assertIn(pnpm_home, os.environ.get("PATH", ""), "PNPM_HOME should be in PATH")

            finally:
                # Restore original environment
                if original_pnpm_home is not None:
                    os.environ["PNPM_HOME"] = original_pnpm_home
                else:
                    os.environ.pop("PNPM_HOME", None)
                os.environ["PATH"] = original_path


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
        import shutil
        import subprocess

        # Make sure the base is there.
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
        # Step 2: Verify turbo is not available initially (we expect it might still be there)
        self.assertFalse(check_binary_available("turbo"), "turbo should not be available initially")

        # Step 3: Run nodejs installation
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

        # Step 4: Verify all binaries are now available
        pnpm_available_after = check_binary_available("pnpm")
        self.assertTrue(pnpm_available_after, "pnpm should be available after nodejs installation")

        turbo_available_after = check_binary_available("turbo")
        self.assertTrue(turbo_available_after, "turbo should be installed globally by nodejs setup")

        # Step 5: Verify pnpm works
        try:
            result = subprocess.run(["pnpm", "--version"], capture_output=True, text=True, check=True)
            self.assertIsNotNone(result.stdout.strip(), "pnpm should return version")
        except subprocess.CalledProcessError as e:
            self.fail(f"pnpm --version failed: {e}")

        # Step 6: Verify turbo works
        try:
            result = subprocess.run(["turbo", "--version"], capture_output=True, text=True, check=True)
            self.assertIsNotNone(result.stdout.strip(), "turbo should return version")
        except subprocess.CalledProcessError as e:
            self.fail(f"turbo --version failed: {e}")

        # Step 7: Verify PNPM_HOME is set correctly
        pnpm_home = os.environ.get("PNPM_HOME")
        self.assertIsNotNone(pnpm_home, "PNPM_HOME should be set after installation")
        if pnpm_home:
            self.assertTrue(os.path.exists(pnpm_home), "PNPM_HOME directory should exist")

        # Step 8: Verify PNPM_HOME is in PATH
        path_env = os.environ.get("PATH", "")
        self.assertIn(pnpm_home, path_env, "PNPM_HOME should be in PATH")

        # Step 9: Verify root repo dependencies were installed
        node_modules_path = self.repo_root / "node_modules"
        self.assertTrue(node_modules_path.exists(), "node_modules should exist after pnpm install")
        self.assertTrue(node_modules_path.is_dir(), "node_modules should be a directory")

        # Verify some expected dependencies exist (from package.json devDependencies)
        expected_deps = ["prettier", "turbo"]  # Known deps from root package.json
        for dep in expected_deps:
            dep_path = node_modules_path / dep
            if dep_path.exists():
                # At least one expected dependency was found, confirming pnpm install worked
                print(f"✓ Found expected dependency: {dep}")
                break
        else:
            self.fail(
                f"None of the expected dependencies {expected_deps} were found in node_modules. "
                f"This suggests 'pnpm install --frozen-lockfile' did not run properly."
            )

        # Step 10: Critical test - verify pnpm is available in PATH
        pnpm_path = shutil.which("pnpm")
        self.assertIsNotNone(pnpm_path, "pnpm must be available in PATH after nodejs installation")
        if pnpm_path and pnpm_home:
            self.assertTrue(
                pnpm_path.startswith(pnpm_home),
                f"pnpm binary should be from PNPM_HOME ({pnpm_home}), but found at {pnpm_path}",
            )


if __name__ == "__main__":
    unittest.main()
