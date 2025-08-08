#!/usr/bin/env python3
"""
Integration tests for `metta install` command.

These tests ensure that the installation process works correctly for different profiles
and catches issues that might not be noticed when developers aren't building from scratch.
"""

import unittest

import pytest

from metta.setup.config import SetupConfig, UserType
from tests.setup.test_base import BaseMettaSetupTest


@pytest.mark.setup
@pytest.mark.profile("softmax")
class TestInstallSoftmax(BaseMettaSetupTest):
    """Integration tests for metta install with softmax profile (condensed)."""

    def test_softmax_end_to_end_flow(self):
        """Run a condensed end-to-end flow for softmax profile in one test."""
        self._create_test_config(UserType.SOFTMAX)

        # Full install
        r_install_all = self._run_metta_command(["install"])
        assert r_install_all.returncode == 0

        # Install specific components
        r_install_some = self._run_metta_command(["install", "core", "system", "aws"])
        assert r_install_some.returncode == 0

        # Force re-install
        r_force = self._run_metta_command(["install", "--force"])
        assert r_force.returncode == 0

        # Status (non-interactive)
        r_status = self._run_metta_command(["status", "--non-interactive"])
        assert r_status.returncode == 0
        assert "Component" in r_status.stdout

        # Configure and run githooks
        r_cfg = self._run_metta_command(["configure", "githooks"])
        assert r_cfg.returncode == 0
        r_install_githooks = self._run_metta_command(["install", "githooks"])
        assert r_install_githooks.returncode == 0
        r_run = self._run_metta_command(["run", "githooks", "pre-commit"])
        assert r_run.returncode in [0, 1]

        # Clean and symlink setup
        r_clean = self._run_metta_command(["clean"])
        assert r_clean.returncode == 0
        r_symlink = self._run_metta_command(["symlink-setup"])
        assert r_symlink.returncode == 0

        # Verify config written
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.SOFTMAX

    def test_install_without_config_fails(self):
        """Installing without a config should fail in non-interactive tests."""
        result = self._run_metta_command(["install"])
        assert result.returncode == 1

    def test_install_once_components(self):
        """Installing install_once components repeatedly should be idempotent."""
        self._create_test_config(UserType.SOFTMAX)
        r1 = self._run_metta_command(["install", "aws"])
        assert r1.returncode == 0
        r2 = self._run_metta_command(["install", "aws"])
        assert r2.returncode == 0


@pytest.mark.setup
@pytest.mark.profile("external")
class TestInstallExternal(BaseMettaSetupTest):
    def test_install_softmax_profile(self):
        self._create_test_config(UserType.EXTERNAL)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"


@pytest.mark.setup
@pytest.mark.profile("cloud")
class TestInstallCloud(BaseMettaSetupTest):
    def test_install_cloud_profile(self):
        self._create_test_config(UserType.CLOUD)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.CLOUD
        assert config.is_component_enabled("aws")


@pytest.mark.setup
@pytest.mark.profile("custom")
class TestInstallCustom(BaseMettaSetupTest):
    def test_install_custom_profile(self):
        self._create_test_config(UserType.EXTERNAL, custom_config=True)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.EXTERNAL
        assert config.is_custom_config


@pytest.mark.setup
@pytest.mark.profile("external")
class TestFromScratchExternal(BaseMettaSetupTest):
    def test_fresh_install_external(self):
        """Test fresh installation with external profile."""
        self._create_test_config(UserType.EXTERNAL, custom_config=True)
        install_result = self._run_metta_command(["install"])
        assert install_result.returncode == 0, f"Install failed: {install_result.stderr}"


@pytest.mark.setup
@pytest.mark.profile("cloud")
class TestFromScratchCloud(BaseMettaSetupTest):
    def test_fresh_install_cloud(self):
        """Test fresh installation with cloud profile."""
        self._create_test_config(UserType.CLOUD)
        install_result = self._run_metta_command(["install"])
        assert install_result.returncode == 0, f"Install failed: {install_result.stderr}"


if __name__ == "__main__":
    unittest.main()
