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


@pytest.mark.profile("softmax")
class TestInstallSoftmax(BaseMettaSetupTest):
    """Integration tests for metta install with external profile."""

    def test_install_external_profile(self):
        self._create_test_config(UserType.EXTERNAL)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.EXTERNAL
        assert not config.is_custom_config

    def test_install_specific_components(self):
        self._create_test_config(UserType.SOFTMAX)
        result = self._run_metta_command(["install", "core", "system", "aws"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"

    def test_install_with_force(self):
        self._create_test_config(UserType.SOFTMAX)
        result1 = self._run_metta_command(["install"])
        assert result1.returncode == 0
        result2 = self._run_metta_command(["install", "--force"])
        assert result2.returncode == 0

    def test_install_without_config(self):
        result = self._run_metta_command(["install"])
        assert result.returncode == 1, "Command should not complete. Unconfigured."

    def test_install_once_components(self):
        self._create_test_config(UserType.SOFTMAX)
        result1 = self._run_metta_command(["install", "aws"])
        assert result1.returncode == 0
        result2 = self._run_metta_command(["install", "aws"])
        assert result2.returncode == 0

    def test_install_status_check(self):
        self._create_test_config(UserType.SOFTMAX)
        install_result = self._run_metta_command(["install"])
        assert install_result.returncode == 0
        status_result = self._run_metta_command(["status", "--non-interactive"])
        assert status_result.returncode == 0
        assert "Component" in status_result.stdout
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.SOFTMAX

    def test_configure_component(self):
        self._create_test_config(UserType.SOFTMAX)
        result = self._run_metta_command(["configure", "githooks"])
        assert result.returncode == 0, f"Configure failed: {result.stderr}"

    def test_run_component_command(self):
        self._create_test_config(UserType.SOFTMAX)
        install_result = self._run_metta_command(["install", "githooks"])
        assert install_result.returncode == 0
        run_result = self._run_metta_command(["run", "githooks", "pre-commit"])
        assert run_result.returncode in [0, 1]

    def test_clean_command(self):
        self._create_test_config(UserType.SOFTMAX)
        result = self._run_metta_command(["clean"])
        assert result.returncode == 0, f"Clean failed: {result.stderr}"

    def test_symlink_setup(self):
        self._create_test_config(UserType.SOFTMAX)
        result = self._run_metta_command(["symlink-setup"])
        assert result.returncode == 0, f"Symlink setup failed: {result.stderr}"


@pytest.mark.profile("external")
class TestInstallExternal(BaseMettaSetupTest):
    def test_install_softmax_profile(self):
        self._create_test_config(UserType.EXTERNAL)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"


@pytest.mark.profile("cloud")
class TestInstallCloud(BaseMettaSetupTest):
    def test_install_cloud_profile(self):
        self._create_test_config(UserType.CLOUD)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.CLOUD
        assert config.is_component_enabled("aws")


@pytest.mark.profile("custom")
class TestInstallCustom(BaseMettaSetupTest):
    def test_install_custom_profile(self):
        self._create_test_config(UserType.EXTERNAL, custom_config=True)
        result = self._run_metta_command(["install"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"
        config = SetupConfig(self.test_config_dir / "config.yaml")
        assert config.user_type == UserType.EXTERNAL
        assert config.is_custom_config


@pytest.mark.profile("external")
class TestFromScratchExternal(BaseMettaSetupTest):
    def test_fresh_install_external(self):
        """Test fresh installation with external profile."""
        self._create_test_config(UserType.EXTERNAL, custom_config=True)
        install_result = self._run_metta_command(["install"])
        assert install_result.returncode == 0, f"Install failed: {install_result.stderr}"


@pytest.mark.profile("cloud")
class TestFromScratchCloud(BaseMettaSetupTest):
    def test_fresh_install_cloud(self):
        """Test fresh installation with cloud profile."""
        self._create_test_config(UserType.CLOUD)
        install_result = self._run_metta_command(["install"])
        assert install_result.returncode == 0, f"Install failed: {install_result.stderr}"


if __name__ == "__main__":
    unittest.main()
