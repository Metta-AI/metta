#!/usr/bin/env python3
"""
Specific tests for AWS profile setup functionality.

These tests focus on the AWS profile configuration, particularly ensuring
that `export AWS_PROFILE=softmax` is added to shell config files when
using the softmax profile.
"""

import pytest

from metta.setup.profiles import UserType
from tests.setup.test_base import BaseMettaSetupTest


class AWSAssertionsMixin:
    """Helpers for asserting AWS-related filesystem state in tests."""

    def _check_aws_config_exists(self) -> bool:  # type: ignore[override]
        aws_dir = self.test_home / ".aws"  # type: ignore[attr-defined]
        if not aws_dir.exists():
            return False
        config_file = aws_dir / "config"
        return config_file.exists()

    def _check_aws_profile_config(self, profile_name: str) -> bool:  # type: ignore[override]
        config_file = self.test_home / ".aws" / "config"  # type: ignore[attr-defined]
        if not config_file.exists():
            return False
        content = config_file.read_text()
        return f"[profile {profile_name}]" in content


@pytest.mark.setup
@pytest.mark.profile("softmax")
class TestAWSProfileSoftmax(AWSAssertionsMixin, BaseMettaSetupTest):
    """Tests for AWS profile setup functionality."""

    def test_softmax_profile_aws_installation(self):
        """Test that softmax profile AWS installation works correctly."""
        self._create_test_config(UserType.SOFTMAX)

        # Run AWS install (bypass base class mocking)

        result = self._run_metta_command(["install", "aws", "--force"])
        assert result.returncode == 0, f"AWS install failed: {result.stderr}"

        # Check that AWS_PROFILE export was added to shell config
        zshrc_path = self._get_zshrc_path()
        bashrc_path = self._get_bashrc_path()

        assert self._check_shell_config_contains(zshrc_path, "export AWS_PROFILE=softmax"), (
            f"AWS_PROFILE export should be added to {zshrc_path} for softmax profile"
        )

        # Also check .bashrc
        assert self._check_shell_config_contains(bashrc_path, "export AWS_PROFILE=softmax"), (
            f"AWS_PROFILE export should be added to {bashrc_path} for softmax profile"
        )

        # Check that AWS config exists
        assert self._check_aws_config_exists(), "AWS config should be created"

        # Check that specific profiles are configured
        assert self._check_aws_profile_config("softmax"), "softmax profile should be configured"
        assert self._check_aws_profile_config("softmax-admin"), "softmax-admin profile should be configured"

        # Check the exact format of the export
        zshrc_content = zshrc_path.read_text()
        assert "export AWS_PROFILE=softmax" in zshrc_content, (
            "AWS_PROFILE export should be exactly 'export AWS_PROFILE=softmax'"
        )

        # Check that it's on its own line
        lines = zshrc_content.split("\n")
        assert "export AWS_PROFILE=softmax" in lines, "AWS_PROFILE export should be on its own line"

        # Check that the setup script was mentioned in output
        assert "Running AWS profile setup" in result.stdout, (
            "Should mention running AWS profile setup for softmax profile"
        )

    def test_softmax_profile_aws_installation_with_zdotdir(self):
        """Test that softmax profile AWS installation works with ZDOTDIR set."""
        # Set up ZDOTDIR
        zdotdir = self.test_home / "config" / "zsh"
        zdotdir.mkdir(parents=True, exist_ok=True)
        self._set_env_var("ZDOTDIR", str(zdotdir))

        # Create shell config files in ZDOTDIR (zsh) and HOME (bash)
        zshrc = zdotdir / ".zshrc"
        bashrc = self.test_home / ".bashrc"
        zshrc.write_text("# Test zshrc in ZDOTDIR\n")
        bashrc.write_text("# Test bashrc in $HOME\n")

        self._create_test_config(UserType.SOFTMAX)

        # Run AWS install (bypass base class mocking)
        result = self._run_metta_command(["install", "aws", "--force"])
        assert result.returncode == 0, f"AWS install failed: {result.stderr}"

        # Check that AWS_PROFILE export was added to shell config in ZDOTDIR
        zshrc_path = self._get_zshrc_path()
        bashrc_path = self._get_bashrc_path()

        assert self._check_shell_config_contains(zshrc_path, "export AWS_PROFILE=softmax"), (
            f"AWS_PROFILE export should be added to {zshrc_path} for softmax profile with ZDOTDIR"
        )

        # Also check .bashrc in HOME (not ZDOTDIR)
        assert self._check_shell_config_contains(bashrc_path, "export AWS_PROFILE=softmax"), (
            f"AWS_PROFILE export should be added to {bashrc_path} for softmax profile with ZDOTDIR"
        )

        # Check that existing content is preserved and export is added
        zshrc_content = zshrc_path.read_text()
        print(zshrc_content)
        assert "export AWS_PROFILE=softmax" in zshrc_content, "AWS_PROFILE export should be added to .zshrc"

        bashrc_content = bashrc_path.read_text()
        print(bashrc_content)
        assert "export AWS_PROFILE=softmax" in bashrc_content, "AWS_PROFILE export should be added to .bashrc"

        export_count = zshrc_content.count("export AWS_PROFILE=softmax")
        print(export_count)
        assert export_count == 1, f"Should have exactly one AWS_PROFILE export, found {export_count}"


@pytest.mark.setup
@pytest.mark.profile("external")
class TestAWSProfileExternal(AWSAssertionsMixin, BaseMettaSetupTest):
    """Ensure the external profile does not touch AWS config or shell exports."""

    def test_external_install_writes_no_aws_state(self):
        """Install with external profile and assert no AWS artifacts were written."""
        # Create external profile config
        self._create_test_config(UserType.EXTERNAL)

        # Run install (bypass base class mocking)
        result = self._run_metta_command(["install", "aws"])
        assert result.returncode == 0, f"Install failed: {result.stderr}"

        # Assert no AWS config directory/file was created
        assert not self._check_aws_config_exists(), "External profile should not create ~/.aws/config"

        # Assert no AWS profiles are present
        assert not self._check_aws_profile_config("softmax"), "External profile should not add softmax profile"
        assert not self._check_aws_profile_config("softmax-admin"), (
            "External profile should not add softmax-admin profile"
        )

        # Assert no AWS_PROFILE export added to shell configs
        zshrc_path = self._get_zshrc_path()
        bashrc_path = self._get_bashrc_path()
        assert not self._check_shell_config_contains(zshrc_path, "export AWS_PROFILE=softmax"), (
            f"External profile should not add AWS_PROFILE to {zshrc_path}"
        )
        assert not self._check_shell_config_contains(bashrc_path, "export AWS_PROFILE=softmax"), (
            f"External profile should not add AWS_PROFILE to {bashrc_path}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
