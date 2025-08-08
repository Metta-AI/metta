#!/usr/bin/env python3
"""
Minimal, real-install tests for `./install.sh`, gated by pytest profile markers.
"""

import pytest

from tests.setup.test_base import BaseMettaSetupTest


@pytest.mark.profile("softmax")
class TestInstallShSoftmax(BaseMettaSetupTest):
    def test_install_sh_softmax_profile(self):
        cfg = self._run_metta_command(["configure", "--profile=softmax"])
        assert cfg.returncode == 0, f"configure failed for softmax: {cfg.stderr}"

        inst = self._run_metta_command(["install", "--force"])
        assert inst.returncode == 0, f"install failed for softmax: {inst.stderr}"


@pytest.mark.profile("external")
class TestInstallShExternal(BaseMettaSetupTest):
    def test_install_sh_external_profile(self):
        cfg = self._run_metta_command(["configure", "--profile=external"])
        assert cfg.returncode == 0, f"configure failed for external: {cfg.stderr}"

        inst = self._run_metta_command(["install", "--force"])  # no explicit aws component here
        assert inst.returncode == 0, f"install failed for external: {inst.stderr}"
