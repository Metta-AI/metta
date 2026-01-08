import pytest

from metta.setup.components.system_packages.bootstrap import version_ge


@pytest.mark.setup
class TestVersionGe:
    def test_equal_versions(self):
        assert version_ge("1.2.3", "1.2.3") is True

    def test_greater_major(self):
        assert version_ge("2.0.0", "1.9.9") is True

    def test_greater_minor(self):
        assert version_ge("1.3.0", "1.2.9") is True

    def test_greater_patch(self):
        assert version_ge("1.2.4", "1.2.3") is True

    def test_less_major(self):
        assert version_ge("1.0.0", "2.0.0") is False

    def test_less_minor(self):
        assert version_ge("1.2.0", "1.3.0") is False

    def test_less_patch(self):
        assert version_ge("1.2.2", "1.2.3") is False

    def test_none_current(self):
        assert version_ge(None, "1.0.0") is False

    def test_empty_current(self):
        assert version_ge("", "1.0.0") is False

    def test_different_lengths_current_longer(self):
        assert version_ge("1.2.3.4", "1.2.3") is True

    def test_different_lengths_required_longer(self):
        assert version_ge("1.2", "1.2.0") is True

    def test_different_lengths_less(self):
        assert version_ge("1.2", "1.2.1") is False

    def test_real_versions_nim(self):
        assert version_ge("2.2.6", "2.2.6") is True
        assert version_ge("2.2.7", "2.2.6") is True
        assert version_ge("2.2.5", "2.2.6") is False

    def test_real_versions_nimby(self):
        assert version_ge("0.1.13", "0.1.13") is True
        assert version_ge("0.1.14", "0.1.13") is True
        assert version_ge("0.1.12", "0.1.13") is False

    def test_real_versions_bazel(self):
        assert version_ge("7.0.0", "7.0.0") is True
        assert version_ge("7.1.0", "7.0.0") is True
        assert version_ge("6.5.0", "7.0.0") is False
