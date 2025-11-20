import pytest

from cogames.cli import policy as policy_cli


def test_parse_policy_spec_accepts_mpt(tmp_path):
    checkpoint = tmp_path / "model.mpt"
    checkpoint.write_text("placeholder")

    spec = policy_cli._parse_policy_spec(f"stateless:{checkpoint}")

    assert spec.data_path == str(checkpoint)
    assert spec.proportion == 1.0


def test_parse_policy_spec_fraction_with_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("placeholder")

    spec = policy_cli._parse_policy_spec(f"stateless:{checkpoint}:2.5")

    assert spec.data_path == str(checkpoint)
    assert spec.proportion == 2.5


def test_parse_policy_spec_fraction_without_checkpoint():
    spec = policy_cli._parse_policy_spec("stateless::3.25")

    assert spec.data_path is None
    assert spec.proportion == 3.25


def test_parse_policy_spec_rejects_non_positive_fraction():
    with pytest.raises(ValueError):
        policy_cli._parse_policy_spec("stateless::0")

    with pytest.raises(ValueError):
        policy_cli._parse_policy_spec("stateless:: -1")
