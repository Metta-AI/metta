from pathlib import Path

from cogames.cli import policy


def test_parse_policy_accepts_colon_in_path(tmp_path):
    data_path = tmp_path / "checkpoint:v23.mpt"
    data_path.write_text("")

    spec = policy._parse_policy_spec(
        f"metta.agent.policies.vit.ViTDefaultPolicy:{data_path}",
    )

    assert spec.data_path == str(Path(data_path))
    assert spec.proportion == 1.0


def test_parse_policy_with_colon_path_and_fraction(tmp_path):
    data_path = tmp_path / "checkpoint:v23.mpt"
    data_path.write_text("")

    spec = policy._parse_policy_spec(
        f"metta.agent.policies.vit.ViTDefaultPolicy:{data_path}:0.25",
    )

    assert spec.data_path == str(Path(data_path))
    assert spec.proportion == 0.25
