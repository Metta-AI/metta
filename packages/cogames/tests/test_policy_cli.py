from pathlib import Path

import torch

from cogames.cli import policy
from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.policy_artifact import save_policy_artifact_safetensors


def _write_dummy_mpt(path: Path) -> None:
    tensors = {"_sequential_network.dummy": torch.zeros(1)}
    save_policy_artifact_safetensors(
        path,
        policy_architecture=ViTDefaultConfig(),
        state_dict=tensors,
    )


def test_parse_policy_accepts_colon_in_path(tmp_path):
    data_path = tmp_path / "checkpoint:v23.pt"
    data_path.write_text("")

    spec = policy._parse_policy_spec(
        f"metta.agent.policies.vit.ViTDefaultPolicy:{data_path}",
    )

    assert spec.data_path == str(Path(data_path))
    assert spec.proportion == 1.0


def test_parse_policy_with_colon_path_and_fraction(tmp_path):
    data_path = tmp_path / "checkpoint:v23.pt"
    data_path.write_text("")

    spec = policy._parse_policy_spec(
        f"metta.agent.policies.vit.ViTDefaultPolicy:{data_path}:0.25",
    )

    assert spec.data_path == str(Path(data_path))
    assert spec.proportion == 0.25


def test_parse_policy_allows_checkpoint_only(tmp_path):
    data_path = tmp_path / "checkpoint.mpt"
    _write_dummy_mpt(data_path)

    spec = policy._parse_policy_spec(str(data_path))

    assert spec.class_path == "metta.agent.policy_auto_builder.PolicyAutoBuilder"
    assert spec.data_path == str(Path(data_path))
    assert spec.proportion == 1.0
    assert "config" in spec.init_kwargs


def test_parse_policy_with_colon_path_and_no_class(tmp_path):
    data_path = tmp_path / "checkpoint:v23.mpt"
    _write_dummy_mpt(data_path)

    spec = policy._parse_policy_spec(str(data_path))

    assert spec.class_path == "metta.agent.policy_auto_builder.PolicyAutoBuilder"
    assert spec.data_path == str(Path(data_path))
    assert "config" in spec.init_kwargs
