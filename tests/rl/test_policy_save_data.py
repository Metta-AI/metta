from pathlib import Path
from typing import Mapping

import torch

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.policy_artifact import load_policy_artifact
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def test_policy_save_data_includes_architecture(tmp_path: Path) -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    arch = ViTDefaultConfig()
    policy = arch.make_policy(env_info)

    destination = tmp_path / "policy.mpt"
    policy.save_policy_data(str(destination))

    artifact = load_policy_artifact(destination)
    assert artifact.policy_architecture is not None
    assert artifact.policy_architecture.class_path == arch.class_path
    assert artifact.state_dict is not None


def test_policy_load_data_reads_artifacts(tmp_path: Path) -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    arch = ViTDefaultConfig()

    source_policy = arch.make_policy(env_info)
    destination = tmp_path / "policy.mpt"
    source_policy.save_policy_data(str(destination))

    target_policy = arch.make_policy(env_info)
    target_policy.load_policy_data(str(destination))

    for p_ref, p_loaded in zip(
        source_policy.parameters(),
        target_policy.parameters(),
        strict=True,
    ):
        assert torch.equal(p_ref, p_loaded)


def test_policy_save_data_respects_pt_extension(tmp_path: Path) -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    arch = ViTDefaultConfig()
    policy = arch.make_policy(env_info)

    destination = tmp_path / "policy.pt"
    policy.save_policy_data(str(destination))

    state_dict = torch.load(destination, map_location="cpu")
    assert isinstance(state_dict, Mapping)
    assert state_dict  # ensure tensors were serialized


def test_policy_load_data_reads_artifacts_with_pt_extension(tmp_path: Path) -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    arch = ViTDefaultConfig()

    source_policy = arch.make_policy(env_info)
    artifact_path = tmp_path / "policy.mpt"
    source_policy.save_policy_data(str(artifact_path))

    pt_path = tmp_path / "policy.pt"
    source_policy.save_policy_data(str(pt_path))

    target_policy = arch.make_policy(env_info)
    target_policy.load_policy_data(str(pt_path))

    for p_ref, p_loaded in zip(
        source_policy.parameters(),
        target_policy.parameters(),
        strict=True,
    ):
        assert torch.equal(p_ref, p_loaded)
