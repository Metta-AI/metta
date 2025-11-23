from pathlib import Path

import torch

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.policy_artifact import load_policy_artifact
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.loader import save_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def test_policy_save_data_includes_architecture(tmp_path: Path) -> None:
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    arch = ViTDefaultConfig()
    policy = arch.make_policy(env_info)

    destination = tmp_path / "policy.mpt"
    save_policy(destination, policy, arch_hint=arch)

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
    save_policy(destination, source_policy, arch_hint=arch)

    target_policy = arch.make_policy(env_info)
    artifact = load_policy_artifact(destination)
    target_policy.load_state_dict(artifact.state_dict)

    for p_ref, p_loaded in zip(
        source_policy.parameters(),
        target_policy.parameters(),
        strict=True,
    ):
        assert torch.equal(p_ref, p_loaded)
