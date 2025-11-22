from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.policy_artifact import load_policy_artifact
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def test_policy_save_data_includes_architecture(tmp_path) -> None:
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
