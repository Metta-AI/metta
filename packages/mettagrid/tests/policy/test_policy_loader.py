import zipfile

import torch
from safetensors.torch import save_file

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class _ToyPolicy(TrainablePolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self.module = torch.nn.Linear(4, 2, bias=True)

    def network(self):
        return self.module

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        raise NotImplementedError


def test_policy_loads_mpt_checkpoint(tmp_path):
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    policy = _ToyPolicy(policy_env_info)

    tensors = {
        "_sequential_network.weight": torch.randn_like(policy.module.weight),
        "_sequential_network.bias": torch.randn_like(policy.module.bias),
    }

    weights_path = tmp_path / "weights.safetensors"
    save_file(tensors, str(weights_path))

    archive_path = tmp_path / "checkpoint.mpt"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(weights_path, arcname="weights.safetensors")

    policy.load_policy_data(str(archive_path))

    assert torch.equal(policy.module.weight, tensors["_sequential_network.weight"])
    assert torch.equal(policy.module.bias, tensors["_sequential_network.bias"])
