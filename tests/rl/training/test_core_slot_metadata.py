import torch
from tensordict import TensorDict

from metta.rl.training.core import CoreTrainingLoop


class _DummyEnvInfo:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents


class _DummyEnv:
    def __init__(self, num_agents: int):
        self.policy_env_info = _DummyEnvInfo(num_agents)


class _DummyContext:
    def __init__(self, num_agents: int):
        self.env = _DummyEnv(num_agents)
        self.slot_id_per_agent = torch.tensor([3])
        self.loss_profile_id_per_agent = torch.tensor([2])
        self.trainable_agent_mask = torch.tensor([True])


def test_injects_metadata_for_single_agent_single_env():
    td = TensorDict({}, batch_size=[1])
    context = _DummyContext(num_agents=1)

    loop = object.__new__(CoreTrainingLoop)
    loop.context = context
    loop._metadata_cache = {}

    loop._inject_slot_metadata(td, slice(0, 1))

    assert torch.equal(td["slot_id"], torch.tensor([3]))
    assert torch.equal(td["loss_profile_id"], torch.tensor([2]))
    assert torch.equal(td["is_trainable_agent"], torch.tensor([True]))
