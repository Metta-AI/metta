import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.trainer_config import SamplingConfig
from metta.rl.training.core import CoreTrainingLoop
from metta.rl.training.experience import Experience


class _StubAction:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubActions:
    def __init__(self, names: list[str]) -> None:
        self._actions = [_StubAction(n) for n in names]

    def actions(self):
        return self._actions


class _StubPolicyEnvInfo:
    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents
        self.actions = _StubActions(["noop"])
        self.action_space = type("AS", (), {"n": 1})
        self.observation_space = type("OS", (), {"shape": (1, 1)})
        self.obs_features = []
        self.tags = []
        self.obs_width = 1
        self.obs_height = 1
        self.assembler_protocols = []
        self.tag_id_to_name = {}


class _StubPolicy(torch.nn.Module):
    def __init__(self, policy_env_info) -> None:
        super().__init__()
        self._policy_env_info = policy_env_info
        self._param = torch.nn.Parameter(torch.zeros(1))

    def get_agent_experience_spec(self) -> Composite:
        return Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64))

    def forward(self, td, action=None):
        td.set("actions", torch.zeros(td.batch_size.numel(), dtype=torch.int64, device=td.device))
        return td

    def reset_memory(self):
        return None

    @property
    def device(self):
        return self._param.device

    def initialize_to_environment(self, policy_env_info, device: torch.device):
        return None


class _StubContext:
    def __init__(self, env_info: _StubPolicyEnvInfo, slot_ids: torch.Tensor, profile_ids: torch.Tensor):
        self.env = type("Env", (), {"policy_env_info": env_info})
        self.slot_id_per_agent = slot_ids
        self.loss_profile_id_per_agent = profile_ids
        self.trainable_agent_mask = torch.tensor([True] * slot_ids.numel())


def test_slot_metadata_injected_and_repeated_by_envs():
    env_info = _StubPolicyEnvInfo(num_agents=2)
    policy = _StubPolicy(env_info)
    losses: dict[str, torch.nn.Module] = {}
    device = torch.device("cpu")

    experience = Experience(
        total_agents=2,
        batch_size=4,
        bptt_horizon=2,
        minibatch_size=2,
        max_minibatch_size=2,
        experience_spec=policy.get_agent_experience_spec(),
        device=device,
        sampling_config=SamplingConfig(),
    )

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    context = _StubContext(env_info, slot_ids=torch.tensor([0, 1], dtype=torch.long), profile_ids=torch.tensor([5, 6]))
    loop = CoreTrainingLoop(
        policy=policy,
        experience=experience,
        losses=losses,
        optimizer=optimizer,
        device=device,
        context=context,
    )

    td = TensorDict({"env_obs": torch.zeros(4, 1)}, batch_size=[4])
    loop._inject_slot_metadata(td, slice(0, 2))

    assert torch.equal(td["slot_id"], torch.tensor([0, 1, 0, 1], dtype=torch.long))
    assert torch.equal(td["loss_profile_id"], torch.tensor([5, 6, 5, 6], dtype=torch.long))
    assert torch.equal(td["is_trainable_agent"], torch.tensor([1, 1, 1, 1], dtype=torch.bool))
