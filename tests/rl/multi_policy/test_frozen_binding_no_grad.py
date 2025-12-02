import torch
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.binding_controller import BindingControllerPolicy


class _TrainablePolicy(Policy):
    def __init__(self, policy_env_info):
        super().__init__(policy_env_info)
        self.linear = torch.nn.Linear(1, 1)

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64))

    def forward(self, td, action=None):
        td.set("actions", torch.zeros(td.batch_size.numel(), dtype=torch.int64))
        return td

    def initialize_to_environment(self, policy_env_info, device):
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_memory(self):
        return None


def test_frozen_binding_requires_grad_false():
    env_info = type("EnvInfo", (), {"num_agents": 1})
    trainable = _TrainablePolicy(env_info)
    # Manually freeze
    for p in trainable.parameters():
        p.requires_grad = False

    controller = BindingControllerPolicy(
        binding_lookup={"frozen": 0},
        bindings=[],
        binding_policies={0: trainable},
        policy_env_info=env_info,
        device="cpu",
        agent_binding_map=torch.tensor([0]),
    )

    loss = controller.linear.weight.sum()
    loss.backward()
    assert trainable.linear.weight.grad is None
