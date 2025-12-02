import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.slot_controller import SlotControllerPolicy


class _StubPolicy(Policy):
    def __init__(self, policy_env_info, action_val: int) -> None:
        super().__init__(policy_env_info)
        self._action_val = action_val
        self._device = torch.device("cpu")

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        act_dtype = torch.int64
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        return Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            act_log_prob=scalar_f32,
            entropy=scalar_f32,
            values=scalar_f32,
        )

    def forward(self, td: TensorDict, action=None):  # noqa: D401
        batch = td.batch_size.numel()
        td.set("actions", torch.full((batch,), self._action_val, dtype=torch.int64, device=td.device))
        td.set("act_log_prob", torch.zeros(batch, device=td.device))
        td.set("entropy", torch.zeros(batch, device=td.device))
        td.set("values", torch.zeros(batch, device=td.device))
        return td

    def initialize_to_environment(self, policy_env_info, device: torch.device):  # noqa: D401
        self._device = device

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device

    def reset_memory(self) -> None:  # noqa: D401
        return None


def test_binding_controller_merges_only_action_keys() -> None:
    # Two bindings, alternating agents
    slot_lookup = {"a": 0, "b": 1}
    env_info = type("EnvInfo", (), {"num_agents": 2})  # minimal stub
    policies = {
        0: _StubPolicy(env_info, action_val=1),
        1: _StubPolicy(env_info, action_val=9),
    }
    agent_slot_map = torch.tensor([0, 1], dtype=torch.long)
    controller = SlotControllerPolicy(
        slot_lookup=slot_lookup,
        slots=[],
        slot_policies=policies,
        policy_env_info=env_info,
        device="cpu",
        agent_slot_map=agent_slot_map,
    )

    td = TensorDict({"env_obs": torch.zeros(4, 1)}, batch_size=[4])
    td.set("slot_id", torch.tensor([0, 1, 0, 1]))

    out = controller.forward(td.clone())
    assert torch.equal(out["actions"], torch.tensor([1, 9, 1, 9]))


def test_slot_controller_generates_slot_ids_from_agent_map() -> None:
    slot_lookup = {"a": 0, "b": 1}
    env_info = type("EnvInfo", (), {"num_agents": 2})
    policies = {
        0: _StubPolicy(env_info, action_val=2),
        1: _StubPolicy(env_info, action_val=5),
    }
    agent_slot_map = torch.tensor([1, 0], dtype=torch.long)
    controller = SlotControllerPolicy(
        slot_lookup=slot_lookup,
        slots=[],
        slot_policies=policies,
        policy_env_info=env_info,
        device="cpu",
        agent_slot_map=agent_slot_map,
    )

    td = TensorDict({"env_obs": torch.zeros(4, 1)}, batch_size=[4])
    out = controller.forward(td.clone())
    assert torch.equal(out["actions"], torch.tensor([5, 2, 5, 2]))
