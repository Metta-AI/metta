import gymnasium as gym
import torch
from tensordict import TensorDict

from metta.agent.policies.npc import SimpleNPCPolicy
from metta.rl.slot import SlotControllerPolicy


class _StubActions:
    def __init__(self, names: list[str]) -> None:
        self._actions = [type("Action", (), {"name": name}) for name in names]

    def actions(self):
        return self._actions


class _StubEnv:
    def __init__(self, action_names: list[str], num_agents: int = 1) -> None:
        self.actions = _StubActions(action_names)
        self.action_space = gym.spaces.Discrete(len(action_names))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 1), dtype=float)
        self.num_agents = num_agents
        self.tags: list[str] = []
        self.obs_features: list[str] = []
        self.obs_width = 1
        self.obs_height = 1
        self.assembler_protocols = []
        self.tag_id_to_name = {}
        self.action_names = action_names


def test_npc_policy_defaults_to_noop() -> None:
    env = _StubEnv(["noop", "move", "attack"])
    policy = SimpleNPCPolicy(env)
    policy.initialize_to_environment(env, torch.device("cpu"))

    td = TensorDict({}, batch_size=[4])
    out = policy.forward(td)
    assert torch.equal(out["actions"], torch.zeros(4, dtype=torch.int64))
    assert out["act_log_prob"].shape == torch.Size([4])
    assert out["entropy"].shape == torch.Size([4])
    assert out["values"].shape == torch.Size([4])


def test_npc_policy_respects_provided_actions() -> None:
    env = _StubEnv(["noop", "move"])
    policy = SimpleNPCPolicy(env)
    policy.initialize_to_environment(env, torch.device("cpu"))

    td = TensorDict({}, batch_size=[2])
    provided_actions = torch.tensor([1, 0], dtype=torch.int64)
    out = policy.forward(td, action=provided_actions)
    assert torch.equal(out["actions"], provided_actions)


def test_npc_policy_runs_via_slot_controller() -> None:
    env = _StubEnv(["noop", "move"], num_agents=2)
    policy = SimpleNPCPolicy(env)
    policy.initialize_to_environment(env, torch.device("cpu"))

    agent_slot_map = torch.tensor([0, 0], dtype=torch.long)
    controller = SlotControllerPolicy(
        slot_lookup={"npc": 0},
        slots=[],
        slot_policies={0: policy},
        policy_env_info=env,
        agent_slot_map=agent_slot_map,
    ).to("cpu")

    td = TensorDict({"env_obs": torch.zeros(4, 1)}, batch_size=[4])
    out = controller.forward(td)
    assert torch.equal(out["actions"], torch.zeros(4, dtype=torch.int64))
