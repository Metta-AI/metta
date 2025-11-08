from types import SimpleNamespace

import torch
from gymnasium import spaces
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.ppo import PPO, PPOConfig


class _SimplePolicy(Policy):
    """Deterministic policy that emits easily distinguishable tensors."""

    def __init__(self, offset: int) -> None:
        super().__init__()
        self._offset = offset
        self._device = torch.device("cpu")

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        batch = td.batch_size.numel()
        device = td.device
        td.set("actions", torch.arange(batch, device=device, dtype=torch.int64) + self._offset)
        td.set("act_log_prob", torch.full((batch,), float(self._offset + 10), device=device))
        td.set("entropy", torch.full((batch,), float(self._offset + 20), device=device))
        td.set("values", torch.full((batch,), float(self._offset + 30), device=device))
        td.set("full_log_probs", torch.full((batch, 1), float(self._offset + 40), device=device))
        return td

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        return Composite(
            env_obs=UnboundedContinuous(shape=torch.Size([1]), dtype=torch.float32),
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            act_log_prob=scalar_f32,
            entropy=scalar_f32,
            values=scalar_f32,
            full_log_probs=UnboundedContinuous(shape=torch.Size([1]), dtype=torch.float32),
        )

    def initialize_to_environment(self, game_rules, device: torch.device) -> None:  # noqa: D401
        self._device = device

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device

    def reset_memory(self) -> None:  # noqa: D401
        return None


class _NorthCheckingPolicy(Policy):
    """NPC policy that always selects the 'north' action and verifies metadata."""

    def __init__(self, north_action: int = 0) -> None:
        super().__init__()
        self._north_action = north_action
        self._device = torch.device("cpu")
        self._saw_metadata = False
        self._spec = _SimplePolicy(offset=0).get_agent_experience_spec()

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        required = ("bptt", "batch", "training_env_ids")
        missing = [key for key in required if key not in td.keys(include_nested=True)]
        if missing:
            raise KeyError(f"Missing rollout metadata for NPC policy: {missing}")

        self._saw_metadata = True

        batch = td.batch_size.numel()
        device = td.device
        td.set("actions", torch.full((batch,), self._north_action, dtype=torch.int64, device=device))
        td.set("act_log_prob", torch.zeros(batch, device=device))
        td.set("entropy", torch.zeros(batch, device=device))
        td.set("values", torch.zeros(batch, device=device))
        td.set("full_log_probs", torch.zeros(batch, 1, device=device))
        return td

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return self._spec

    def initialize_to_environment(self, game_rules, device: torch.device) -> None:  # noqa: D401
        self._device = device

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device

    def reset_memory(self) -> None:  # noqa: D401
        return None


def _build_test_ppo() -> PPO:
    policy = _SimplePolicy(offset=0)
    env = SimpleNamespace(single_action_space=spaces.Discrete(4))

    trainer_cfg = SimpleNamespace(
        dual_policy=SimpleNamespace(
            enabled=True,
            training_agents_pct=0.5,
            npc_policy_uri="mock://test",
        )
    )
    loss_cfg = PPOConfig()

    # Minimal trainer config namespace to satisfy attributes accessed by Loss
    trainer_cfg.total_timesteps = 1
    trainer_cfg.batch_size = 1
    trainer_cfg.minibatch_size = 1
    trainer_cfg.bptt_horizon = 1
    trainer_cfg.vtrace = loss_cfg.vtrace
    trainer_cfg.prioritized_experience_replay = loss_cfg.prioritized_experience_replay
    trainer_cfg.ppo = loss_cfg

    return PPO(policy, trainer_cfg, env, torch.device("cpu"), "ppo", loss_cfg)


def test_dual_policy_selection_is_non_strict() -> None:
    ppo = _build_test_ppo()
    td = TensorDict({"env_obs": torch.zeros(4, 1)}, batch_size=[4])

    # Should not raise even though rollout outputs are absent pre-forward
    npc_td = td.select(*ppo.policy_experience_spec.keys(include_nested=True), strict=False)

    assert "env_obs" in npc_td.keys()
    assert "actions" not in npc_td.keys()


def test_dual_policy_inject_overwrites_npc_agents() -> None:
    trainer_ppo = _build_test_ppo()

    npc_policy = _NorthCheckingPolicy(north_action=7)
    num_agents = 4
    env_obs = torch.zeros(num_agents, 1)

    td = TensorDict({"env_obs": env_obs.clone()}, batch_size=[num_agents])

    td.set("bptt", torch.ones(num_agents, dtype=torch.long))
    td.set("batch", torch.full((num_agents,), 2, dtype=torch.long))
    td.set("training_env_ids", torch.arange(num_agents, dtype=torch.long).unsqueeze(1))

    student_mask = torch.tensor([False, False, True, True])
    npc_mask = ~student_mask

    ctx = SimpleNamespace(
        dual_policy_enabled=True,
        npc_policy=npc_policy,
        npc_mask_per_env=npc_mask,
        student_mask_per_env=student_mask,
        training_env_id=slice(0, num_agents),
    )

    class _StubReplay:
        def __init__(self) -> None:
            self.calls: list[tuple[TensorDict, slice]] = []

        def store(self, data_td: TensorDict, env_id: slice) -> None:  # noqa: D401
            self.calls.append((data_td.clone(), env_id))

    trainer_ppo.replay = _StubReplay()

    trainer_ppo.run_rollout(td, ctx)

    npc_indices = torch.nonzero(npc_mask, as_tuple=False).flatten()
    student_indices = torch.nonzero(student_mask, as_tuple=False).flatten()

    expected_actions = torch.arange(num_agents, dtype=torch.int64)
    expected_actions[npc_indices] = npc_policy._north_action
    assert torch.all(td["actions"] == expected_actions)
    assert torch.all(td["actions"][npc_indices] == npc_policy._north_action)
    assert torch.all(td["is_student_agent"][npc_indices] == 0.0)
    assert torch.all(td["is_student_agent"][student_indices] == 1.0)

    assert npc_policy._saw_metadata

    stored_td, stored_env = trainer_ppo.replay.calls[0]
    assert stored_env == slice(0, num_agents)
    assert torch.all(stored_td["is_student_agent"][npc_indices] == 0.0)
