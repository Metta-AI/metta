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


def _build_test_ppo() -> PPO:
    policy = _SimplePolicy(offset=0)
    env = SimpleNamespace(single_action_space=spaces.Discrete(4))

    trainer_cfg = SimpleNamespace(dual_policy=SimpleNamespace(enabled=True, training_agents_pct=0.5))
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

    npc_policy = _SimplePolicy(offset=5)
    num_agents = 4
    env_obs = torch.zeros(num_agents, 1)

    td = TensorDict({"env_obs": env_obs.clone()}, batch_size=[num_agents])
    npc_input_td = td.select(*trainer_ppo.policy_experience_spec.keys(include_nested=True), strict=False).clone()

    trainer_ppo.policy.forward(td)
    npc_policy.forward(npc_input_td)

    student_mask = torch.tensor([False, False, True, True])
    npc_mask = ~student_mask

    ctx = SimpleNamespace(
        dual_policy_enabled=True,
        npc_policy=npc_policy,
        npc_mask_per_env=npc_mask,
        student_mask_per_env=student_mask,
    )

    student_actions_before = td["actions"].clone()
    npc_actions_before = npc_input_td["actions"].clone()

    trainer_ppo._inject_dual_policy_outputs(td, npc_input_td, ctx)

    npc_indices = torch.nonzero(npc_mask, as_tuple=False).flatten()
    student_indices = torch.nonzero(student_mask, as_tuple=False).flatten()

    assert torch.all(td["actions"][npc_indices] == npc_actions_before[npc_indices])
    assert torch.all(td["actions"][student_indices] == student_actions_before[student_indices])

    mask = td["is_student_agent"].reshape(-1)
    assert torch.all(mask[npc_indices] == 0.0)
    assert torch.all(mask[student_indices] == 1.0)
