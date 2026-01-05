from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation


@dataclass
class ChaosMonkeyConfig:
    fail_step: int = 10
    fail_probability: float = 1.0  # Probability of failure when the threshold is reached
    seed: Optional[int] = None


class _ChaosMonkeyAgent(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, cfg: ChaosMonkeyConfig, agent_id: int):
        super().__init__(policy_env_info)
        self._cfg = cfg
        self._step = 0
        self._failed = False
        self._noop = policy_env_info.actions.noop.Noop()
        seed = cfg.seed
        self._rng = random.Random(seed + agent_id) if seed is not None else random.Random()

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        self._step = 0
        self._failed = False

    def step(self, obs: AgentObservation) -> Action:
        if self._failed:
            return self._noop

        if self._step >= self._cfg.fail_step and self._rng.random() < self._cfg.fail_probability:
            self._failed = True
            raise RuntimeError(f"Chaos monkey triggered at step {self._step}")

        self._step += 1
        return self._noop


class ChaosMonkeyPolicy(MultiAgentPolicy):
    """A scripted policy that intentionally fails mid-episode to test robustness."""

    short_names = ["chaos-monkey"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        fail_step: int = 10,
        fail_probability: float = 1.0,
        seed: Optional[int] = None,
        device: str = "cpu",
        **_: object,
    ):
        super().__init__(policy_env_info, device=device)
        self._cfg = ChaosMonkeyConfig(fail_step=fail_step, fail_probability=fail_probability, seed=seed)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return _ChaosMonkeyAgent(self.policy_env_info, self._cfg, agent_id)
