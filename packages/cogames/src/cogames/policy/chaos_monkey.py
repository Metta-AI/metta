from __future__ import annotations

import random
from typing import Optional

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation


class _ChaosMonkeyAgent(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, fail_step: int, fail_probability: float):
        super().__init__(policy_env_info)
        self._fail_step = fail_step
        self._fail_probability = fail_probability
        self._step = 0
        self._failed = False
        self._noop = policy_env_info.actions.noop.Noop()

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        self._step = 0
        self._failed = False

    def step(self, obs: AgentObservation) -> Action:
        if self._failed:
            return self._noop

        if self._step >= self._fail_step and random.random() < self._fail_probability:
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
        device: str = "cpu",
        **_: object,
    ):
        super().__init__(policy_env_info, device=device)
        self._fail_step = fail_step
        self._fail_probability = fail_probability

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return _ChaosMonkeyAgent(self.policy_env_info, self._fail_step, self._fail_probability)
