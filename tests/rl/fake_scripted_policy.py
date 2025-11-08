"""Minimal scripted policy used for adapter tests."""

from __future__ import annotations

from typing import Any


class _FakeAgentPolicy:
    def __init__(self, agent_id: int, multiplier: int) -> None:
        self._agent_id = agent_id
        self._multiplier = multiplier
        self._steps = 0

    def step(self, obs: Any) -> int:
        del obs  # Observations unused in this fake implementation
        self._steps += 1
        return self._agent_id + self._steps * self._multiplier

    def reset(self) -> None:
        self._steps = 0


class FakeScriptedAgentPolicy:
    """Stand-in for a Cogames scripted policy."""

    def __init__(self, env: Any, multiplier: int = 1) -> None:
        self._env = env
        self._multiplier = multiplier

    def agent_policy(self, agent_id: int) -> _FakeAgentPolicy:
        return _FakeAgentPolicy(agent_id, self._multiplier)
