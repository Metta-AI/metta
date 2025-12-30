from __future__ import annotations

from typing import Sequence


class MettascopeAction:
    action_name: str | bytes
    agent_id: int


class MettascopeResponse:
    should_close: bool
    actions: Sequence[MettascopeAction] | None


def init(data_dir: str, replay: str) -> MettascopeResponse: ...


def render(step: int, replay_step: str) -> MettascopeResponse: ...


class MettascopeError(Exception): ...
