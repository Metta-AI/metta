"""Core types and controller for the scripted agent phase machine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

Guard = Callable[[object, "Context"], bool]
Hook = Callable[[object, "Context"], None]


class GamePhase(Enum):
    GATHER_GERMANIUM = auto()
    GATHER_SILICON = auto()
    GATHER_CARBON = auto()
    GATHER_OXYGEN = auto()
    ASSEMBLE_HEART = auto()
    DEPOSIT_HEART = auto()
    RECHARGE = auto()
    EXPLORE = auto()
    UNCLIP_STATION = auto()
    CRAFT_DECODER = auto()


@dataclass(frozen=True)
class Transition:
    src: GamePhase
    dst: GamePhase
    guard: Guard
    priority: int = 0
    min_dwell_steps: int = 0
    on_enter: Optional[Hook] = None
    on_exit: Optional[Hook] = None


@dataclass
class Context:
    obs: Any
    env: Any
    step: int
    policy_impl: Any | None = None


class PhaseController:
    def __init__(self, initial: GamePhase, transitions: List[Transition]):
        self.initial = initial
        self._transitions = transitions

    def maybe_transition(self, state: object, ctx: Context, logger_: logging.Logger) -> GamePhase:
        current_phase = state.current_phase

        runtime = state.phase_runtime.setdefault(current_phase.name, {"entered_at_step": ctx.step, "visits": 0})
        candidates = [t for t in self._transitions if t.src == current_phase]

        logger_.debug("[PhaseController] Current phase: %s, candidates: %d", current_phase.name, len(candidates))
        for t in candidates:
            logger_.debug(
                "  %s -> %s (priority=%s, min_dwell=%s)", t.src.name, t.dst.name, t.priority, t.min_dwell_steps
            )

        entered_at = runtime["entered_at_step"]
        enabled: List[Transition] = []
        for t in candidates:
            dwell = ctx.step - entered_at
            if dwell < t.min_dwell_steps:
                logger_.debug("  Skipping %s->%s due to hysteresis", t.src.name, t.dst.name)
                continue
            try:
                guard_result = t.guard(state, ctx)
                logger_.debug("  Guard %s->%s: %s", t.src.name, t.dst.name, guard_result)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger_.warning("[Phase] Guard error %s->%s: %s", t.src.name, t.dst.name, exc)
                guard_result = False
            if guard_result:
                enabled.append(t)

        if not enabled:
            logger_.debug("[PhaseController] No enabled transitions, staying in %s", current_phase.name)
            return current_phase

        enabled.sort(key=lambda tr: (tr.priority, -tr.dst.value), reverse=True)
        chosen = enabled[0]

        if chosen.dst == current_phase:
            return current_phase

        if chosen.on_exit:
            try:
                chosen.on_exit(state, ctx)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger_.warning("[Phase] on_exit %s: %s", chosen.src, exc)

        next_runtime = state.phase_runtime.setdefault(chosen.dst.name, {"entered_at_step": ctx.step, "visits": 0})
        next_runtime["visits"] += 1
        next_runtime["entered_at_step"] = ctx.step

        if chosen.on_enter:
            try:
                chosen.on_enter(state, ctx)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger_.warning("[Phase] on_enter %s: %s", chosen.dst, exc)

        logger_.info("[Phase] %s → %s (priority=%s)", chosen.src.name, chosen.dst.name, chosen.priority)
        return chosen.dst


__all__ = [
    "Guard",
    "Hook",
    "Transition",
    "Context",
    "GamePhase",
    "PhaseController",
]
