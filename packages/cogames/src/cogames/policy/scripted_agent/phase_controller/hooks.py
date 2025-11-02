"""Hook helpers invoked on phase transitions."""

from __future__ import annotations

from typing import Callable

from .controller import GamePhase


def enter_deposit(state, ctx):  # noqa: ARG001
    state.just_deposited = False


def get_last_gathering_phase(state) -> GamePhase:
    gathering_phases = {
        GamePhase.GATHER_CARBON,
        GamePhase.GATHER_OXYGEN,
        GamePhase.GATHER_GERMANIUM,
        GamePhase.GATHER_SILICON,
    }
    for phase in reversed(state.phase_history):
        if phase in gathering_phases:
            return phase
    return GamePhase.GATHER_GERMANIUM


def enter_recharge(state, ctx):  # noqa: ARG001
    state.recharge_last_energy = getattr(state, "energy", state.recharge_last_energy)
    state.recharge_ticks_without_gain = 0
    state.recharge_total_gained = 0


def enter_gather_any(state, ctx):  # noqa: ARG001
    return None


def enter_craft_decoder(state, ctx):  # noqa: ARG001
    return None


def enter_unclip_station(state, ctx):  # noqa: ARG001
    state.current_glyph = "default"


def exit_unclip_station(state, ctx):  # noqa: ARG001
    state.blocked_by_clipped_extractor = None


def enter_explore(goal: str) -> Callable[[object, object], None]:
    def _hook(state, ctx):  # noqa: ARG001
        state.explore_goal = goal

    return _hook


def exit_explore(state, ctx):  # noqa: ARG001
    state.explore_goal = None


__all__ = [
    "enter_deposit",
    "get_last_gathering_phase",
    "enter_recharge",
    "enter_gather_any",
    "enter_craft_decoder",
    "enter_unclip_station",
    "exit_unclip_station",
    "enter_explore",
    "exit_explore",
]
