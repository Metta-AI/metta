"""
Phase Controller - Finite State Machine for Agent Phases

This module implements a declarative finite-state machine for managing agent phases,
replacing the scattered if/else logic with a clean, auditable transition system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

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
    priority: int = 0  # higher wins
    min_dwell_steps: int = 0  # hysteresis
    on_enter: Optional[Hook] = None
    on_exit: Optional[Hook] = None


@dataclass
class PhaseRuntime:
    entered_at_step: int = 0
    visits: int = 0


@dataclass
class Context:
    # read-only data you already have available each step
    obs: Any
    env: Any
    step: int
    # optional backpointer to policy implementation for guards needing richer context
    policy_impl: Any | None = None


class PhaseController:
    def __init__(self, initial: GamePhase, transitions: List[Transition]):
        self.phase = initial
        self._transitions = transitions
        self._rt: Dict[GamePhase, PhaseRuntime] = {p: PhaseRuntime() for p in GamePhase}
        self._rt[initial].entered_at_step = 0

    def current(self) -> GamePhase:
        return self.phase

    def maybe_transition(self, state: object, ctx: Context, logger) -> GamePhase:
        rt = self._rt[self.phase]
        # Filter transitions that start from the current phase
        candidates = [t for t in self._transitions if t.src == self.phase]

        # Debug logging
        logger.debug(f"[PhaseController] Current phase: {self.phase.name}, candidates: {len(candidates)}")
        for t in candidates:
            logger.debug(f"  {t.src.name} -> {t.dst.name} (priority={t.priority}, min_dwell={t.min_dwell_steps})")

        # Enforce hysteresis
        dwell_ok = (ctx.step - rt.entered_at_step) >= max(t.min_dwell_steps for t in candidates) if candidates else True
        logger.debug(
            f"[PhaseController] Dwell check: step={ctx.step}, entered_at={rt.entered_at_step}, dwell_ok={dwell_ok}"
        )

        enabled: List[Transition] = []
        for t in candidates:
            if (ctx.step - rt.entered_at_step) < t.min_dwell_steps:
                logger.debug(f"  Skipping {t.src.name}->{t.dst.name} due to hysteresis")
                continue
            try:
                guard_result = t.guard(state, ctx)
                logger.debug(f"  Guard {t.src.name}->{t.dst.name}: {guard_result}")
                if guard_result:
                    enabled.append(t)
            except Exception as e:
                logger.warning(f"[Phase] Guard error {t.src}->{t.dst}: {e}")

        logger.debug(f"[PhaseController] Enabled transitions: {len(enabled)}")
        for t in enabled:
            logger.debug(f"  ENABLED: {t.src.name} -> {t.dst.name} (priority={t.priority})")

        if not enabled:
            logger.debug(f"[PhaseController] No enabled transitions, staying in {self.phase.name}")
            return self.phase

        # Deterministic resolve: priority desc, then stable tie-break by enum order
        enabled.sort(key=lambda tr: (tr.priority, -tr.dst.value), reverse=True)
        chosen = enabled[0]
        if chosen.dst != self.phase:
            self._run_exit_hook(chosen, state, ctx, logger)
            self.phase = chosen.dst
            self._rt[self.phase].visits += 1
            self._rt[self.phase].entered_at_step = ctx.step
            self._run_enter_hook(chosen, state, ctx, logger)
            logger.info(f"[Phase] {chosen.src.name} → {chosen.dst.name} (priority={chosen.priority})")
        return self.phase

    def _run_enter_hook(self, t: Transition, state, ctx, logger):
        if t.on_enter:
            try:
                t.on_enter(state, ctx)
            except Exception as e:
                logger.warning(f"[Phase] on_enter {t.dst}: {e}")

    def _run_exit_hook(self, t: Transition, state, ctx, logger):
        if t.on_exit:
            try:
                t.on_exit(state, ctx)
            except Exception as e:
                logger.warning(f"[Phase] on_exit {t.src}: {e}")


# Guard functions
def has_all_materials(state, ctx):
    germ_needed = 5 if state.hearts_assembled == 0 else max(2, 5 - state.hearts_assembled)

    # Use fixed energy requirement
    energy_req = 20  # Fixed energy requirement

    return (
        state.germanium >= germ_needed
        and state.silicon >= 50
        and state.carbon >= 20
        and state.oxygen >= 20
        and state.energy >= energy_req
    )


def low_energy(state, ctx):
    # Use hyperparameter thresholds
    map_size = max(ctx.env.c_env.map_width, ctx.env.c_env.map_height)
    threshold = (
        ctx.policy_impl.hyperparams.recharge_start_small
        if map_size < 50
        else ctx.policy_impl.hyperparams.recharge_start_large
    )
    return state.energy < threshold


def recharged_enough(state, ctx):
    # Use hyperparameter thresholds
    map_size = max(ctx.env.c_env.map_width, ctx.env.c_env.map_height)
    threshold = (
        ctx.policy_impl.hyperparams.recharge_stop_small
        if map_size < 50
        else ctx.policy_impl.hyperparams.recharge_stop_large
    )
    return state.energy >= threshold


def carrying_heart(state, ctx):
    return state.heart > 0


def have_assembler_discovered(state, ctx):
    return getattr(state, "assembler_discovered", False)


def blocked_by_clipped(state, ctx):
    """Check if the agent is blocked by a clipped extractor."""
    return state.blocked_by_clipped_extractor is not None


def get_blocked_extractor_resource_type(state, ctx):
    """Get the resource type of the blocked extractor."""
    if state.blocked_by_clipped_extractor is None:
        return None

    if hasattr(state, "_policy_impl"):
        policy_impl = state._policy_impl
    else:
        policy_impl = getattr(ctx, "policy_impl", None)

    if policy_impl is None:
        return None

    extractor = policy_impl.extractor_memory.get_at_position(state.blocked_by_clipped_extractor)
    return extractor.resource_type if extractor else None


def need_craft_resource_for_blocked(state, ctx):
    """Check if we need to gather the craft resource for the blocked extractor."""
    resource_type = get_blocked_extractor_resource_type(state, ctx)
    if resource_type is None:
        return False

    # Get the unclip recipes
    unclip_recipes = (
        state.unclip_recipes
        if hasattr(state, "unclip_recipes") and state.unclip_recipes
        else {
            "oxygen": "carbon",
            "carbon": "oxygen",
            "germanium": "silicon",
            "silicon": "germanium",
        }
    )

    craft_resource = unclip_recipes.get(resource_type)
    if craft_resource is None:
        return False

    # Check if we have enough of the craft resource
    craft_amount = getattr(state, craft_resource, 0)
    return craft_amount < 1


def have_chest_discovered(state, ctx):
    return getattr(state, "chest_discovered", False)


def decoder_ready_for_unclipping(state, ctx):
    """Check if we have the correct unclip item for the blocked extractor.

    With reactive clipping, we only craft/unclip when blocked_by_clipped_extractor is set.
    This function checks if we have the appropriate unclip item for that specific extractor.
    """
    if state.blocked_by_clipped_extractor is None:
        return False

    # Get the resource type of the blocked extractor
    resource_type = get_blocked_extractor_resource_type(state, ctx)
    if resource_type is None:
        return False

    # Check if we have the appropriate unclip item
    # Mapping: oxygen → decoder, carbon → modulator, germanium → resonator, silicon → scrambler
    unclip_item_map = {
        "oxygen": "decoder",
        "carbon": "modulator",
        "germanium": "resonator",
        "silicon": "scrambler",
    }

    unclip_item_name = unclip_item_map.get(resource_type)
    if unclip_item_name is None:
        return False

    return getattr(state, unclip_item_name, 0) > 0


def progress_stalled(max_steps: int) -> Guard:
    def _g(state, ctx):
        # if no inventory delta since enter
        pinv = state.phase_entry_inventory or {}
        now = dict(g=state.germanium, si=state.silicon, c=state.carbon, o=state.oxygen)
        no_delta = all(
            now.get(k, 0) <= pinv.get(kmap, 0)
            for k, kmap in [("g", "germanium"), ("si", "silicon"), ("c", "carbon"), ("o", "oxygen")]
        )
        return no_delta and (ctx.step - state.phase_entry_step) >= max_steps

    return _g


# Hook functions
def enter_deposit(state, ctx):
    """Enter deposit heart phase."""
    state.just_deposited = False


def enter_recharge(state, ctx):
    """Enter recharge phase."""
    pass


def enter_gather_any(state, ctx):
    """Enter gathering phase."""
    pass


def enter_craft_decoder(state, ctx):
    """Enter craft decoder phase."""
    # Don't set glyph here - let the _execute_phase handle it properly
    pass


def enter_unclip_station(state, ctx):
    """Enter unclip station phase."""
    # Set default glyph for unclipping
    state.current_glyph = "default"


def exit_unclip_station(state, ctx):
    """Exit unclip station phase - clear the blocked flag."""
    state.blocked_by_clipped_extractor = None


# Explore entry/exit hooks - set the goal for why we're exploring
def enter_explore(goal: str):
    """Create an explore entry hook that sets the given goal."""

    def _hook(state, ctx):
        state.explore_goal = goal

    return _hook


def exit_explore(state, ctx):
    """Exit explore phase - clear the goal."""
    state.explore_goal = None


# Guards for detecting when to explore
def no_extractors_available(phase: GamePhase) -> Guard:
    """Check if no extractors are available for the given gathering phase."""
    resource_map = {
        GamePhase.GATHER_GERMANIUM: "germanium",
        GamePhase.GATHER_SILICON: "silicon",
        GamePhase.GATHER_CARBON: "carbon",
        GamePhase.GATHER_OXYGEN: "oxygen",
        GamePhase.RECHARGE: "charger",
    }

    def _g(state, ctx):
        resource = resource_map.get(phase)
        if not resource:
            return False

        # Check if policy_impl has extractor memory
        if not hasattr(ctx, "policy_impl") or not hasattr(ctx.policy_impl, "extractor_memory"):
            return False

        # Get all extractors for this resource
        extractors = ctx.policy_impl.extractor_memory.get_by_type(resource)
        if not extractors:
            return True  # No extractors discovered yet

        # Check if any are available (not depleted, not clipped, cooldown ready)
        available = [
            e
            for e in extractors
            if not e.is_depleted() and e.is_available(ctx.step, ctx.policy_impl.cooldown_remaining)
        ]
        return len(available) == 0

    return _g


def have_charger_discovered(state, ctx):
    """Check if charger has been discovered."""
    if not hasattr(ctx, "policy_impl") or not hasattr(ctx.policy_impl, "extractor_memory"):
        return False
    chargers = ctx.policy_impl.extractor_memory.get_by_type("charger")
    return len(chargers) > 0


# Define all transitions
def create_transitions() -> List[Transition]:
    """Create the complete transition table."""
    T: List[Transition] = [
        # Global/urgent - deposit heart when carrying one
        Transition(
            src=p,
            dst=GamePhase.DEPOSIT_HEART,
            guard=carrying_heart,
            priority=100,
            min_dwell_steps=0,
            on_enter=enter_deposit,
        )
        for p in GamePhase
    ] + [
        # Recharge transitions - if no charger found, explore to find it
        Transition(
            GamePhase.RECHARGE,
            GamePhase.EXPLORE,
            guard=lambda s, c: recharged_enough(s, c) and not have_charger_discovered(s, c),
            priority=60,
            on_enter=enter_explore("find_charger"),
            on_exit=exit_explore,
        ),
        # If charger found and recharged, resume gathering
        Transition(
            GamePhase.RECHARGE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: recharged_enough(s, c) and have_charger_discovered(s, c),
            priority=50,
            min_dwell_steps=1,
        ),
        # Any → RECHARGE when low
        *[
            Transition(src=p, dst=GamePhase.RECHARGE, guard=low_energy, priority=80, on_enter=enter_recharge)
            for p in GamePhase
            if p != GamePhase.RECHARGE
        ],
        # REACTIVE CLIPPING: When blocked by a clipped extractor during gathering
        # These transitions have higher priority (85) than the old proactive ones (80)
        # From any gathering phase, if blocked and need craft resource, go gather it
        *[
            Transition(
                src=p,
                dst=GamePhase.GATHER_CARBON,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "oxygen"
                and s.carbon < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for p in [GamePhase.GATHER_OXYGEN, GamePhase.GATHER_GERMANIUM, GamePhase.GATHER_SILICON]
        ],
        *[
            Transition(
                src=p,
                dst=GamePhase.GATHER_OXYGEN,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "carbon"
                and s.oxygen < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for p in [GamePhase.GATHER_CARBON, GamePhase.GATHER_GERMANIUM, GamePhase.GATHER_SILICON]
        ],
        *[
            Transition(
                src=p,
                dst=GamePhase.GATHER_SILICON,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "germanium"
                and s.silicon < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for p in [GamePhase.GATHER_CARBON, GamePhase.GATHER_OXYGEN, GamePhase.GATHER_GERMANIUM]
        ],
        *[
            Transition(
                src=p,
                dst=GamePhase.GATHER_GERMANIUM,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "silicon"
                and s.germanium < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for p in [GamePhase.GATHER_CARBON, GamePhase.GATHER_OXYGEN, GamePhase.GATHER_SILICON]
        ],
        # Once we have the craft resource, go craft the unclip item
        *[
            Transition(
                src=p,
                dst=GamePhase.CRAFT_DECODER,
                guard=lambda s, c: blocked_by_clipped(s, c) and not need_craft_resource_for_blocked(s, c),
                priority=85,
                min_dwell_steps=1,
                on_enter=enter_craft_decoder,
            )
            for p in [
                GamePhase.GATHER_CARBON,
                GamePhase.GATHER_OXYGEN,
                GamePhase.GATHER_GERMANIUM,
                GamePhase.GATHER_SILICON,
            ]
        ],
        # Assemble when ready & assembler known
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.ASSEMBLE_HEART,
            guard=lambda s, c: has_all_materials(s, c) and have_assembler_discovered(s, c),
            priority=40,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.ASSEMBLE_HEART,
            guard=lambda s, c: has_all_materials(s, c) and have_assembler_discovered(s, c),
            priority=40,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.ASSEMBLE_HEART,
            guard=lambda s, c: has_all_materials(s, c) and have_assembler_discovered(s, c),
            priority=40,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_OXYGEN,
            GamePhase.ASSEMBLE_HEART,
            guard=lambda s, c: has_all_materials(s, c) and have_assembler_discovered(s, c),
            priority=40,
            min_dwell_steps=2,
        ),
        # Normal resource gathering transitions
        # Note: Reactive clipping logic (priority 85) will override these if blocked by clipped extractor
        # GATHER_GERMANIUM -> GATHER_SILICON when we have enough germanium but not enough silicon
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: s.germanium >= 5 and s.silicon < 50,
            priority=60,
            min_dwell_steps=2,
        ),
        # GATHER_GERMANIUM -> GATHER_CARBON when we have enough germanium but not enough carbon
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: s.germanium >= 5 and s.carbon < 20,
            priority=55,
            min_dwell_steps=2,
        ),
        # GATHER_GERMANIUM -> GATHER_OXYGEN when we have enough germanium but not enough oxygen
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.germanium >= 5 and s.oxygen < 20,
            priority=50,
            min_dwell_steps=2,
        ),
        # GATHER_SILICON -> GATHER_CARBON when we have enough silicon but not enough carbon
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: s.silicon >= 50 and s.carbon < 20,
            priority=60,
            min_dwell_steps=2,
        ),
        # GATHER_SILICON -> GATHER_OXYGEN when we have enough silicon but not enough oxygen
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.silicon >= 50 and s.oxygen < 20,
            priority=55,
            min_dwell_steps=2,
        ),
        # GATHER_CARBON -> GATHER_OXYGEN when we have enough carbon but not enough oxygen
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.carbon >= 20 and s.oxygen < 20,
            priority=60,
            min_dwell_steps=2,
        ),
        # CRAFT_DECODER → EXPLORE: If assembler not discovered, explore to find it
        Transition(
            GamePhase.CRAFT_DECODER,
            GamePhase.EXPLORE,
            guard=lambda s, c: not have_assembler_discovered(s, c),
            priority=75,
            min_dwell_steps=1,
            on_enter=enter_explore("find_assembler"),
            on_exit=exit_explore,
        ),
        # Gathering phases → EXPLORE: If no extractors available, explore to find them
        *[
            Transition(
                src=phase,
                dst=GamePhase.EXPLORE,
                guard=no_extractors_available(phase),
                priority=50,
                min_dwell_steps=5,
                on_enter=enter_explore("find_extractor"),
                on_exit=exit_explore,
            )
            for phase in [
                GamePhase.GATHER_GERMANIUM,
                GamePhase.GATHER_SILICON,
                GamePhase.GATHER_CARBON,
                GamePhase.GATHER_OXYGEN,
            ]
        ],
        # CRAFT_DECODER → UNCLIP_STATION: After crafting the unclip item, go unclip
        Transition(
            GamePhase.CRAFT_DECODER,
            GamePhase.UNCLIP_STATION,
            guard=decoder_ready_for_unclipping,
            priority=70,
            min_dwell_steps=1,
            on_enter=enter_unclip_station,
        ),
        # UNCLIP_STATION → GATHER_*: After unclipping, return to gathering
        # The reactive logic will have cleared blocked_by_clipped_extractor, so normal gathering resumes
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.oxygen < 20,
            priority=65,
            min_dwell_steps=10,
            on_exit=exit_unclip_station,
        ),
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: s.carbon < 20,
            priority=65,
            min_dwell_steps=10,
            on_exit=exit_unclip_station,
        ),
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: s.germanium < 5,
            priority=65,
            min_dwell_steps=10,
            on_exit=exit_unclip_station,
        ),
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: s.silicon < 50,
            priority=65,
            min_dwell_steps=10,
            on_exit=exit_unclip_station,
        ),
        # EXPLORE → gathering phases (goal-aware and inventory-aware)
        # If exploring to find assembler, go back to crafting once found
        Transition(
            GamePhase.EXPLORE,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: s.explore_goal == "find_assembler" and have_assembler_discovered(s, c),
            priority=75,
            min_dwell_steps=5,
        ),
        # If exploring to find charger, resume gathering once found
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: s.explore_goal == "find_charger"
            and have_charger_discovered(s, c)
            and not low_energy(s, c),
            priority=40,
            min_dwell_steps=5,
        ),
        # If exploring to find extractors, resume gathering based on inventory needs
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: s.explore_goal == "find_extractor" and s.germanium < 5 and not low_energy(s, c),
            priority=30,
            min_dwell_steps=10,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: s.explore_goal == "find_extractor"
            and s.germanium >= 5
            and s.silicon < 50
            and not low_energy(s, c),
            priority=30,
            min_dwell_steps=10,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: s.explore_goal == "find_extractor"
            and s.silicon >= 50
            and s.carbon < 20
            and not low_energy(s, c),
            priority=30,
            min_dwell_steps=10,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.explore_goal == "find_extractor"
            and s.carbon >= 20
            and s.oxygen < 20
            and not low_energy(s, c),
            priority=30,
            min_dwell_steps=10,
        ),
        # If exploring to get unstuck, resume gathering after a while
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: s.explore_goal == "unstuck" and not low_energy(s, c),
            priority=20,
            min_dwell_steps=20,
        ),
        # Stalls → explore (to get unstuck)
        *[
            Transition(
                src=phase,
                dst=GamePhase.EXPLORE,
                guard=progress_stalled(60 if phase != GamePhase.GATHER_SILICON else 80),
                priority=10,
                min_dwell_steps=10,
                on_enter=enter_explore("unstuck"),
                on_exit=exit_explore,
            )
            for phase in [
                GamePhase.GATHER_GERMANIUM,
                GamePhase.GATHER_SILICON,
                GamePhase.GATHER_CARBON,
                GamePhase.GATHER_OXYGEN,
            ]
        ],
        # Transitions from DEPOSIT_HEART back to gathering phases
        Transition(
            GamePhase.DEPOSIT_HEART,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: not carrying_heart(s, c) and not low_energy(s, c),
            priority=30,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.DEPOSIT_HEART,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: not carrying_heart(s, c) and not low_energy(s, c),
            priority=25,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.DEPOSIT_HEART,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: not carrying_heart(s, c) and not low_energy(s, c),
            priority=20,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.DEPOSIT_HEART,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: not carrying_heart(s, c) and not low_energy(s, c),
            priority=15,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.DEPOSIT_HEART,
            GamePhase.RECHARGE,
            guard=lambda s, c: not carrying_heart(s, c) and low_energy(s, c),
            priority=50,
            min_dwell_steps=1,
        ),
    ]
    print(f"[DEBUG] Created {len(T)} transitions")
    for t in T:
        if t.src == GamePhase.EXPLORE and t.dst == GamePhase.GATHER_OXYGEN:
            print(f"  EXPLORE -> GATHER_OXYGEN: priority={t.priority}, min_dwell={t.min_dwell_steps}")
    return T


def create_controller(initial: GamePhase = GamePhase.GATHER_GERMANIUM) -> PhaseController:
    """Create a phase controller with all transitions."""
    transitions = create_transitions()
    return PhaseController(initial=initial, transitions=transitions)
