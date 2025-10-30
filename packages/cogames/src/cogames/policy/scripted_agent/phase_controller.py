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


def have_chest_discovered(state, ctx):
    return getattr(state, "chest_discovered", False)


def need_decoder_for_clipped(state, ctx):
    """Check if we need a decoder for any clipped extractor."""
    return (
        any(e.is_clipped for L in ctx.env.policy_impl.extractor_memory._extractors.values() for e in L)
        and state.decoder == 0
    )


def has_carbon_for_decoder(state, ctx):
    """Check if we have carbon to craft a decoder."""
    return state.carbon >= 1


def _resource_extractor_clipped(state, ctx, resource_type: str) -> bool:
    """Check if a specific resource extractor is clipped."""
    # Access policy_impl through the context (same pattern as need_decoder_for_clipped)
    if not hasattr(ctx, "env") or not hasattr(ctx.env, "policy_impl"):
        return False

    policy_impl = ctx.env.policy_impl
    extractors = policy_impl.extractor_memory.get_by_type(resource_type)
    return any(e.is_clipped for e in extractors)


def oxygen_extractor_clipped(state, ctx):
    """Check if oxygen extractor is clipped."""
    return _resource_extractor_clipped(state, ctx, "oxygen")


def carbon_extractor_clipped(state, ctx):
    """Check if carbon extractor is clipped."""
    return _resource_extractor_clipped(state, ctx, "carbon")


def germanium_extractor_clipped(state, ctx):
    """Check if germanium extractor is clipped."""
    return _resource_extractor_clipped(state, ctx, "germanium")


def silicon_extractor_clipped(state, ctx):
    """Check if silicon extractor is clipped."""
    return _resource_extractor_clipped(state, ctx, "silicon")


def decoder_ready_for_unclipping(state, ctx):
    """Check if we have the correct unclip item for any clipped extractor."""
    return has_unclip_item_for_clipped(state, ctx)


def any_extractor_clipped(state, ctx):
    """Check if ANY resource extractor is clipped."""
    return (
        oxygen_extractor_clipped(state, ctx)
        or carbon_extractor_clipped(state, ctx)
        or germanium_extractor_clipped(state, ctx)
        or silicon_extractor_clipped(state, ctx)
    )


def has_unclip_item_for_clipped(state, ctx) -> bool:
    """Check if we have the appropriate unclip item for any clipped resource.

    Mapping (from eval_missions.py):
    - Oxygen clipped → decoder (crafted from carbon)
    - Carbon clipped → modulator (crafted from oxygen)
    - Germanium clipped → resonator (crafted from silicon)
    - Silicon clipped → scrambler (crafted from germanium)
    """
    if oxygen_extractor_clipped(state, ctx) and state.decoder > 0:
        return True
    if carbon_extractor_clipped(state, ctx) and state.modulator > 0:
        return True
    if germanium_extractor_clipped(state, ctx) and state.resonator > 0:
        return True
    if silicon_extractor_clipped(state, ctx) and state.scrambler > 0:
        return True
    return False


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
        # Recharge transitions
        Transition(
            GamePhase.RECHARGE, GamePhase.EXPLORE, guard=recharged_enough, priority=50, on_enter=enter_gather_any
        ),
        # Any → RECHARGE when low
        *[
            Transition(src=p, dst=GamePhase.RECHARGE, guard=low_energy, priority=80, on_enter=enter_recharge)
            for p in GamePhase
            if p != GamePhase.RECHARGE
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
        # GATHER_GERMANIUM -> GATHER_SILICON when we have enough germanium but not enough silicon
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: not silicon_extractor_clipped(s, c) and s.germanium >= 5 and s.silicon < 50,
            priority=60,
            min_dwell_steps=2,
        ),
        # GATHER_GERMANIUM -> GATHER_CARBON when we have enough germanium but not enough carbon
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: not carbon_extractor_clipped(s, c) and s.germanium >= 5 and s.carbon < 20,
            priority=55,
            min_dwell_steps=2,
        ),
        # GATHER_GERMANIUM -> GATHER_OXYGEN when we have enough germanium but not enough oxygen
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: not oxygen_extractor_clipped(s, c) and s.germanium >= 5 and s.oxygen < 20,
            priority=50,
            min_dwell_steps=2,
        ),
        # GATHER_SILICON -> GATHER_CARBON when we have enough silicon but not enough carbon
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: not carbon_extractor_clipped(s, c) and s.silicon >= 50 and s.carbon < 20,
            priority=60,
            min_dwell_steps=2,
        ),
        # GATHER_SILICON -> GATHER_OXYGEN when we have enough silicon but not enough oxygen
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: not oxygen_extractor_clipped(s, c) and s.silicon >= 50 and s.oxygen < 20,
            priority=55,
            min_dwell_steps=2,
        ),
        # GATHER_CARBON -> GATHER_OXYGEN when we have enough carbon but not enough oxygen
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: not oxygen_extractor_clipped(s, c) and s.carbon >= 20 and s.oxygen < 20,
            priority=60,
            min_dwell_steps=2,
        ),
        # Clip handling - more specific transitions
        # From any gathering phase, prioritize oxygen if it's clipped AND we have decoder
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.decoder > 0,
            priority=85,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.decoder > 0,
            priority=85,
            min_dwell_steps=1,
        ),
        # If oxygen is clipped and we DON'T have a decoder yet, go get carbon first
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.decoder == 0 and s.carbon < 1,
            priority=78,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.decoder == 0 and s.carbon < 1,
            priority=78,
            min_dwell_steps=1,
        ),
        # GATHER_CARBON should transition to CRAFT_DECODER when it has carbon, not back to GATHER_OXYGEN
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.carbon >= 1 and s.decoder == 0,
            priority=80,
            min_dwell_steps=1,
            on_enter=enter_craft_decoder,
        ),
        # Within GATHER_OXYGEN, handle unclipping
        Transition(
            GamePhase.GATHER_OXYGEN,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and has_carbon_for_decoder(s, c) and s.decoder == 0,
            priority=80,
            min_dwell_steps=2,
            on_enter=enter_craft_decoder,
        ),
        Transition(
            GamePhase.GATHER_OXYGEN,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.carbon < 1 and s.decoder == 0,
            priority=75,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.CRAFT_DECODER,
            GamePhase.UNCLIP_STATION,
            guard=decoder_ready_for_unclipping,
            priority=70,
            min_dwell_steps=1,
            on_enter=enter_unclip_station,
        ),
        # After unclipping, go back to gathering the resource that was clipped
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: not oxygen_extractor_clipped(s, c) and s.oxygen < 20,
            priority=65,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: not carbon_extractor_clipped(s, c) and s.carbon < 20,
            priority=65,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: not germanium_extractor_clipped(s, c) and s.germanium < 5,
            priority=65,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.UNCLIP_STATION,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: not silicon_extractor_clipped(s, c) and s.silicon < 50,
            priority=65,
            min_dwell_steps=1,
        ),
        # Generalized clipping: Route TO the clipped resource phase first (higher priority)
        # If carbon is clipped and we need it, go to GATHER_CARBON (even though it's clipped)
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: carbon_extractor_clipped(s, c) and s.carbon < 20 and s.germanium >= 5,
            priority=79,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: carbon_extractor_clipped(s, c) and s.carbon < 20 and s.silicon >= 50,
            priority=79,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_OXYGEN,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: carbon_extractor_clipped(s, c) and s.carbon < 20 and s.oxygen >= 20,
            priority=79,
            min_dwell_steps=1,
        ),
        # If germanium is clipped and we need it, go to GATHER_GERMANIUM
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: germanium_extractor_clipped(s, c) and s.germanium < 5,
            priority=79,
            min_dwell_steps=1,
        ),
        # If silicon is clipped and we need it, go to GATHER_SILICON
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: silicon_extractor_clipped(s, c) and s.silicon < 50 and s.germanium >= 5,
            priority=79,
            min_dwell_steps=1,
        ),
        # When IN a gathering phase and that resource is clipped, get the correct resource to craft unclip item
        # GATHER_OXYGEN is clipped -> get carbon to craft decoder
        Transition(
            GamePhase.GATHER_OXYGEN,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.decoder == 0 and s.carbon < 1,
            priority=78,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_OXYGEN,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: oxygen_extractor_clipped(s, c) and s.carbon >= 1 and s.decoder == 0,
            priority=80,
            min_dwell_steps=1,
            on_enter=enter_craft_decoder,
        ),
        # GATHER_CARBON is clipped -> get oxygen to craft modulator
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: carbon_extractor_clipped(s, c) and s.modulator == 0 and s.oxygen < 1,
            priority=78,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: carbon_extractor_clipped(s, c) and s.oxygen >= 1 and s.modulator == 0,
            priority=80,
            min_dwell_steps=1,
            on_enter=enter_craft_decoder,
        ),
        # GATHER_GERMANIUM is clipped -> get silicon to craft resonator
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: germanium_extractor_clipped(s, c) and s.resonator == 0 and s.silicon < 1,
            priority=78,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: germanium_extractor_clipped(s, c) and s.silicon >= 1 and s.resonator == 0,
            priority=80,
            min_dwell_steps=1,
            on_enter=enter_craft_decoder,
        ),
        # GATHER_SILICON is clipped -> get germanium to craft scrambler
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: silicon_extractor_clipped(s, c) and s.scrambler == 0 and s.germanium < 1,
            priority=78,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: silicon_extractor_clipped(s, c) and s.germanium >= 1 and s.scrambler == 0,
            priority=80,
            min_dwell_steps=1,
            on_enter=enter_craft_decoder,
        ),
        # EXPLORE → gathering phases
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: not low_energy(s, c) and oxygen_extractor_clipped(s, c),
            priority=40,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: not low_energy(s, c)
            and oxygen_extractor_clipped(s, c)
            and s.decoder == 0
            and s.carbon < 1,
            priority=35,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: not low_energy(s, c),
            priority=30,
            min_dwell_steps=5,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: not low_energy(s, c),
            priority=20,
            min_dwell_steps=5,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: not low_energy(s, c),
            priority=15,
            min_dwell_steps=5,
        ),
        # Stalls → explore
        Transition(
            GamePhase.GATHER_GERMANIUM, GamePhase.EXPLORE, guard=progress_stalled(60), priority=10, min_dwell_steps=10
        ),
        Transition(
            GamePhase.GATHER_SILICON, GamePhase.EXPLORE, guard=progress_stalled(80), priority=10, min_dwell_steps=10
        ),
        Transition(
            GamePhase.GATHER_CARBON, GamePhase.EXPLORE, guard=progress_stalled(60), priority=10, min_dwell_steps=10
        ),
        Transition(
            GamePhase.GATHER_OXYGEN, GamePhase.EXPLORE, guard=progress_stalled(60), priority=10, min_dwell_steps=10
        ),
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
