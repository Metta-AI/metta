"""Phase controller package exposing the scripted agent FSM utilities."""

from .controller import Context, GamePhase, Guard, Hook, PhaseController, Transition
from .guards import (
    assemble_slot_available,
    blocked_by_clipped,
    carrying_heart,
    decoder_ready_for_unclipping,
    get_blocked_extractor_resource_type,
    has_all_materials,
    have_assembler_discovered,
    have_chest_discovered,
    low_energy,
    need_craft_resource_for_blocked,
    no_extractors_available,
    progress_stalled,
    recharged_enough,
)
from .hooks import (
    enter_craft_decoder,
    enter_deposit,
    enter_explore,
    enter_gather_any,
    enter_recharge,
    enter_unclip_station,
    exit_explore,
    exit_unclip_station,
    get_last_gathering_phase,
)
from .transitions import create_controller, create_transitions

__all__ = [
    "Guard",
    "Hook",
    "Transition",
    "Context",
    "GamePhase",
    "PhaseController",
    "create_controller",
    "create_transitions",
    # Guards
    "assemble_slot_available",
    "blocked_by_clipped",
    "carrying_heart",
    "decoder_ready_for_unclipping",
    "get_blocked_extractor_resource_type",
    "has_all_materials",
    "have_assembler_discovered",
    "have_chest_discovered",
    "low_energy",
    "need_craft_resource_for_blocked",
    "no_extractors_available",
    "progress_stalled",
    "recharged_enough",
    # Hooks
    "enter_craft_decoder",
    "enter_deposit",
    "enter_explore",
    "enter_gather_any",
    "enter_recharge",
    "enter_unclip_station",
    "exit_explore",
    "exit_unclip_station",
    "get_last_gathering_phase",
]
