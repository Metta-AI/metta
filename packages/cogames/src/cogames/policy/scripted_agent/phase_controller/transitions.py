"""Transition table assembly for the scripted agent phase machine."""

from __future__ import annotations

import logging
from typing import List

from .controller import GamePhase, PhaseController, Transition
from .guards import (
    blocked_by_clipped,
    carrying_heart,
    decoder_ready_for_unclipping,
    get_blocked_extractor_resource_type,
    has_all_materials,
    have_assembler_discovered,
    have_charger_discovered,
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
    enter_recharge,
    enter_unclip_station,
    exit_explore,
    exit_unclip_station,
    get_last_gathering_phase,
)

logger = logging.getLogger(__name__)


def create_transitions() -> List[Transition]:
    transitions: List[Transition] = [
        *(
            Transition(
                src=phase,
                dst=GamePhase.DEPOSIT_HEART,
                guard=carrying_heart,
                priority=100,
                min_dwell_steps=0,
                on_enter=enter_deposit,
            )
            for phase in GamePhase
            if phase != GamePhase.DEPOSIT_HEART
        ),
        Transition(
            GamePhase.RECHARGE,
            GamePhase.EXPLORE,
            guard=lambda s, c: recharged_enough(s, c) and not have_charger_discovered(s, c),
            priority=60,
            on_enter=enter_explore("find_charger"),
            on_exit=exit_explore,
        ),
        Transition(
            GamePhase.RECHARGE,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: (
                recharged_enough(s, c)
                and have_charger_discovered(s, c)
                and get_last_gathering_phase(s) == GamePhase.GATHER_CARBON
            ),
            priority=50,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.RECHARGE,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: (
                recharged_enough(s, c)
                and have_charger_discovered(s, c)
                and get_last_gathering_phase(s) == GamePhase.GATHER_OXYGEN
            ),
            priority=50,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.RECHARGE,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: (
                recharged_enough(s, c)
                and have_charger_discovered(s, c)
                and get_last_gathering_phase(s) == GamePhase.GATHER_SILICON
            ),
            priority=50,
            min_dwell_steps=1,
        ),
        Transition(
            GamePhase.RECHARGE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: (
                recharged_enough(s, c)
                and have_charger_discovered(s, c)
                and get_last_gathering_phase(s) == GamePhase.GATHER_GERMANIUM
            ),
            priority=50,
            min_dwell_steps=1,
        ),
        *[
            Transition(src=phase, dst=GamePhase.RECHARGE, guard=low_energy, priority=80, on_enter=enter_recharge)
            for phase in GamePhase
            if phase != GamePhase.RECHARGE
        ],
        *[
            Transition(
                src=phase,
                dst=GamePhase.GATHER_CARBON,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "oxygen"
                and s.carbon < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for phase in [GamePhase.GATHER_OXYGEN, GamePhase.GATHER_GERMANIUM, GamePhase.GATHER_SILICON]
        ],
        *[
            Transition(
                src=phase,
                dst=GamePhase.GATHER_OXYGEN,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "carbon"
                and s.oxygen < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for phase in [GamePhase.GATHER_CARBON, GamePhase.GATHER_GERMANIUM, GamePhase.GATHER_SILICON]
        ],
        *[
            Transition(
                src=phase,
                dst=GamePhase.GATHER_SILICON,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "germanium"
                and s.silicon < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for phase in [GamePhase.GATHER_CARBON, GamePhase.GATHER_OXYGEN, GamePhase.GATHER_GERMANIUM]
        ],
        *[
            Transition(
                src=phase,
                dst=GamePhase.GATHER_GERMANIUM,
                guard=lambda s, c: blocked_by_clipped(s, c)
                and get_blocked_extractor_resource_type(s, c) == "silicon"
                and s.germanium < 1,
                priority=85,
                min_dwell_steps=1,
            )
            for phase in [GamePhase.GATHER_CARBON, GamePhase.GATHER_OXYGEN, GamePhase.GATHER_SILICON]
        ],
        *[
            Transition(
                src=phase,
                dst=GamePhase.CRAFT_DECODER,
                guard=lambda s, c: blocked_by_clipped(s, c) and not need_craft_resource_for_blocked(s, c),
                priority=85,
                min_dwell_steps=1,
                on_enter=enter_craft_decoder,
            )
            for phase in [
                GamePhase.GATHER_CARBON,
                GamePhase.GATHER_OXYGEN,
                GamePhase.GATHER_GERMANIUM,
                GamePhase.GATHER_SILICON,
            ]
        ],
        *[
            Transition(
                src=phase,
                dst=GamePhase.ASSEMBLE_HEART,
                guard=lambda s, c: has_all_materials(s, c) and have_assembler_discovered(s, c),
                priority=40,
                min_dwell_steps=2,
            )
            for phase in (
                GamePhase.GATHER_GERMANIUM,
                GamePhase.GATHER_SILICON,
                GamePhase.GATHER_CARBON,
                GamePhase.GATHER_OXYGEN,
            )
        ],
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_SILICON,
            guard=lambda s, c: s.germanium >= 5 and s.silicon < 50,
            priority=60,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: s.germanium >= 5 and s.carbon < 20,
            priority=55,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_GERMANIUM,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.germanium >= 5 and s.oxygen < 20,
            priority=50,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_CARBON,
            guard=lambda s, c: s.silicon >= 50 and s.carbon < 20,
            priority=60,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_SILICON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.silicon >= 50 and s.oxygen < 20,
            priority=55,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.GATHER_CARBON,
            GamePhase.GATHER_OXYGEN,
            guard=lambda s, c: s.carbon >= 20 and s.oxygen < 20,
            priority=60,
            min_dwell_steps=2,
        ),
        Transition(
            GamePhase.CRAFT_DECODER,
            GamePhase.EXPLORE,
            guard=lambda s, c: not have_assembler_discovered(s, c),
            priority=75,
            min_dwell_steps=1,
            on_enter=enter_explore("find_assembler"),
            on_exit=exit_explore,
        ),
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
        Transition(
            GamePhase.CRAFT_DECODER,
            GamePhase.UNCLIP_STATION,
            guard=decoder_ready_for_unclipping,
            priority=70,
            min_dwell_steps=1,
            on_enter=enter_unclip_station,
        ),
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
        Transition(
            GamePhase.EXPLORE,
            GamePhase.CRAFT_DECODER,
            guard=lambda s, c: s.explore_goal == "find_assembler" and have_assembler_discovered(s, c),
            priority=75,
            min_dwell_steps=5,
        ),
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: s.explore_goal == "find_charger"
            and have_charger_discovered(s, c)
            and not low_energy(s, c),
            priority=40,
            min_dwell_steps=5,
        ),
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
        Transition(
            GamePhase.EXPLORE,
            GamePhase.GATHER_GERMANIUM,
            guard=lambda s, c: s.explore_goal == "unstuck" and not low_energy(s, c),
            priority=20,
            min_dwell_steps=20,
        ),
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

    logger.debug("Created %d transitions", len(transitions))
    return transitions


def create_controller(initial: GamePhase = GamePhase.GATHER_GERMANIUM) -> PhaseController:
    return PhaseController(initial=initial, transitions=create_transitions())


__all__ = ["create_transitions", "create_controller"]
