"""Streamlined hyperparameters for the scripted agent policy - only impactful parameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Hyperparameters:
    """Streamlined hyperparameters with only parameters that create meaningful behavioral differences."""

    # === Core Strategy ===
    strategy_type: str = "greedy_opportunistic"
    exploration_phase_steps: int = 100
    min_energy_for_silicon: int = 70

    # === Energy Management (IMPACTFUL) ===
    recharge_start_small: int = 65  # When to start recharging on small maps
    recharge_start_large: int = 45  # When to start recharging on large maps
    recharge_stop_small: int = 90  # When to stop recharging on small maps
    recharge_stop_large: int = 75  # When to stop recharging on large maps

    # === Cooldown/Waiting Behavior (IMPACTFUL) ===
    wait_if_cooldown_leq: int = 2  # Try using when cooldown <= this value

    # === Extractor Scoring (used in fallback logic) ===
    depletion_threshold: float = 0.25  # When to consider an extractor "low"

    # === Reproducibility ===
    seed: Optional[int] = None


# === Behavior Presets ===


def create_aggressive_preset() -> Hyperparameters:
    """Aggressive energy management - recharges early and often."""
    return Hyperparameters(
        strategy_type="aggressive",
        recharge_start_small=75,
        recharge_start_large=55,
        recharge_stop_small=95,
        recharge_stop_large=80,
        wait_if_cooldown_leq=1,  # Very impatient
    )


def create_conservative_preset() -> Hyperparameters:
    """Conservative energy management - waits longer before recharging."""
    return Hyperparameters(
        strategy_type="conservative",
        recharge_start_small=55,
        recharge_start_large=35,
        recharge_stop_small=85,
        recharge_stop_large=70,
        wait_if_cooldown_leq=3,  # More patient
    )


def create_balanced_preset() -> Hyperparameters:
    """Balanced approach - middle ground."""
    return Hyperparameters(
        strategy_type="balanced",
        recharge_start_small=65,
        recharge_start_large=45,
        recharge_stop_small=90,
        recharge_stop_large=75,
        wait_if_cooldown_leq=2,
    )


def create_impatient_preset() -> Hyperparameters:
    """Impatient waiting - tries to use extractors even with high cooldown."""
    return Hyperparameters(
        strategy_type="impatient",
        recharge_start_small=65,
        recharge_start_large=45,
        recharge_stop_small=90,
        recharge_stop_large=75,
        wait_if_cooldown_leq=0,  # Always try to use
    )


def create_patient_preset() -> Hyperparameters:
    """Patient waiting - waits for low cooldown before using."""
    return Hyperparameters(
        strategy_type="patient",
        recharge_start_small=65,
        recharge_start_large=45,
        recharge_stop_small=90,
        recharge_stop_large=75,
        wait_if_cooldown_leq=4,  # Very patient
    )


def create_mixture_presets() -> list[Hyperparameters]:
    """Create a diverse mixture of hyperparameter presets."""
    return [
        # Energy management variants
        Hyperparameters(
            strategy_type="energy_aggressive",
            seed=1,
            recharge_start_small=70,
            recharge_start_large=50,
            recharge_stop_small=95,
            recharge_stop_large=80,
            wait_if_cooldown_leq=2,
        ),
        Hyperparameters(
            strategy_type="energy_conservative",
            seed=2,
            recharge_start_small=60,
            recharge_start_large=40,
            recharge_stop_small=85,
            recharge_stop_large=70,
            wait_if_cooldown_leq=2,
        ),
        # Waiting behavior variants
        Hyperparameters(
            strategy_type="waiting_impatient",
            seed=3,
            recharge_start_small=65,
            recharge_start_large=45,
            recharge_stop_small=90,
            recharge_stop_large=75,
            wait_if_cooldown_leq=0,
        ),
        Hyperparameters(
            strategy_type="waiting_patient",
            seed=4,
            recharge_start_small=65,
            recharge_start_large=45,
            recharge_stop_small=90,
            recharge_stop_large=75,
            wait_if_cooldown_leq=4,
        ),
        # Balanced default
        Hyperparameters(
            strategy_type="balanced_default",
            seed=5,
            recharge_start_small=65,
            recharge_start_large=45,
            recharge_stop_small=90,
            recharge_stop_large=75,
            wait_if_cooldown_leq=2,
        ),
    ]


# === Legacy Support ===
# Keep the old hyperparameters class for backward compatibility
def create_legacy_hyperparameters() -> "LegacyHyperparameters":
    """Create a legacy hyperparameters object for backward compatibility."""
    return LegacyHyperparameters()


@dataclass
class LegacyHyperparameters:
    """Legacy hyperparameters class for backward compatibility."""

    # Core strategy
    strategy_type: str = "greedy_opportunistic"
    exploration_phase_steps: int = 100
    min_energy_for_silicon: int = 70

    # Energy management
    recharge_start_small: int = 65
    recharge_start_large: int = 45
    recharge_stop_small: int = 90
    recharge_stop_large: int = 75
    passive_regen_per_step: float = 1.0
    risk_buffer_moves: int = 10

    # Waiting behavior
    will_wait_max_steps: int = 12
    wait_if_cooldown_leq: int = 2
    rotate_on_cooldown_ge: int = 3
    prefer_short_queue_radius: int = 7

    # Scoring
    depletion_threshold: float = 0.25
    distance_weight: float = 0.7
    efficiency_weight: float = 0.3
    clip_avoidance_bias: float = 0.5

    # Phase policy
    germ_needed_base: int = 5
    assembler_greediness: float = 1.0
    deposit_hysteresis_steps: int = 2

    # Learning
    cooldown_learn_window: int = 5
    unseen_cooldown_default: int = 20
    estimation_priority: str = "last_used"
    estimation_mix_beta: float = 0.7

    # Pathfinding
    use_astar: bool = True
    astar_threshold: int = 20
    optimistic_planning: bool = True
    bfs_sweep_bias: float = 0.0

    # Anti-oscillation
    max_retarget_per_k_steps: int = 3
    retarget_window_k: int = 20
    jitter_target_every: int = 0

    # Unclipping
    auto_craft_decoder: bool = True
    unclip_max_distance: int = 30
    unclip_only_when_blocking: bool = False

    # Reproducibility
    seed: Optional[int] = None
