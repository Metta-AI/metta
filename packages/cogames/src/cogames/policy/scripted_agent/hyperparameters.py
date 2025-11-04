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

    # === Exploration & Probes ===
    use_probes: bool = False
    enable_visit_scoring: bool = True
    enable_probe_module: bool = True
    probe_frontier_radius: int = 12
    probe_max_targets: int = 6
    probe_revisit_cooldown: int = 200
    max_recent_positions: int = 10

    # === Energy Management ===
    recharge_start_small: int = 40
    recharge_start_large: int = 25
    recharge_stop_small: int = 100
    recharge_stop_large: int = 100
    recharge_until_full: bool = True

    # === Cooldown/Waiting Behavior ===
    wait_if_cooldown_leq: int = 2
    max_patience_steps: int = 12
    patience_multiplier: float = 1.0

    # === Extractor Scoring ===
    depletion_threshold: float = 0.25

    # === Coordination & Targeting ===
    assembly_signal_timeout: int = 30
    resource_focus_limits: dict[str, int] | None = None
    enable_target_reservation: bool = True
    enable_resource_focus_limits: bool = True
    enable_assembly_coordination: bool = True

    # === Navigation ===
    enable_navigation_cache: bool = True

    # === Reproducibility ===
    seed: Optional[int] = None


# === Behavior Presets ===


def story_mode_params() -> Hyperparameters:
    limits = {"carbon": 4, "oxygen": 4, "germanium": 2, "silicon": 2}
    return Hyperparameters(
        strategy_type="story_mode",
        use_probes=False,
        enable_probe_module=False,
        exploration_phase_steps=120,
        min_energy_for_silicon=60,
        recharge_start_small=60,
        recharge_start_large=45,
        recharge_stop_small=100,
        recharge_stop_large=100,
        recharge_until_full=True,
        wait_if_cooldown_leq=5,
        max_patience_steps=20,
        patience_multiplier=1.2,
        max_recent_positions=20,
        depletion_threshold=0.30,
        assembly_signal_timeout=120,
        resource_focus_limits=limits,
        enable_visit_scoring=True,
        enable_target_reservation=True,
        enable_resource_focus_limits=True,
        enable_assembly_coordination=True,
        enable_navigation_cache=True,
    )


def courier_params() -> Hyperparameters:
    limits = {"carbon": 3, "oxygen": 3, "germanium": 2, "silicon": 2}
    return Hyperparameters(
        strategy_type="courier",
        use_probes=False,
        enable_probe_module=False,
        exploration_phase_steps=60,
        min_energy_for_silicon=65,
        recharge_start_small=40,
        recharge_start_large=30,
        recharge_stop_small=85,
        recharge_stop_large=90,
        recharge_until_full=False,
        wait_if_cooldown_leq=3,
        max_patience_steps=10,
        patience_multiplier=1.0,
        max_recent_positions=6,
        depletion_threshold=0.25,
        assembly_signal_timeout=45,
        resource_focus_limits=limits,
        enable_visit_scoring=True,
        enable_target_reservation=True,
        enable_resource_focus_limits=True,
        enable_assembly_coordination=True,
        enable_navigation_cache=True,
    )


def scout_params() -> Hyperparameters:
    limits = {"carbon": 2, "oxygen": 2, "germanium": 2, "silicon": 2}
    return Hyperparameters(
        strategy_type="scout",
        use_probes=True,
        enable_probe_module=True,
        probe_frontier_radius=18,
        probe_max_targets=12,
        probe_revisit_cooldown=250,
        exploration_phase_steps=160,
        min_energy_for_silicon=55,
        recharge_start_small=30,
        recharge_start_large=20,
        recharge_stop_small=80,
        recharge_stop_large=85,
        recharge_until_full=False,
        wait_if_cooldown_leq=1,
        max_patience_steps=6,
        patience_multiplier=0.8,
        max_recent_positions=12,
        depletion_threshold=0.20,
        assembly_signal_timeout=30,
        resource_focus_limits=limits,
        enable_visit_scoring=True,
        enable_target_reservation=True,
        enable_resource_focus_limits=True,
        enable_assembly_coordination=True,
        enable_navigation_cache=True,
    )


def hoarder_params() -> Hyperparameters:
    limits = {"carbon": 5, "oxygen": 5, "germanium": 3, "silicon": 3}
    return Hyperparameters(
        strategy_type="hoarder",
        use_probes=False,
        enable_probe_module=False,
        exploration_phase_steps=90,
        min_energy_for_silicon=80,
        recharge_start_small=55,
        recharge_start_large=40,
        recharge_stop_small=100,
        recharge_stop_large=100,
        recharge_until_full=True,
        wait_if_cooldown_leq=8,
        max_patience_steps=30,
        patience_multiplier=2.0,
        max_recent_positions=8,
        depletion_threshold=0.35,
        assembly_signal_timeout=50,
        resource_focus_limits=limits,
        enable_visit_scoring=True,
        enable_target_reservation=True,
        enable_resource_focus_limits=True,
        enable_assembly_coordination=True,
        enable_navigation_cache=True,
    )


def minimal_baseline_params() -> Hyperparameters:
    """Minimal baseline with only essential features for systematic ablation studies."""
    return Hyperparameters(
        strategy_type="minimal_baseline",
        use_probes=False,
        enable_probe_module=False,
        enable_visit_scoring=True,  # REQUIRED for exploration movement to work
        exploration_phase_steps=80,
        min_energy_for_silicon=60,
        recharge_start_small=40,
        recharge_start_large=30,
        recharge_stop_small=90,
        recharge_stop_large=85,
        recharge_until_full=False,
        wait_if_cooldown_leq=2,
        max_patience_steps=8,
        patience_multiplier=1.0,
        max_recent_positions=5,
        depletion_threshold=0.15,
        assembly_signal_timeout=30,
        resource_focus_limits=None,
        enable_target_reservation=False,
        enable_resource_focus_limits=False,
        enable_assembly_coordination=False,
        enable_navigation_cache=False,
    )


# Legacy factories kept for compatibility


def create_aggressive_preset() -> Hyperparameters:
    return courier_params()


def create_conservative_preset() -> Hyperparameters:
    return hoarder_params()


def create_balanced_preset() -> Hyperparameters:
    return scout_params()


def create_impatient_preset() -> Hyperparameters:
    params = scout_params()
    params.wait_if_cooldown_leq = 0
    params.max_patience_steps = 4
    return params


def create_patient_preset() -> Hyperparameters:
    params = hoarder_params()
    params.wait_if_cooldown_leq = 10
    params.max_patience_steps = 40
    return params


def create_mixture_presets() -> list[Hyperparameters]:
    return [
        story_mode_params(),
        courier_params(),
        scout_params(),
        hoarder_params(),
    ]


# === Legacy Support ===
# Keep the old hyperparameters class for backward compatibility


def create_legacy_hyperparameters() -> "LegacyHyperparameters":
    legacy = LegacyHyperparameters()
    legacy.strategy_type = "legacy_balanced"
    return legacy


@dataclass
class LegacyHyperparameters:
    """Legacy hyperparameters class for backward compatibility."""

    # Core strategy
    strategy_type: str = "greedy_opportunistic"
    exploration_phase_steps: int = 100
    min_energy_for_silicon: int = 70

    # Energy management
    recharge_start_small: int = 40
    recharge_start_large: int = 25
    recharge_stop_small: int = 100
    recharge_stop_large: int = 100
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
