"""Hyperparameters for the scripted agent policy."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Hyperparameters:
    """Hyperparameters controlling scripted agent behavior."""

    # === Core Strategy ===
    strategy_type: str = "greedy_opportunistic"
    exploration_phase_steps: int = 100
    min_energy_for_silicon: int = 70

    # === Exploration Style ===
    levy_alpha: float = 1.3            # 1.1–1.9; lower = longer flights (wilder exploration)
    levy_step_scale: float = 8.0       # 2–20; average step length multiplier
    frontier_center_bias: float = 0.5  # 0–1; weight of "toward map center" vs BFS
    frontier_home_radius: int = 50     # 0 disables; otherwise confines early search
    exploration_noise_eps: float = 0.05  # 0–0.3; epsilon-go-random during explore

    # === Cooldown/Waiting Temperament ===
    will_wait_max_steps: int = 12         # 0–200; patience while adjacent & waiting
    wait_if_cooldown_leq: int = 2         # try-use when rem<=k; 0,1,2 typical
    rotate_on_cooldown_ge: int = 3        # if est rem>=k, prefer rotate to another extractor
    prefer_short_queue_radius: int = 7    # if another candidate available within R, rotate

    # === Extractor Choice & Depletion Attitude ===
    depletion_threshold: float = 0.25     # 0–0.5; when to tag "low"
    distance_weight: float = 0.7          # 0–1; higher = closer wins
    efficiency_weight: float = 0.3        # 0–1; higher = avg_output wins
    clip_avoidance_bias: float = 0.5      # 0–1; discount clipped ones even if "unclippable"

    # === Energy/Risk Style ===
    passive_regen_per_step: float = 1.0   # matches env fact; try 0.5–1.5
    recharge_start_small: int = 65
    recharge_start_large: int = 45
    recharge_stop_small: int = 90
    recharge_stop_large: int = 75
    risk_buffer_moves: int = 10           # extra energy buffer (in "net-move" units)

    # === Phase Policy Knobs ===
    germ_needed_base: int = 5             # for first heart; later you already reduce
    assembler_greediness: float = 1.0     # 0–1; how aggressively jump to assemble once ready
    deposit_hysteresis_steps: int = 2     # stay on deposit for a few steps after acquire

    # === Learning / Estimation Behavior ===
    cooldown_learn_window: int = 5        # steps after use within which to re-learn total cd
    unseen_cooldown_default: int = 20     # fallback if never used/seen
    estimation_priority: str = "last_used"  # "last_used" | "observed" | "mix"
    estimation_mix_beta: float = 0.7      # 0–1; blend weight for mix

    # === Pathfinding Flavor ===
    use_astar: bool = True
    astar_threshold: int = 20
    optimistic_planning: bool = True      # treat unknown as walkable during target nav
    bfs_sweep_bias: float = 0.0           # 0–1; >0 biases boustrophedon sweep even when a* equals

    # === Anti-oscillation / Randomness ===
    max_retarget_per_k_steps: int = 3     # throttle flip-flopping between targets
    retarget_window_k: int = 20
    jitter_target_every: int = 0          # 0 disables; every N steps add ±1 cell jitter

    # === Unclipping Aggressiveness ===
    auto_craft_decoder: bool = True
    unclip_max_distance: int = 30         # skip unclipping far-away stations
    unclip_only_when_blocking: bool = False  # only unclip if needed for current resource

    # === Reproducibility ===
    seed: Optional[int] = None


# === Behavior Presets ===

def create_forager_preset() -> Hyperparameters:
    """Wild explorer - aggressive exploration, low patience."""
    return Hyperparameters(
        strategy_type="forager",
        levy_alpha=1.2,
        levy_step_scale=14,
        exploration_noise_eps=0.15,
        frontier_center_bias=0.2,
        rotate_on_cooldown_ge=2,
        will_wait_max_steps=3,
        exploration_phase_steps=150,
    )


def create_waiter_preset() -> Hyperparameters:
    """Patient opportunist - high patience, efficiency focused."""
    return Hyperparameters(
        strategy_type="waiter",
        rotate_on_cooldown_ge=6,
        will_wait_max_steps=40,
        wait_if_cooldown_leq=2,
        distance_weight=0.6,
        efficiency_weight=0.4,
        depletion_threshold=0.2,
    )


def create_sprinter_preset() -> Hyperparameters:
    """Energy-risk taker - aggressive energy management."""
    return Hyperparameters(
        strategy_type="sprinter",
        passive_regen_per_step=1.2,
        risk_buffer_moves=2,
        recharge_start_small=45,
        recharge_stop_small=70,
        assembler_greediness=1.0,
        recharge_start_large=35,
        recharge_stop_large=60,
    )


def create_hoarder_preset() -> Hyperparameters:
    """Low depletion tolerance - avoids depleted extractors."""
    return Hyperparameters(
        strategy_type="hoarder",
        depletion_threshold=0.35,
        efficiency_weight=0.5,
        distance_weight=0.5,
        prefer_short_queue_radius=10,
        clip_avoidance_bias=0.3,
    )


def create_paranoid_preset() -> Hyperparameters:
    """Safe & methodical - conservative energy management."""
    return Hyperparameters(
        strategy_type="paranoid",
        risk_buffer_moves=18,
        recharge_start_small=75,
        recharge_stop_small=95,
        assembler_greediness=0.5,
        frontier_home_radius=40,
        recharge_start_large=60,
        recharge_stop_large=85,
    )


def create_decoder_first_preset() -> Hyperparameters:
    """Terrain unlocker - aggressive unclipping."""
    return Hyperparameters(
        strategy_type="decoder_first",
        clip_avoidance_bias=0.1,
        auto_craft_decoder=True,
        unclip_only_when_blocking=False,
        unclip_max_distance=999,
        assembler_greediness=0.8,
    )


def create_mixture_presets() -> list[Hyperparameters]:
    """Create a diverse mixture of hyperparameter presets."""
    return [
        # Default balanced
        Hyperparameters(strategy_type="balanced", seed=1),

        # Exploration focused
        Hyperparameters(
            strategy_type="explorer",
            seed=2,
            levy_alpha=1.2,
            levy_step_scale=12,
            exploration_noise_eps=0.1,
            exploration_phase_steps=200,
        ),

        # Efficiency focused
        Hyperparameters(
            strategy_type="efficiency",
            seed=3,
            distance_weight=0.5,
            efficiency_weight=0.5,
            will_wait_max_steps=30,
            depletion_threshold=0.2,
        ),

        # Energy conservative
        Hyperparameters(
            strategy_type="conservative",
            seed=4,
            passive_regen_per_step=0.8,
            risk_buffer_moves=16,
            assembler_greediness=0.6,
            recharge_start_small=70,
            recharge_stop_small=85,
        ),

        # Unclipping focused
        Hyperparameters(
            strategy_type="unclipper",
            seed=5,
            clip_avoidance_bias=0.2,
            auto_craft_decoder=True,
            unclip_max_distance=50,
            assembler_greediness=0.9,
        ),
    ]
