"""
Hyperparameter Presets for Scripted Agent

Curated core set of 5 presets that vary the impactful knobs:
- strategy_type, exploration_phase_steps, min_energy_for_silicon
- recharge_start_small/large, recharge_stop_small/large
- wait_if_cooldown_leq, depletion_threshold
"""

from cogames.policy.scripted_agent.hyperparameters import Hyperparameters

# Core presets
EXPLORER_LONG = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=150,
    min_energy_for_silicon=75,
    recharge_start_small=35,
    recharge_start_large=20,
    recharge_stop_small=100,
    recharge_stop_large=100,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

GREEDY_CONSERVATIVE = Hyperparameters(
    strategy_type="greedy_opportunistic",
    exploration_phase_steps=75,
    min_energy_for_silicon=80,
    recharge_start_small=50,
    recharge_start_large=35,
    recharge_stop_small=100,
    recharge_stop_large=100,
    wait_if_cooldown_leq=3,
    depletion_threshold=0.30,
)

EFFICIENCY_HEAVY = Hyperparameters(
    strategy_type="efficiency_learner",
    exploration_phase_steps=120,
    min_energy_for_silicon=80,
    recharge_start_small=40,
    recharge_start_large=25,
    recharge_stop_small=100,
    recharge_stop_large=100,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.20,
)

SEQUENTIAL_BASELINE = Hyperparameters(
    strategy_type="sequential_simple",
    exploration_phase_steps=50,
    min_energy_for_silicon=70,
    recharge_start_small=40,
    recharge_start_large=25,
    recharge_stop_small=100,
    recharge_stop_large=100,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

BALANCED = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=80,
    min_energy_for_silicon=70,
    recharge_start_small=40,
    recharge_start_large=25,
    recharge_stop_small=100,
    recharge_stop_large=100,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

# =============================================================================
# Preset Dictionary (for easy access)
# =============================================================================

HYPERPARAMETER_PRESETS = {
    # Core 5 presets
    "balanced": BALANCED,
    "explorer_long": EXPLORER_LONG,
    "greedy_conservative": GREEDY_CONSERVATIVE,
    "efficiency_heavy": EFFICIENCY_HEAVY,
    "sequential_baseline": SEQUENTIAL_BASELINE,
}
