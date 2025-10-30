"""
Hyperparameter Presets for Scripted Agent

Curated set of 10 presets that vary the impactful knobs:
- strategy_type, exploration_phase_steps, min_energy_for_silicon
- recharge_start_small/large, recharge_stop_small/large
- wait_if_cooldown_leq, depletion_threshold
"""

from cogames.policy.scripted_agent.hyperparameters import Hyperparameters

# Additional targeted presets
EXPLORER_SHORT = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=50,
    min_energy_for_silicon=65,
    recharge_start_small=68,
    recharge_start_large=48,
    recharge_stop_small=88,
    recharge_stop_large=73,
    wait_if_cooldown_leq=1,
    depletion_threshold=0.25,
)

EXPLORER_LONG = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=150,
    min_energy_for_silicon=75,
    recharge_start_small=60,
    recharge_start_large=40,
    recharge_stop_small=85,
    recharge_stop_large=70,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

GREEDY_AGGRESSIVE = Hyperparameters(
    strategy_type="greedy_opportunistic",
    exploration_phase_steps=25,
    min_energy_for_silicon=55,
    recharge_start_small=70,
    recharge_start_large=50,
    recharge_stop_small=90,
    recharge_stop_large=75,
    wait_if_cooldown_leq=0,
    depletion_threshold=0.20,
)

GREEDY_CONSERVATIVE = Hyperparameters(
    strategy_type="greedy_opportunistic",
    exploration_phase_steps=75,
    min_energy_for_silicon=80,
    recharge_start_small=70,
    recharge_start_large=50,
    recharge_stop_small=85,
    recharge_stop_large=70,
    wait_if_cooldown_leq=3,
    depletion_threshold=0.30,
)

EFFICIENCY_HEAVY = Hyperparameters(
    strategy_type="efficiency_learner",
    exploration_phase_steps=120,
    min_energy_for_silicon=80,
    recharge_start_small=65,
    recharge_start_large=45,
    recharge_stop_small=92,
    recharge_stop_large=78,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.20,
)

EFFICIENCY_LIGHT = Hyperparameters(
    strategy_type="efficiency_learner",
    exploration_phase_steps=80,
    min_energy_for_silicon=65,
    recharge_start_small=65,
    recharge_start_large=45,
    recharge_stop_small=90,
    recharge_stop_large=75,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

SEQUENTIAL_BASELINE = Hyperparameters(
    strategy_type="sequential_simple",
    exploration_phase_steps=50,
    min_energy_for_silicon=70,
    recharge_start_small=65,
    recharge_start_large=45,
    recharge_stop_small=90,
    recharge_stop_large=75,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

SILICON_RUSH = Hyperparameters(
    strategy_type="greedy_opportunistic",
    exploration_phase_steps=50,
    min_energy_for_silicon=60,
    recharge_start_small=65,
    recharge_start_large=45,
    recharge_stop_small=88,
    recharge_stop_large=73,
    wait_if_cooldown_leq=1,
    depletion_threshold=0.20,
)

OXYGEN_SAFE = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=100,
    min_energy_for_silicon=85,
    recharge_start_small=70,
    recharge_start_large=50,
    recharge_stop_small=95,
    recharge_stop_large=80,
    wait_if_cooldown_leq=3,
    depletion_threshold=0.30,
)

BALANCED = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=80,
    min_energy_for_silicon=70,
    recharge_start_small=65,
    recharge_start_large=45,
    recharge_stop_small=90,
    recharge_stop_large=75,
    wait_if_cooldown_leq=2,
    depletion_threshold=0.25,
)

# =============================================================================
# Preset Dictionary (for easy access)
# =============================================================================

HYPERPARAMETER_PRESETS = {
    # Exactly 10 curated presets using all impactful hypers
    "balanced": BALANCED,
    "explorer_short": EXPLORER_SHORT,
    "explorer_long": EXPLORER_LONG,
    "greedy_aggressive": GREEDY_AGGRESSIVE,
    "greedy_conservative": GREEDY_CONSERVATIVE,
    "efficiency_light": EFFICIENCY_LIGHT,
    "efficiency_heavy": EFFICIENCY_HEAVY,
    "sequential_baseline": SEQUENTIAL_BASELINE,
    "silicon_rush": SILICON_RUSH,
    "oxygen_safe": OXYGEN_SAFE,
}
