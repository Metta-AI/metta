"""
Hyperparameter Presets for Scripted Agent

Simplified presets focusing on the 3 most impactful hyperparameters:
1. strategy_type: High-level decision-making strategy
2. exploration_phase_steps: Duration of initial exploration (scales with map size)
3. min_energy_for_silicon: Energy threshold for silicon gathering
"""

from cogames.policy.scripted_agent import Hyperparameters

# =============================================================================
# Simplified Presets (5 distinct strategies)
# =============================================================================

EXPLORER = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=100,  # Will be scaled by map size
    min_energy_for_silicon=70,
)

GREEDY_SMART = Hyperparameters(
    strategy_type="greedy_opportunistic",
    exploration_phase_steps=50,  # Minimal exploration
    min_energy_for_silicon=70,
)

EFFICIENCY_FOCUSED = Hyperparameters(
    strategy_type="efficiency_learner",
    exploration_phase_steps=100,
    min_energy_for_silicon=75,  # More conservative for efficiency
)

EXPLORER_LOW_ENERGY = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=100,
    min_energy_for_silicon=60,  # More aggressive silicon gathering
)

EXPLORER_HIGH_ENERGY = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=100,
    min_energy_for_silicon=85,  # Very conservative silicon gathering
)

# =============================================================================
# Preset Dictionary (for easy access)
# =============================================================================

HYPERPARAMETER_PRESETS = {
    "explorer": EXPLORER,
    "greedy": GREEDY_SMART,
    "efficiency": EFFICIENCY_FOCUSED,
    "explorer_aggressive": EXPLORER_LOW_ENERGY,
    "explorer_conservative": EXPLORER_HIGH_ENERGY,
}
