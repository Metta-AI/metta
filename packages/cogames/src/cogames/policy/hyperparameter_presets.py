"""
Hyperparameter Presets for Scripted Agent

This module defines diverse hyperparameter configurations that produce different
exploration and resource management behaviors. Each preset is optimized for
different difficulty levels and mission types.

The goal is to have agents with distinct "personalities":
- Conservative: Careful energy management, thorough exploration
- Aggressive: Fast exploration, risk-taking
- Efficient: Optimize for speed, minimal waste
- Adaptive: Balance between all strategies
"""

from cogames.policy.scripted_agent import Hyperparameters

# =============================================================================
# Core Presets (General Purpose)
# =============================================================================

CONSERVATIVE = Hyperparameters(
    # Exploration: Thorough, systematic
    exploration_strategy="frontier",  # Systematic frontier-based exploration
    levy_alpha=1.5,
    exploration_radius=50,  # Wider exploration
    # Energy: Very cautious
    energy_buffer=30,  # Large safety margin
    min_energy_for_silicon=80,  # Wait for high energy before silicon
    charger_search_threshold=50,  # Start looking for chargers early
    # Resources: Patient, wait for cooldowns
    prefer_nearby=True,
    cooldown_tolerance=30,  # Willing to wait longer
    depletion_threshold=0.3,  # Start looking for alternatives early (30% left)
    # Efficiency: Track and optimize
    track_efficiency=True,
    efficiency_weight=0.4,  # Favor efficient extractors
    # Pathfinding: Use best algorithms
    use_astar=True,
    astar_threshold=15,  # Use A* more often
    # Cooldown: Patient waiting
    enable_cooldown_waiting=True,
    max_cooldown_wait=150,  # Very patient
    # Exploration: Center-biased
    prioritize_center=True,
    center_bias_weight=0.6,  # Strong center bias
    max_wait_turns=75,  # Patient
)

AGGRESSIVE = Hyperparameters(
    # Exploration: Fast, wide-ranging
    exploration_strategy="levy",  # Lévy flights for rapid coverage
    levy_alpha=1.2,  # Lower alpha = more long jumps
    exploration_radius=60,  # Very wide exploration
    # Energy: Risk-taking
    energy_buffer=10,  # Minimal safety margin
    min_energy_for_silicon=60,  # Go for silicon earlier
    charger_search_threshold=30,  # Wait longer before charging
    # Resources: Impatient, move on quickly
    prefer_nearby=False,  # Don't favor nearby, explore widely
    cooldown_tolerance=10,  # Don't wait long for cooldowns
    depletion_threshold=0.1,  # Only look for alternatives when nearly depleted
    # Efficiency: Don't track, just go
    track_efficiency=False,
    efficiency_weight=0.1,  # Don't care much about efficiency
    # Pathfinding: Fast greedy approaches
    use_astar=False,  # Use greedy/BFS only
    astar_threshold=30,
    # Cooldown: Don't wait
    enable_cooldown_waiting=False,
    max_cooldown_wait=20,  # Very impatient
    # Exploration: No center bias, explore everywhere
    prioritize_center=False,
    center_bias_weight=0.2,  # Weak center bias
    max_wait_turns=25,  # Impatient
)

EFFICIENT = Hyperparameters(
    # Exploration: Mixed strategy for balance
    exploration_strategy="mixed",  # Combine frontier and Lévy
    levy_alpha=1.5,
    exploration_radius=45,
    # Energy: Balanced
    energy_buffer=15,  # Reasonable safety margin
    min_energy_for_silicon=70,
    charger_search_threshold=40,
    # Resources: Optimize for efficiency
    prefer_nearby=True,  # Minimize travel
    cooldown_tolerance=20,  # Moderate waiting
    depletion_threshold=0.25,  # Plan ahead
    # Efficiency: Heavily optimize
    track_efficiency=True,
    efficiency_weight=0.6,  # Strongly favor efficient extractors
    # Pathfinding: Optimal algorithms
    use_astar=True,
    astar_threshold=20,
    # Cooldown: Strategic waiting
    enable_cooldown_waiting=True,
    max_cooldown_wait=100,
    # Exploration: Moderate center bias
    prioritize_center=True,
    center_bias_weight=0.4,
    max_wait_turns=50,
)

ADAPTIVE = Hyperparameters(
    # Exploration: Balanced approach
    exploration_strategy="mixed",
    levy_alpha=1.5,
    exploration_radius=45,
    # Energy: Balanced
    energy_buffer=20,
    min_energy_for_silicon=70,
    charger_search_threshold=40,
    # Resources: Balanced
    prefer_nearby=True,
    cooldown_tolerance=20,
    depletion_threshold=0.2,
    # Efficiency: Track and use
    track_efficiency=True,
    efficiency_weight=0.3,
    # Pathfinding: Balanced
    use_astar=True,
    astar_threshold=20,
    # Cooldown: Moderate
    enable_cooldown_waiting=True,
    max_cooldown_wait=100,
    # Exploration: Moderate center bias
    prioritize_center=True,
    center_bias_weight=0.5,
    max_wait_turns=50,
)


# =============================================================================
# Difficulty-Specific Presets
# =============================================================================

EASY_MODE = Hyperparameters(
    # For easy difficulties: Can be more aggressive since resources are abundant
    exploration_strategy="levy",  # Fast exploration
    levy_alpha=1.3,
    exploration_radius=50,
    energy_buffer=15,
    min_energy_for_silicon=65,
    charger_search_threshold=35,
    prefer_nearby=False,  # Explore widely
    cooldown_tolerance=15,
    depletion_threshold=0.15,  # Don't worry about depletion much
    track_efficiency=False,  # Don't need to optimize
    efficiency_weight=0.2,
    use_astar=True,
    astar_threshold=25,
    enable_cooldown_waiting=False,  # Don't wait, just find another
    max_cooldown_wait=50,
    prioritize_center=False,
    center_bias_weight=0.3,
    max_wait_turns=30,
)

HARD_MODE = Hyperparameters(
    # For hard difficulties: Must be very careful and strategic
    exploration_strategy="frontier",  # Systematic exploration
    levy_alpha=1.6,
    exploration_radius=55,  # Explore widely to find all extractors
    energy_buffer=25,  # Large safety margin
    min_energy_for_silicon=75,
    charger_search_threshold=45,
    prefer_nearby=True,  # Minimize energy waste
    cooldown_tolerance=25,  # Willing to wait
    depletion_threshold=0.6,  # Start looking EARLY! (was 0.35, gave agent 200-300 step head start)
    track_efficiency=True,
    efficiency_weight=0.5,  # Efficiency matters a lot
    use_astar=True,
    astar_threshold=15,
    enable_cooldown_waiting=True,
    max_cooldown_wait=50,  # Don't waste time waiting (was 150)
    prioritize_center=True,
    center_bias_weight=0.7,  # Strong center bias to find stations
    max_wait_turns=75,  # Less patient (was 100)
)

EXTREME_MODE = Hyperparameters(
    # For extreme difficulties: Perfect play required
    exploration_strategy="mixed",  # Use all strategies
    levy_alpha=1.7,  # More local search
    exploration_radius=60,  # Must find EVERYTHING
    energy_buffer=30,  # Maximum safety
    min_energy_for_silicon=80,
    charger_search_threshold=50,
    prefer_nearby=True,
    cooldown_tolerance=20,  # Don't wait long (was 40)
    depletion_threshold=0.75,  # Start looking VERY early! (was 0.5, critical on EXTREME)
    track_efficiency=True,
    efficiency_weight=0.7,  # Efficiency is critical
    use_astar=True,
    astar_threshold=10,  # Use A* almost always
    enable_cooldown_waiting=True,
    max_cooldown_wait=30,  # Don't waste precious time (was 200)
    prioritize_center=True,
    center_bias_weight=0.8,  # Very strong center bias
    max_wait_turns=50,  # Move on quickly (was 150)
)

SPEED_RUN_MODE = Hyperparameters(
    # For speed_run difficulty: High efficiency, low uses - optimize routing
    exploration_strategy="frontier",  # Systematic to find all quickly
    levy_alpha=1.4,
    exploration_radius=50,
    energy_buffer=10,  # Don't need much buffer with high regen
    min_energy_for_silicon=65,
    charger_search_threshold=35,
    prefer_nearby=True,  # Minimize travel time
    cooldown_tolerance=5,  # Don't wait, move on
    depletion_threshold=0.4,  # Plan ahead since uses are limited
    track_efficiency=True,
    efficiency_weight=0.8,  # Efficiency is everything
    use_astar=True,
    astar_threshold=15,
    enable_cooldown_waiting=False,  # Never wait, just find another
    max_cooldown_wait=20,
    prioritize_center=True,
    center_bias_weight=0.6,
    max_wait_turns=30,
)

ENERGY_CRISIS_MODE = Hyperparameters(
    # For energy_crisis difficulty: Zero regen - every move counts
    exploration_strategy="frontier",  # Systematic, no wasted moves
    levy_alpha=1.8,  # Very local search
    exploration_radius=40,  # Don't explore too far
    energy_buffer=35,  # HUGE safety margin
    min_energy_for_silicon=85,  # Need lots of energy for silicon
    charger_search_threshold=60,  # Start looking early
    prefer_nearby=True,  # Absolutely minimize travel
    cooldown_tolerance=50,  # Very willing to wait to save energy
    depletion_threshold=0.4,  # Plan ahead
    track_efficiency=True,
    efficiency_weight=0.5,
    use_astar=True,
    astar_threshold=10,  # Always use optimal paths
    enable_cooldown_waiting=True,
    max_cooldown_wait=200,  # Waiting is better than moving
    prioritize_center=True,
    center_bias_weight=0.7,
    max_wait_turns=200,  # Very patient
)


# =============================================================================
# Specialized Presets (For specific mission types)
# =============================================================================

OXYGEN_HUNTER = Hyperparameters(
    # Optimized for missions with oxygen bottlenecks
    exploration_strategy="mixed",
    levy_alpha=1.5,
    exploration_radius=50,
    energy_buffer=20,
    min_energy_for_silicon=70,
    charger_search_threshold=40,
    prefer_nearby=False,  # Explore widely to find all oxygen sources
    cooldown_tolerance=30,  # Willing to wait for oxygen
    depletion_threshold=0.3,
    track_efficiency=True,
    efficiency_weight=0.5,  # Efficiency matters for oxygen
    use_astar=True,
    astar_threshold=20,
    enable_cooldown_waiting=True,
    max_cooldown_wait=150,  # Very patient for oxygen
    prioritize_center=True,
    center_bias_weight=0.5,
    max_wait_turns=75,
)

GERMANIUM_FOCUSED = Hyperparameters(
    # Optimized for missions with germanium scarcity
    exploration_strategy="frontier",  # Find all germanium sources
    levy_alpha=1.6,
    exploration_radius=55,
    energy_buffer=25,
    min_energy_for_silicon=75,
    charger_search_threshold=45,
    prefer_nearby=True,
    cooldown_tolerance=20,
    depletion_threshold=0.4,  # Very careful with germanium
    track_efficiency=True,
    efficiency_weight=0.6,
    use_astar=True,
    astar_threshold=15,
    enable_cooldown_waiting=True,
    max_cooldown_wait=120,
    prioritize_center=True,
    center_bias_weight=0.6,
    max_wait_turns=75,
)


# =============================================================================
# Preset Registry
# =============================================================================

HYPERPARAMETER_PRESETS: dict[str, Hyperparameters] = {
    # Core presets
    "conservative": CONSERVATIVE,
    "aggressive": AGGRESSIVE,
    "efficient": EFFICIENT,
    "adaptive": ADAPTIVE,
    # Difficulty-specific
    "easy_mode": EASY_MODE,
    "hard_mode": HARD_MODE,
    "extreme_mode": EXTREME_MODE,
    "speed_run_mode": SPEED_RUN_MODE,
    "energy_crisis_mode": ENERGY_CRISIS_MODE,
    # Specialized
    "oxygen_hunter": OXYGEN_HUNTER,
    "germanium_focused": GERMANIUM_FOCUSED,
}


def get_preset(name: str) -> Hyperparameters:
    """Get a hyperparameter preset by name.

    Args:
        name: Preset name (e.g., "conservative", "aggressive")

    Returns:
        Hyperparameters instance

    Raises:
        KeyError: If preset name not found
    """
    if name not in HYPERPARAMETER_PRESETS:
        available = ", ".join(HYPERPARAMETER_PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return HYPERPARAMETER_PRESETS[name]


def get_recommended_preset_for_difficulty(difficulty: str) -> str:
    """Get recommended hyperparameter preset for a difficulty level.

    Args:
        difficulty: Difficulty name (e.g., "easy", "hard", "extreme")

    Returns:
        Recommended preset name
    """
    recommendations = {
        "easy": "easy_mode",
        "medium": "adaptive",
        "hard": "hard_mode",
        "extreme": "extreme_mode",
        "single_use": "extreme_mode",
        "speed_run": "speed_run_mode",
        "energy_crisis": "energy_crisis_mode",
    }
    return recommendations.get(difficulty, "adaptive")


def list_presets() -> None:
    """Print all available hyperparameter presets."""
    print("\nAvailable Hyperparameter Presets")
    print("=" * 80)

    print("\n## Core Presets (General Purpose)")
    for name in ["conservative", "aggressive", "efficient", "adaptive"]:
        preset = HYPERPARAMETER_PRESETS[name]
        print(f"\n{name.upper()}:")
        print(f"  Exploration: {preset.exploration_strategy}, radius={preset.exploration_radius}")
        print(f"  Energy: buffer={preset.energy_buffer}, min_silicon={preset.min_energy_for_silicon}")
        print(f"  Cooldown: tolerance={preset.cooldown_tolerance}, max_wait={preset.max_cooldown_wait}")
        print(f"  Efficiency: track={preset.track_efficiency}, weight={preset.efficiency_weight}")

    print("\n## Difficulty-Specific Presets")
    for name in ["easy_mode", "hard_mode", "extreme_mode", "speed_run_mode", "energy_crisis_mode"]:
        preset = HYPERPARAMETER_PRESETS[name]
        print(f"\n{name.upper()}:")
        print(f"  Exploration: {preset.exploration_strategy}, radius={preset.exploration_radius}")
        print(f"  Energy: buffer={preset.energy_buffer}, threshold={preset.charger_search_threshold}")
        print(f"  Depletion: threshold={preset.depletion_threshold}")

    print("\n## Specialized Presets")
    for name in ["oxygen_hunter", "germanium_focused"]:
        preset = HYPERPARAMETER_PRESETS[name]
        print(f"\n{name.upper()}:")
        print(f"  Exploration: {preset.exploration_strategy}")
        print(f"  Cooldown: max_wait={preset.max_cooldown_wait}")
        print(f"  Efficiency: weight={preset.efficiency_weight}")


if __name__ == "__main__":
    list_presets()
