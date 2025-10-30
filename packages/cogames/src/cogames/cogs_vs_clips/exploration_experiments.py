"""
Exploration Experiments for Scripted Agent Testing

This module defines 10 experiments designed to test different exploration strategies
for single-agent scenarios. Each experiment varies:
- Outside extractor distribution (C, O, G, S, chargers)
- Extractor efficiency (affects output per use)
- Extractor max_uses (depletion limits)
- Agent energy_regen_amount (passive energy recovery)

See experiments/scripted_agent_exploration_experiments.md for detailed rationale.
"""

from cogames.cogs_vs_clips.mission import Mission, Site
from cogames.cogs_vs_clips.missions import get_map
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)

# =============================================================================
# Experiment Sites
# =============================================================================

EXP1_SITE = Site(
    name="exp1",
    description="Baseline: Standard settings, sparse outside resources",
    map_builder=get_map("extractor_hub_30x30.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP2_SITE = Site(
    name="exp2",
    description="Oxygen Abundance: Breaking the cooldown bottleneck (13 oxygen sources)",
    map_builder=get_map("extractor_hub_80x80.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP3_SITE = Site(
    name="exp3",
    description="Low Efficiency: Energy management challenge",
    map_builder=get_map("extractor_hub_50x50.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP4_SITE = Site(
    name="exp4",
    description="Fast Depletion: Resource scarcity drives exploration",
    map_builder=get_map("extractor_hub_70x70.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP5_SITE = Site(
    name="exp5",
    description="Energy Abundance: High regeneration enables aggressive exploration",
    map_builder=get_map("extractor_hub_70x70.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP6_SITE = Site(
    name="exp6",
    description="Energy Scarcity: Charger network navigation (6 chargers)",
    map_builder=get_map("extractor_hub_50x50.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP7_SITE = Site(
    name="exp7",
    description="High Efficiency: Fast gathering enables more exploration",
    map_builder=get_map("extractor_hub_50x50.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP8_SITE = Site(
    name="exp8",
    description="Zoned Resources: Spatial clustering with distance tradeoffs",
    map_builder=get_map("extractor_hub_100x100.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP9_SITE = Site(
    name="exp9",
    description="Resource Abundance: No scarcity constraints",
    map_builder=get_map("extractor_hub_100x100.map"),
    min_cogs=1,
    max_cogs=1,
)

EXP10_SITE = Site(
    name="exp10",
    description="Complex Mixed: Multi-factor optimization",
    map_builder=get_map("extractor_hub_80x80.map"),
    min_cogs=1,
    max_cogs=1,
)

EXPLORATION_SITES = [
    EXP1_SITE,
    EXP2_SITE,
    EXP3_SITE,
    EXP4_SITE,
    EXP5_SITE,
    EXP6_SITE,
    EXP7_SITE,
    EXP8_SITE,
    EXP9_SITE,
    EXP10_SITE,
]

# =============================================================================
# Experiment Missions
# =============================================================================


class Experiment1Mission(Mission):
    """
    Experiment 1: Baseline Control

    Strategy: Wait-based exploration during oxygen cooldowns
    Outside: 1C, 1O, 1G, 1S, 1 charger @ 25 tiles
    """

    name: str = "baseline"
    description: str = "Baseline with standard settings and sparse outside resources"
    site: Site = EXP1_SITE

    # Standard efficiency (100%), standard max_uses (1000)
    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 1


class Experiment2Mission(Mission):
    """
    Experiment 2: Oxygen Abundance

    Strategy: Rotational harvesting across multiple oxygen sources
    Outside: 1C, 4O @ various distances, 1G, 1S, 1 charger
    """

    name: str = "oxygen_abundance"
    description: str = "Multiple oxygen sources to break the cooldown bottleneck"
    site: Site = EXP2_SITE

    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 1


class Experiment3Mission(Mission):
    """
    Experiment 3: Low Efficiency Challenge

    Strategy: Energy management and frequent charging
    Outside: 2C, 2O, 2G, 2S, 2 chargers @ 20, 30 tiles
    """

    name: str = "low_efficiency"
    description: str = "75% efficiency requires more harvests and careful energy management"
    site: Site = EXP3_SITE

    # 75% efficiency across all extractors
    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=75, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=75, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=75, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=75)

    energy_regen_amount: int = 1


class Experiment4Mission(Mission):
    """
    Experiment 4: Fast Depletion

    Strategy: Depletion-driven exploration to find new sources
    Outside: 3C, 3O, 3G, 3S, 1 charger @ 20 tiles
    """

    name: str = "fast_depletion"
    description: str = "Resources deplete quickly (max_uses=50), forcing exploration"
    site: Site = EXP4_SITE

    # Low max_uses causes fast depletion
    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=50)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=50)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=50)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=50)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 1


class Experiment5Mission(Mission):
    """
    Experiment 5: Energy Abundance

    Strategy: Aggressive exploration enabled by high energy regeneration
    Outside: 2C, 2O, 2G, 3S, 3 chargers @ 15, 25, 35 tiles
    """

    name: str = "energy_abundance"
    description: str = "Double energy regeneration (2/turn) removes energy constraint"
    site: Site = EXP5_SITE

    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 2  # Double regeneration!


class Experiment6Mission(Mission):
    """
    Experiment 6: Energy Scarcity

    Strategy: Charger network navigation, infrastructure-dependent pathfinding
    Outside: 2C, 2O, 2G, 2S, 5 chargers @ 5, 15, 20, 30, 40 tiles
    """

    name: str = "energy_scarcity"
    description: str = "Minimal energy regeneration (1/turn) requires frequent charging"
    site: Site = EXP6_SITE

    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 1  # Minimal regeneration (was 0.5, but must be int)


class Experiment7Mission(Mission):
    """
    Experiment 7: High Efficiency

    Strategy: Time-efficient gathering, maximize exploration time
    Outside: 1C, 1O, 2G, 2S, 1 charger @ 25 tiles
    """

    name: str = "high_efficiency"
    description: str = "Double efficiency (200%) means faster gathering and more exploration time"
    site: Site = EXP7_SITE

    # 200% efficiency across all extractors
    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=200, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=200, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=2, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=200, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=200)

    energy_regen_amount: int = 1


class Experiment8Mission(Mission):
    """
    Experiment 8: Zoned Resources

    Strategy: Zone-based batching, distance vs efficiency tradeoffs
    Outside: West zone (3C, 3O, 1+), East zone (3G, 3S, 1+), North zone (1 each @ 150% eff, 1+)

    Note: North zone uses high-efficiency extractor variants in the map
    """

    name: str = "zoned_resources"
    description: str = "Resources clustered by type in zones, north zone is farther but more efficient"
    site: Site = EXP8_SITE

    # Standard efficiency for most zones
    # North zone will use high-efficiency variants marked in map
    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 1


class Experiment9Mission(Mission):
    """
    Experiment 9: Resource Abundance

    Strategy: Nearest-available policy, no scarcity constraints
    Outside: 5C, 5O, 5G, 5S, 3 chargers @ 15, 25, 35 tiles
    """

    name: str = "resource_abundance"
    description: str = "Abundant resources (5 of each) eliminate scarcity bottlenecks"
    site: Site = EXP9_SITE

    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=100, max_uses=1000)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=100, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=1, max_uses=1000)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=100, max_uses=1000)
    charger: ChargerConfig = ChargerConfig(efficiency=100)

    energy_regen_amount: int = 1


class Experiment10Mission(Mission):
    """
    Experiment 10: Complex Mixed Optimization

    Strategy: Strategic prioritization based on multiple factors
    Outside: 2C, 2O, 3G, 3S, 2 chargers @ 20, 30 tiles

    Mixed efficiency and max_uses create complex optimization problem:
    - Carbon: 150% efficiency, 500 max_uses
    - Oxygen: 75% efficiency, 1000 max_uses (133 turn cooldown)
    - Germanium: 200% efficiency, 300 max_uses
    - Silicon: 125% efficiency, 800 max_uses
    - Charger: 75% efficiency
    - Energy regen: 2/turn
    """

    name: str = "complex_mixed"
    description: str = "Mixed efficiency and depletion rates require strategic prioritization"
    site: Site = EXP10_SITE

    # Mixed parameters create complex optimization
    carbon_extractor: CarbonExtractorConfig = CarbonExtractorConfig(efficiency=150, max_uses=500)
    oxygen_extractor: OxygenExtractorConfig = OxygenExtractorConfig(efficiency=75, max_uses=1000)
    germanium_extractor: GermaniumExtractorConfig = GermaniumExtractorConfig(efficiency=2, max_uses=300)
    silicon_extractor: SiliconExtractorConfig = SiliconExtractorConfig(efficiency=125, max_uses=800)
    charger: ChargerConfig = ChargerConfig(efficiency=75)

    energy_regen_amount: int = 2


# =============================================================================
# Experiment Registry
# =============================================================================

EXPLORATION_MISSIONS = [
    Experiment1Mission,
    Experiment2Mission,
    Experiment3Mission,
    Experiment4Mission,
    Experiment5Mission,
    Experiment6Mission,
    Experiment7Mission,
    Experiment8Mission,
    Experiment9Mission,
    Experiment10Mission,
]


# =============================================================================
# Utility Functions
# =============================================================================


def get_experiment_mission(experiment_number: int) -> type[Mission]:
    """Get experiment mission class by number (1-10).

    Args:
        experiment_number: Experiment number from 1 to 10

    Returns:
        Mission class for the experiment

    Raises:
        ValueError: If experiment number is out of range
    """
    if not 1 <= experiment_number <= 10:
        raise ValueError(f"Experiment number must be 1-10, got {experiment_number}")
    return EXPLORATION_MISSIONS[experiment_number - 1]


def list_experiments() -> None:
    """Print a summary of all exploration experiments."""
    print("\nExploration Experiments for Scripted Agent Testing")
    print("=" * 80)
    for i, mission_class in enumerate(EXPLORATION_MISSIONS, 1):
        mission = mission_class()
        print(f"\nExperiment {i}: {mission.site.name}.{mission.name}")
        print(f"  {mission.site.description}")
        print(f"  Energy regen: {mission.energy_regen_amount}/turn")
        print(
            f"  Efficiency: C={mission.carbon_extractor.efficiency}%, "
            f"O={mission.oxygen_extractor.efficiency}%, "
            f"G={mission.germanium_extractor.efficiency}, "
            f"S={mission.silicon_extractor.efficiency}%"
        )
        print(
            f"  Max uses: C={mission.carbon_extractor.max_uses}, "
            f"O={mission.oxygen_extractor.max_uses}, "
            f"G={mission.germanium_extractor.max_uses}, "
            f"S={mission.silicon_extractor.max_uses}"
        )


if __name__ == "__main__":
    list_experiments()
