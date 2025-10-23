import math
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from cogames.cogs_vs_clips.procedural import make_machina_procedural_map_builder
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from mettagrid.config.mettagrid_config import ChestConfig, GridObjectConfig, MettaGridConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig


def get_map(site: str) -> MapBuilderConfig:
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / site
    return MapBuilderConfig.from_uri(str(map_path))


PROCEDURAL_BASE_BUILDER = make_machina_procedural_map_builder(num_cogs=4)


_EXTRACTOR_ORDER = [
    "chest",
    "charger",
    "carbon_extractor",
    "oxygen_extractor",
    "germanium_extractor",
    "silicon_extractor",
]


def _normalize_weights(raw_weights: list[float]) -> dict[str, float]:
    if len(raw_weights) != len(_EXTRACTOR_ORDER):
        raise ValueError("Extractor weight vector must match extractor order length")

    positive = [max(0.0, float(weight)) for weight in raw_weights]
    total = sum(positive)
    if total <= 0:
        raise ValueError("Extractor weights must contain at least one positive value")

    return {name: value / total for name, value in zip(_EXTRACTOR_ORDER, positive, strict=False)}


def _uniform_weights() -> dict[str, float]:
    return _normalize_weights([1.0] * len(_EXTRACTOR_ORDER))


def _bernoulli_weights(p: float = 0.6) -> dict[str, float]:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return _normalize_weights(
        [
            1.0 - p,
            p,
            1.0 - p,
            p,
            1.0 - p,
            p,
        ]
    )


def _binomial_weights(n: int = 5, p: float = 0.5) -> dict[str, float]:
    weights = [math.comb(n, k) * (p**k) * ((1 - p) ** (n - k)) for k in range(len(_EXTRACTOR_ORDER))]
    return _normalize_weights(weights)


def _geometric_weights(p: float = 0.4) -> dict[str, float]:
    weights = [(1 - p) ** k * p for k in range(len(_EXTRACTOR_ORDER))]
    return _normalize_weights(weights)


def _hypergeometric_weights(population: int = 30, successes: int = 12, draws: int = 6) -> dict[str, float]:
    denominator = math.comb(population, draws)
    weights = []
    for k in range(len(_EXTRACTOR_ORDER)):
        if k > successes or k > draws:
            weights.append(0.0)
            continue
        weights.append(math.comb(successes, k) * math.comb(population - successes, draws - k) / denominator)
    return _normalize_weights(weights)


def _poisson_weights(lam: float = 2.5) -> dict[str, float]:
    weights = [math.exp(-lam) * (lam**k) / math.factorial(k) for k in range(len(_EXTRACTOR_ORDER))]
    return _normalize_weights(weights)


def _exponential_weights(rate: float = 1.0) -> dict[str, float]:
    sample_points = [float(idx) for idx in range(len(_EXTRACTOR_ORDER))]
    weights = [rate * math.exp(-rate * x) for x in sample_points]
    return _normalize_weights(weights)


def _sample_pdf_weights(pdf: Callable[[float], float], points: list[float]) -> dict[str, float]:
    weights = [max(0.0, float(pdf(point))) for point in points]
    return _normalize_weights(weights)


def _standard_normal_weights() -> dict[str, float]:
    points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
    return _sample_pdf_weights(
        lambda x: (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x),
        points,
    )


def _general_normal_weights(mu: float = 1.5, sigma: float = 0.7) -> dict[str, float]:
    points = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    return _sample_pdf_weights(
        lambda x: (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma)),
        points,
    )


def _lognormal_weights(mu: float = 0.0, sigma: float = 0.6) -> dict[str, float]:
    def _pdf(x: float) -> float:
        if x <= 0:
            return 0.0
        return (1.0 / (x * sigma * math.sqrt(2.0 * math.pi))) * math.exp(
            -((math.log(x) - mu) ** 2) / (2.0 * sigma * sigma)
        )

    points = [0.5, 0.9, 1.3, 1.9, 2.7, 3.5]
    return _sample_pdf_weights(_pdf, points)


def _gamma_weights(shape: float = 2.0, scale: float = 1.0) -> dict[str, float]:
    def _pdf(x: float) -> float:
        if x < 0:
            return 0.0
        return (x ** (shape - 1)) * math.exp(-x / scale) / (math.gamma(shape) * (scale**shape))

    points = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5]
    return _sample_pdf_weights(_pdf, points)


def _beta_weights(alpha: float = 2.0, beta: float = 5.0) -> dict[str, float]:
    beta_fn = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

    def _pdf(x: float) -> float:
        if not (0 < x < 1):
            return 0.0
        return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / beta_fn

    points = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    return _sample_pdf_weights(_pdf, points)


_EXTRACTOR_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "discrete_uniform": _uniform_weights(),
    "bernoulli": _bernoulli_weights(),
    "binomial": _binomial_weights(),
    "geometric": _geometric_weights(),
    "hypergeometric": _hypergeometric_weights(),
    "poisson": _poisson_weights(),
    "continuous_uniform": _uniform_weights(),
    "exponential": _exponential_weights(),
    "standard_normal": _standard_normal_weights(),
    "general_normal": _general_normal_weights(),
    "lognormal": _lognormal_weights(),
    "gamma": _gamma_weights(),
    "beta": _beta_weights(),
}


class ExtractorDistributionVariant(MissionVariant):
    distribution_key: str
    extractor_coverage: float = 0.012

    def apply(self, mission: Mission) -> Mission:
        if not hasattr(mission, "procedural_overrides"):
            return mission

        weights = _EXTRACTOR_DISTRIBUTIONS.get(self.distribution_key)
        if weights is None:
            raise ValueError(f"Unknown extractor distribution '{self.distribution_key}'")

        mission.procedural_overrides["extractors"] = dict(weights)
        mission.procedural_overrides["extractor_coverage"] = self.extractor_coverage
        return mission


def _make_extractor_variant_class(
    key: str,
    *,
    description: str,
    coverage: float,
) -> type[ExtractorDistributionVariant]:
    class_name = "".join(part.capitalize() for part in key.split("_")) + "ExtractorVariant"

    class _Variant(ExtractorDistributionVariant):  # type: ignore[misc]
        name = f"extractors_{key}"
        description = description
        distribution_key = key
        extractor_coverage = coverage

    _Variant.__name__ = class_name
    return _Variant


_EXTRACTOR_VARIANT_DEFINITIONS: dict[str, tuple[str, float]] = {
    "discrete_uniform": ("Uniform extractor distribution across all station types.", 0.012),
    "bernoulli": ("Alternating high/low probability pockets inspired by Bernoulli trials.", 0.012),
    "binomial": ("Peaked distribution centred on mid-tier extractors.", 0.014),
    "geometric": ("Front-loaded extractors with rapidly decreasing frequency.", 0.012),
    "hypergeometric": ("Weighted distribution modelling draws without replacement.", 0.013),
    "poisson": ("Skewed towards sparse pockets with occasional dense clusters.", 0.012),
    "continuous_uniform": ("Even spread leveraging continuous uniform sampling.", 0.012),
    "exponential": ("Clusters near spawn with exponential falloff across extractors.", 0.011),
    "standard_normal": ("Bell-curve concentration around central extractor types.", 0.013),
    "general_normal": ("Shifted Gaussian emphasising mid-to-heavy extractor mix.", 0.013),
    "lognormal": ("Log-normal bias favouring higher index extractors.", 0.011),
    "gamma": ("Gamma-distributed pockets providing a long-tailed density profile.", 0.012),
    "beta": ("Beta-distributed mix favouring early extractor types with tapered tail.", 0.012),
}


EXTRACTOR_VARIANTS: list[type[MissionVariant]] = [
    _make_extractor_variant_class(key, description=text, coverage=cov)
    for key, (text, cov) in _EXTRACTOR_VARIANT_DEFINITIONS.items()
]


class MinedOutVariant(MissionVariant):
    name: str = "mined_out"
    description: str = "Some resources are depleted. You must be efficient to survive."

    def apply(self, mission: Mission) -> Mission:
        mission.carbon_extractor.efficiency -= 50
        mission.oxygen_extractor.efficiency -= 50
        mission.germanium_extractor.efficiency -= 50
        mission.silicon_extractor.efficiency -= 50
        return mission


class DarkSideVariant(MissionVariant):
    name: str = "dark_side"
    description: str = "You're on the dark side of the asteroid. You recharge slower."

    def apply(self, mission: Mission) -> Mission:
        mission.energy_regen_amount = 0
        return mission


class BrightSideVariant(MissionVariant):
    name: str = "super_charged"
    description: str = "The sun is shining on you. You recharge faster."

    def apply(self, mission: Mission) -> Mission:
        mission.energy_regen_amount += 2
        return mission


class RoughTerrainVariant(MissionVariant):
    name: str = "rough_terrain"
    description: str = "The terrain is rough. Moving is more energy intensive."

    def apply(self, mission: Mission) -> Mission:
        mission.move_energy_cost += 2
        return mission


class SolarFlareVariant(MissionVariant):
    name: str = "solar_flare"
    description: str = "Chargers have been damaged by the solar flare."

    def apply(self, mission: Mission) -> Mission:
        mission.charger.efficiency -= 50
        return mission


class ExtractorDistributionVariant(MissionVariant):
    distribution_key: str
    extractor_coverage: float = 0.01

    def apply(self, mission: Mission) -> Mission:
        if not hasattr(mission, "procedural_overrides"):
            return mission

        weights = _EXTRACTOR_DISTRIBUTIONS.get(self.distribution_key)
        if weights is None:
            raise ValueError(f"Unknown extractor distribution '{self.distribution_key}'")

        mission.procedural_overrides["extractors"] = weights
        mission.procedural_overrides["extractor_coverage"] = self.extractor_coverage
        return mission


# Biome variants (weather) for procedural maps
class DesertBiomeVariant(MissionVariant):
    name: str = "desert"
    description: str = "The desert sands make navigation challenging."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides["biome_weights"] = {"desert": 1.0, "caves": 0.0, "forest": 0.0, "city": 0.0}
            mission.procedural_overrides["base_biome"] = "desert"
        return mission


class ForestBiomeVariant(MissionVariant):
    name: str = "forest"
    description: str = "Dense forests obscure your view."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides["biome_weights"] = {"forest": 1.0, "caves": 0.0, "desert": 0.0, "city": 0.0}
            mission.procedural_overrides["base_biome"] = "forest"
        return mission


class CityBiomeVariant(MissionVariant):
    name: str = "city"
    description: str = "Ancient city ruins provide structured pathways."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides.update(
                {
                    "base_biome": "city",
                    "biome_weights": {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0},
                    # Fill almost the entire map with the city layer
                    "density_scale": 1.0,
                    "biome_count": 1,
                    "max_biome_zone_fraction": 0.95,
                    # Disable dungeon overlays so they don't overwrite the grid
                    "dungeon_weights": {"bsp": 0.0, "maze": 0.0, "radial": 0.0},
                    "max_dungeon_zone_fraction": 0.0,
                    # Tighten the city grid itself
                }
            )
        return mission


class CavesBiomeVariant(MissionVariant):
    name: str = "caves"
    description: str = "Winding cave systems create a natural maze."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides["biome_weights"] = {"caves": 1.0, "desert": 0.0, "forest": 0.0, "city": 0.0}
            mission.procedural_overrides["base_biome"] = "caves"
        return mission


VARIANTS = [
    MinedOutVariant,
    DarkSideVariant,
    BrightSideVariant,
    RoughTerrainVariant,
    SolarFlareVariant,
    *EXTRACTOR_VARIANTS,
    DesertBiomeVariant,
    ForestBiomeVariant,
    CityBiomeVariant,
    CavesBiomeVariant,
]


# Define Sites
TRAINING_FACILITY = Site(
    name="training_facility",
    description="COG Training Facility. Basic training facility with open spaces and no obstacles.",
    map_builder=get_map("training_facility_open_1.map"),
    min_cogs=1,
    max_cogs=4,
)

HELLO_WORLD = Site(
    name="hello_world",
    description="Welcome to space..",
    map_builder=get_map("machina_100_stations.map"),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = Site(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=get_map("machina_200_stations.map"),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_PROCEDURAL = Site(
    name="machina_procedural",
    description="Procedurally generated asteroid arena with sanctum hub and resource pockets.",
    map_builder=PROCEDURAL_BASE_BUILDER,
    min_cogs=1,
    max_cogs=20,
)

SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    MACHINA_PROCEDURAL,
]


# Training Facility Missions
class HarvestMission(Mission):
    name: str = "harvest"
    description: str = "Collect resources and store them in the communal chest. Make sure to stay charged!"
    site: Site = TRAINING_FACILITY

    def configure(self):
        pass


class AssembleMission(Mission):
    name: str = "assemble"
    description: str = "Make HEARTs by using the assembler. Coordinate your team to maximize efficiency."
    site: Site = TRAINING_FACILITY


class VibeCheckMission(Mission):
    name: str = "vibe_check"
    description: str = "Modulate the group vibe to assemble HEARTs and Gear."
    site: Site = TRAINING_FACILITY


class RepairMission(Mission):
    name: str = "repair"
    description: str = "Repair disabled stations to restore their functionality."
    site: Site = TRAINING_FACILITY


class SignsAndPortentsMission(Mission):
    name: str = "signs_and_portents"
    description: str = "Interpret the signs and portents to discover new assembler protocols."
    site: Site = TRAINING_FACILITY


# Hello World Missions
class ExploreMission(Mission):
    name: str = "explore"
    description: str = "There are HEARTs scattered around the map. Collect them all."
    site: Site = HELLO_WORLD


class TreasureHuntMission(Mission):
    name: str = "treasure_hunt"
    description: str = (
        "The solar flare is making the germanium extractors really fiddly. "
        "A team of 4 is required to harvest germanium."
    )
    site: Site = HELLO_WORLD


class HelloWorldOpenWorldMission(Mission):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."
    site: Site = HELLO_WORLD


# Machina 1 Missions
class Machina1OpenWorldMission(Mission):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."
    site: Site = MACHINA_1


# Base class for procedural missions
class ProceduralMissionBase(Mission):
    site: Site = MACHINA_PROCEDURAL
    procedural_overrides: dict[str, Any] = Field(default_factory=dict)

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
    ) -> "Mission":
        # Use standard mission instantiation first (handles configure + variants)
        mission = super().instantiate(map_builder, num_cogs, variant)

        # Build procedural map using mission-specific overrides
        overrides = dict(mission.procedural_overrides)
        procedural_builder = make_machina_procedural_map_builder(num_cogs=num_cogs, **overrides)
        mission.map = procedural_builder

        return mission


# Procedural Missions
class MachinaProceduralExploreMission(ProceduralMissionBase):
    name: str = "explore"
    description: str = "There are HEARTs scattered around the map. Collect them all."

    def configure(self):
        # Mission defaults that don't depend on num_cogs
        self.heart_capacity = 99
        # Only chests for explore mission
        self.procedural_overrides = {
            "extractors": {"chest": 1.0, "charger": 1.0},
            "extractor_coverage": 0.004,
        }

    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        # Reward agents for hearts they personally hold
        if self.num_cogs and self.num_cogs > 0:
            reward_weight = 1.0 / float(self.num_cogs)
        else:
            reward_weight = 1.0 / float(max(1, getattr(env.game, "num_agents", 1)))
        env.game.agent.rewards.inventory = {"heart": reward_weight}
        env.game.agent.rewards.stats = {}
        env.game.agent.rewards.inventory_max = {}
        env.game.agent.rewards.stats_max = {}

        # Ensure every chest template starts with one heart
        chest_cfg = env.game.objects.get("chest")
        if isinstance(chest_cfg, ChestConfig):
            chest_cfg.initial_inventory = 1
        return env


class ProceduralOpenWorldMission(ProceduralMissionBase):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."


MISSIONS = [
    HarvestMission,
    AssembleMission,
    VibeCheckMission,
    RepairMission,
    SignsAndPortentsMission,
    ExploreMission,
    TreasureHuntMission,
    HelloWorldOpenWorldMission,
    Machina1OpenWorldMission,
    MachinaProceduralExploreMission,
    ProceduralOpenWorldMission,
]


def _get_default_map_objects() -> dict[str, GridObjectConfig]:
    """Get default map objects for cogs vs clips missions."""
    carbon_extractor = CarbonExtractorConfig()
    oxygen_extractor = OxygenExtractorConfig()
    germanium_extractor = GermaniumExtractorConfig()
    silicon_extractor = SiliconExtractorConfig()
    charger = ChargerConfig()
    chest = CvCChestConfig()
    wall = CvCWallConfig()
    assembler = CvCAssemblerConfig()

    return {
        "carbon_extractor": carbon_extractor.station_cfg(),
        "oxygen_extractor": oxygen_extractor.station_cfg(),
        "germanium_extractor": germanium_extractor.station_cfg(),
        "silicon_extractor": silicon_extractor.station_cfg(),
        "charger": charger.station_cfg(),
        "chest": chest.station_cfg(),
        "wall": wall.station_cfg(),
        "assembler": assembler.station_cfg(),
    }


def make_game(num_cogs: int = 2, map_name: str = "training_facility_open_1.map") -> MettaGridConfig:
    """Create a default cogs vs clips game configuration."""
    mission = HarvestMission()
    map_builder = get_map(map_name)
    # Use no variant (default)
    variant = MissionVariant(name="default", description="Default mission variant")
    return mission.instantiate(map_builder, num_cogs, variant).make_env()
