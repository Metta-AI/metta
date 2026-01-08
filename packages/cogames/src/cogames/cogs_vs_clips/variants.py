from typing import Iterable, Sequence, override

from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_VARIANTS
from cogames.cogs_vs_clips.mission import MissionVariant
from cogames.cogs_vs_clips.procedural import BaseHubVariant, MachinaArenaVariant
from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    ProtocolConfig,
    ResourceLimitsConfig,
    VibeTransfer,
)
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import DEFAULT_EXTRACTORS as HUB_EXTRACTORS
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType


class MinedOutVariant(MissionVariant):
    name: str = "mined_out"
    description: str = "All resources are depleted. You must be efficient to survive."

    @override
    def modify_mission(self, mission):
        # Clamp efficiency to minimum of 50 to prevent negative values
        mission.carbon_extractor.max_uses = 2
        mission.oxygen_extractor.max_uses = 2
        mission.silicon_extractor.max_uses = 2


class DarkSideVariant(MissionVariant):
    name: str = "dark_side"
    description: str = "You're on the dark side of the asteroid. You recharge slower."

    @override
    def modify_mission(self, mission):
        mission.energy_regen_amount = 0


class LonelyHeartVariant(MissionVariant):
    name: str = "lonely_heart"
    description: str = "Making hearts for one agent is easy."

    @override
    def modify_mission(self, mission):
        mission.assembler.first_heart_cost = 1
        mission.assembler.additional_heart_cost = 0
        mission.heart_capacity = max(mission.heart_capacity, 255)

    @override
    def modify_env(self, mission, env):
        simplified_inputs = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1, "energy": 1}

        assembler = env.game.objects["assembler"]
        if not isinstance(assembler, AssemblerConfig):
            raise TypeError("Expected 'assembler' to be AssemblerConfig")

        for idx, proto in enumerate(assembler.protocols):
            if proto.output_resources.get("heart", 0) == 0:
                continue
            updated = proto.model_copy(deep=True)
            updated.input_resources = dict(simplified_inputs)
            assembler.protocols[idx] = updated

        germanium = env.game.objects["germanium_extractor"]
        if not isinstance(germanium, AssemblerConfig):
            raise TypeError("Expected 'germanium_extractor' to be AssemblerConfig")
        germanium.max_uses = 0
        updated_protocols: list[ProtocolConfig] = []
        for proto in germanium.protocols:
            new_proto = proto.model_copy(deep=True)
            output = dict(new_proto.output_resources)
            output["germanium"] = max(output.get("germanium", 0), 1)
            new_proto.output_resources = output
            new_proto.cooldown = max(new_proto.cooldown, 1)
            updated_protocols.append(new_proto)
        if updated_protocols:
            germanium.protocols = updated_protocols


class SuperChargedVariant(MissionVariant):
    name: str = "super_charged"
    description: str = "The sun is shining on you. You recharge faster."

    @override
    def modify_mission(self, mission):
        mission.energy_regen_amount += 2


class RoughTerrainVariant(MissionVariant):
    name: str = "rough_terrain"
    description: str = "The terrain is rough. Moving is more energy intensive."

    @override
    def modify_mission(self, mission):
        mission.move_energy_cost += 2


class SolarFlareVariant(MissionVariant):
    name: str = "solar_flare"
    description: str = "Chargers have been damaged by the solar flare."

    @override
    def modify_mission(self, mission):
        # Clamp efficiency to minimum of 1 to prevent negative values
        mission.charger.efficiency = max(1, mission.charger.efficiency - 50)


class TrainingVariant(MissionVariant):
    name: str = "training"
    description: str = "Training-friendly: max cargo, fast extractors, chest only deposits hearts."

    @override
    def modify_mission(self, mission):
        mission.cargo_capacity = 255  # Maximum cargo for easier resource collection

    @override
    def modify_env(self, mission, env):
        # Set all extractor cooldowns to 5ms (fast)
        for extractor_name in ["carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor"]:
            extractor = env.game.objects.get(extractor_name)
            if isinstance(extractor, AssemblerConfig):
                updated_protocols = []
                for proto in extractor.protocols:
                    updated_proto = proto.model_copy(deep=True)
                    updated_proto.cooldown = 5
                    updated_protocols.append(updated_proto)
                extractor.protocols = updated_protocols

        # Modify chest to only deposit hearts by default (not all resources)
        chest = env.game.objects.get("chest")
        if isinstance(chest, ChestConfig):
            chest.vibe_transfers = {
                "heart_b": {"heart": 1},
                "carbon_a": {"carbon": -10},
                "carbon_b": {"carbon": 10},
                "oxygen_a": {"oxygen": -10},
                "oxygen_b": {"oxygen": 10},
                "germanium_a": {"germanium": -1},
                "germanium_b": {"germanium": 1},
                "silicon_a": {"silicon": -25},
                "silicon_b": {"silicon": 25},
            }


class PackRatVariant(MissionVariant):
    name: str = "pack_rat"
    description: str = "Raise heart, cargo, energy, and gear caps to 255."

    @override
    def modify_mission(self, mission):
        mission.heart_capacity = max(mission.heart_capacity, 255)
        mission.energy_capacity = max(mission.energy_capacity, 255)
        mission.cargo_capacity = max(mission.cargo_capacity, 255)
        mission.gear_capacity = max(mission.gear_capacity, 255)


class EnergizedVariant(MissionVariant):
    name: str = "energized"
    description: str = "Max energy and full regen so agents never run dry."

    @override
    def modify_mission(self, mission):
        mission.energy_capacity = max(mission.energy_capacity, 255)
        mission.energy_regen_amount = mission.energy_capacity


class ResourceBottleneckVariant(MissionVariant):
    name: str = "resource_bottleneck"
    description: str = "A resource is the limiting factor. Agents must prioritize it over other resources."
    resource: Sequence[str] | str = ("oxygen", "germanium", "silicon", "carbon")

    @override
    def modify_mission(self, mission):
        # Accept either a single resource or an iterable of resources to bottleneck
        if isinstance(self.resource, str):
            resources: Iterable[str] = [self.resource]
        else:
            resources = list(self.resource)

        for resource in resources:
            if resource in {"carbon", "oxygen", "germanium", "silicon"}:
                extractor_attr = f"{resource}_extractor"
            elif resource == "energy":
                extractor_attr = "charger"
            else:
                raise ValueError(f"Unsupported resource for bottleneck: {resource}")

            extractor = getattr(mission, extractor_attr, None)
            if extractor is None:
                raise AttributeError(f"Mission has no extractor attribute '{extractor_attr}'")

            # Clamp efficiency to minimum of 1 to prevent negative values
            extractor.efficiency = max(1, int(extractor.efficiency) - 50)


class SingleToolUnclipVariant(MissionVariant):
    name: str = "single_tool_unclip"
    description: str = "Only one tool is available: the decoder."
    resource: str = "carbon"

    @override
    def modify_env(self, mission, env):
        # Restrict assembler to a single generic gear recipe: carbon -> decoder (no vibes required)
        # Since the protocol doesn't require vibes, agents won't need to change vibes
        assembler = env.game.objects.get("assembler")
        if isinstance(assembler, AssemblerConfig):
            assembler.protocols = [
                ProtocolConfig(vibes=[], input_resources={self.resource: 1}, output_resources={"decoder": 1})
            ]


class CompassVariant(MissionVariant):
    name: str = "compass"
    description: str = "Enable compass observation pointing toward the assembler."

    @override
    def modify_mission(self, mission):
        mission.compass_enabled = True


class HeartChorusVariant(MissionVariant):
    name: str = "heart_chorus"
    description: str = "Heart-centric reward shaping with gentle resource bonuses."

    @override
    def modify_env(self, mission, env):
        # Supplemental shaping: focus rewards on the acting agent for heart progress.
        rewards = dict(env.game.agent.rewards.stats)
        rewards.update(
            {
                "assembler.heart.created": 1.0,
                "chest.heart.deposited_by_agent": 1.0,
                "chest.heart.withdrawn_by_agent": -1.0,
                "inventory.diversity.ge.2": 0.17,
                "inventory.diversity.ge.3": 0.18,
                "inventory.diversity.ge.4": 0.60,
                "inventory.diversity.ge.5": 0.97,
            }
        )
        env.game.agent.rewards.stats = rewards


class TinyHeartProtocolsVariant(MissionVariant):
    """Prepend low-cost heart/red-heart assembler protocols for easy hearts."""

    name: str = "tiny_heart_protocols"
    description: str = "Prepend low-cost heart/red-heart assembler protocols."

    # Allow customization if ever needed; defaults match prior inline block.
    carbon_cost: int = 2
    oxygen_cost: int = 2
    germanium_cost: int = 1
    silicon_cost: int = 3
    energy_cost: int = 2

    @override
    def modify_env(self, mission, env) -> None:
        assembler = env.game.objects.get("assembler")
        if not isinstance(assembler, AssemblerConfig):
            raise TypeError("Expected 'assembler' to be AssemblerConfig")

        tiny_inputs = {
            "carbon": self.carbon_cost,
            "oxygen": self.oxygen_cost,
            "germanium": self.germanium_cost,
            "silicon": self.silicon_cost,
            "energy": self.energy_cost,
        }

        tiny_protocols = [
            ProtocolConfig(
                vibes=[vibe] * (i + 1),
                input_resources=tiny_inputs,
                output_resources={"heart": i + 1},
            )
            for vibe in ("heart_a", "red-heart")
            for i in range(4)
        ]
        tiny_keys = {(tuple(p.vibes), p.min_agents) for p in tiny_protocols}
        existing = [p for p in assembler.protocols if (tuple(p.vibes), p.min_agents) not in tiny_keys]
        assembler.protocols = [*tiny_protocols, *existing]


class VibeCheckMin2Variant(MissionVariant):
    name: str = "vibe_check_min_2"
    description: str = "Require at least 2 heart vibes to craft a heart."
    min_vibes: int = 2

    @override
    def modify_env(self, mission, env):
        assembler = env.game.objects["assembler"]
        if not isinstance(assembler, AssemblerConfig):
            raise TypeError("Expected 'assembler' to be AssemblerConfig")

        filtered: list[ProtocolConfig] = []
        for proto in assembler.protocols:
            # Keep non-heart protocols as-is (e.g., gear recipes)
            if proto.output_resources.get("heart", 0) == 0:
                filtered.append(proto)
                continue
            # Keep only heart protocols that require >= 2 'heart' vibes
            if len(proto.vibes) >= 2 and all(v == "heart_a" for v in proto.vibes):
                filtered.append(proto)
        assembler.protocols = filtered


class Small50Variant(MissionVariant):
    name: str = "small_50"
    description: str = "Set map size to 50x50 for quick runs."

    def modify_env(self, mission, env) -> None:
        map_builder = env.game.map_builder
        # Only set width/height if instance is a SceneConfig, not a MapBuilderConfig
        # When instance is a MapBuilderConfig, width and height must be None
        if isinstance(map_builder, MapGen.Config) and isinstance(map_builder.instance, MapBuilderConfig):
            # Skip setting width/height for MapBuilderConfig instances
            return
        env.game.map_builder = map_builder.model_copy(update={"width": 50, "height": 50})


class InventoryHeartTuneVariant(MissionVariant):
    name: str = "inventory_heart_tune"
    description: str = "Tune starting agent inventory to N hearts worth of inputs; optional heart capacity."
    hearts: int = 1
    heart_capacity: int | None = None

    @override
    def modify_env(self, mission, env) -> None:
        hearts = max(0, int(self.hearts))
        if hearts == 0 and self.heart_capacity is None:
            return

        heart_cost = mission.assembler.first_heart_cost
        per_heart = {
            "carbon": heart_cost,
            "oxygen": heart_cost,
            "germanium": max(heart_cost // 10, 1),
            "silicon": 3 * heart_cost,
            "energy": 0,
        }

        if hearts > 0:
            agent_cfg = env.game.agent
            agent_cfg.inventory.initial = dict(agent_cfg.inventory.initial)

            def _limit_for(resource: str) -> int:
                return agent_cfg.inventory.get_limit(resource)

            for resource_name, per_heart_value in per_heart.items():
                current = int(agent_cfg.inventory.initial.get(resource_name, 0))
                target = current + per_heart_value * hearts
                cap = _limit_for(resource_name)
                agent_cfg.inventory.initial[resource_name] = min(cap, target)

        if self.heart_capacity is not None:
            agent_cfg = env.game.agent
            hearts_limit = agent_cfg.inventory.limits.get("heart")
            if hearts_limit is None:
                hearts_limit = ResourceLimitsConfig(limit=self.heart_capacity, resources=["heart"])
            hearts_limit.limit = max(int(hearts_limit.limit), int(self.heart_capacity))
            agent_cfg.inventory.limits["heart"] = hearts_limit


class ChestHeartTuneVariant(MissionVariant):
    name: str = "chest_heart_tune"
    description: str = "Tune chest starting inventory to N hearts worth of inputs."
    hearts: int = 2

    @override
    def modify_env(self, mission, env) -> None:
        hearts = max(0, int(self.hearts))
        if hearts == 0:
            return
        heart_cost = mission.assembler.first_heart_cost
        per_heart = {
            "carbon": heart_cost,
            "oxygen": heart_cost,
            "germanium": max(heart_cost // 10, 1),
            "silicon": 3 * heart_cost,
        }
        chest_cfg = env.game.objects["chest"]
        if not isinstance(chest_cfg, ChestConfig):
            raise TypeError("Expected 'chest' to be ChestConfig")
        start = dict(chest_cfg.inventory.initial)
        for k, v in per_heart.items():
            start[k] = start.get(k, 0) + v * hearts
        chest_cfg.inventory.initial = start


class ExtractorHeartTuneVariant(MissionVariant):
    name: str = "extractor_heart_tune"
    description: str = "Tune extractors for N hearts production capability."
    hearts: int = 1

    @override
    def modify_mission(self, mission):
        hearts = max(0, int(self.hearts))
        if hearts == 0:
            return
        heart_cost = mission.assembler.first_heart_cost
        one_heart = {
            "carbon": heart_cost,
            "oxygen": heart_cost,
            "germanium": max(heart_cost // 10, 1),
            "silicon": 3 * heart_cost,
        }

        # Carbon per-use depends on efficiency
        carbon_per_use = max(1, 4 * mission.carbon_extractor.efficiency // 100)
        carbon_needed = one_heart["carbon"] * hearts
        mission.carbon_extractor.max_uses = (carbon_needed + carbon_per_use - 1) // carbon_per_use

        # Oxygen is 20 per use
        oxygen_per_use = 20
        oxygen_needed = one_heart["oxygen"] * hearts
        mission.oxygen_extractor.max_uses = (oxygen_needed + oxygen_per_use - 1) // oxygen_per_use

        # Silicon is ~25 per use (scaled by efficiency); silicon extractor divides by 10 internally
        silicon_per_use = max(1, int(25 * mission.silicon_extractor.efficiency // 100))
        silicon_needed = one_heart["silicon"] * hearts
        silicon_uses = (silicon_needed + silicon_per_use - 1) // silicon_per_use
        mission.silicon_extractor.max_uses = max(1, silicon_uses * 10)

        # Germanium: fixed one use producing all required
        mission.germanium_extractor.efficiency = int(one_heart["germanium"] * hearts)


class CyclicalUnclipVariant(MissionVariant):
    name: str = "cyclical_unclip"
    description: str = "Required resources for unclipping recipes are cyclical. \
                        So Germanium extractors require silicon-based unclipping recipes."

    @override
    def modify_env(self, mission, env):
        if env.game.clipper is not None:
            env.game.clipper.unclipping_protocols = [
                ProtocolConfig(input_resources={"scrambler": 1}, cooldown=1),
                ProtocolConfig(input_resources={"resonator": 1}, cooldown=1),
                ProtocolConfig(input_resources={"modulator": 1}, cooldown=1),
                ProtocolConfig(input_resources={"decoder": 1}, cooldown=1),
            ]


class ClipHubStationsVariant(MissionVariant):
    name: str = "clip_hub_stations"
    description: str = "Clip the specified base stations (by name)."
    # Valid names: "carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor", "charger"
    clip: list[str] = ["carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor", "charger"]

    @override
    def modify_mission(self, mission):
        for station_name in self.clip:
            station = getattr(mission, station_name, None)
            if station is not None:
                station.start_clipped = True


class ClipPeriodOnVariant(MissionVariant):
    name: str = "clip_period_on"
    description: str = "Enable global clipping with a small non-zero clip period."
    clip_period: int = 50

    @override
    def modify_mission(self, mission):
        mission.clip_period = self.clip_period


# Biome variants (weather) for procedural maps
class DesertVariant(MachinaArenaVariant):
    name: str = "desert"
    description: str = "The desert sands make navigation challenging."

    @override
    def modify_node(self, node):
        node.biome_weights = {"desert": 1.0, "caves": 0.0, "forest": 0.0, "city": 0.0}
        node.base_biome = "desert"


class ForestVariant(MachinaArenaVariant):
    name: str = "forest"
    description: str = "Dense forests obscure your view."

    @override
    def modify_node(self, node):
        node.biome_weights = {"forest": 1.0, "caves": 0.0, "desert": 0.0, "city": 0.0}
        node.base_biome = "forest"


class CityVariant(MachinaArenaVariant):
    name: str = "city"
    description: str = "Ancient city ruins provide structured pathways."

    def modify_node(self, node):
        node.biome_weights = {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0}
        node.base_biome = "city"
        # Fill almost the entire map with the city layer
        node.density_scale = 1.0
        node.biome_count = 1
        node.max_biome_zone_fraction = 0.95
        # Tighten the city grid itself


class CavesVariant(MachinaArenaVariant):
    name: str = "caves"
    description: str = "Winding cave systems create a natural maze."

    @override
    def modify_node(self, node):
        node.biome_weights = {"caves": 1.0, "desert": 0.0, "forest": 0.0, "city": 0.0}
        node.base_biome = "caves"


class DistantResourcesVariant(MachinaArenaVariant):
    name: str = "distant_resources"
    description: str = "Resources scattered far from base; heavy routing coordination."
    building_names: list[str] = ["carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor"]

    @override
    def modify_node(self, node):
        # Bias buildings toward the map edges using bimodal clusters centered at
        node.building_coverage = 0.01

        vertical_edges = DistributionConfig(
            type=DistributionType.BIMODAL,
            center1_x=0.92,  # top right corner
            center1_y=0.08,
            center2_x=0.08,  # bottom left corner
            center2_y=0.92,
            cluster_std=0.18,
        )
        horizontal_edges = DistributionConfig(
            type=DistributionType.BIMODAL,
            center1_x=0.08,  # top left corner
            center1_y=0.08,
            center2_x=0.92,  # bottom right corner
            center2_y=0.92,
            cluster_std=0.18,
        )

        # Apply edge-biased distributions to extractors; other buildings follow the global distribution
        names = list(self.building_names)
        node.building_distributions = {
            name: (vertical_edges if i % 2 == 0 else horizontal_edges) for i, name in enumerate(names)
        }
        # Fallback for any unspecified building types
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


class SingleUseSwarmVariant(MissionVariant):
    name: str = "single_use_swarm"
    description: str = "Everything is single use; agents must fan out and reconverge."
    building_coverage: float = 0.03

    @override
    def modify_mission(self, mission):
        # Make each extractor single-use
        for res in ("carbon", "oxygen", "silicon"):
            extractor = getattr(mission, f"{res}_extractor", None)
            if extractor is not None:
                extractor.max_uses = 1

    @override
    def modify_env(self, mission, env):
        # Ensure charger is also single-use (its Config defaults to unlimited)
        charger = env.game.objects.get("charger")
        if isinstance(charger, AssemblerConfig):
            charger.max_uses = 1

        # Increase building coverage a bit to create many single-use points
        map_builder = getattr(env.game, "map_builder", None)
        instance = getattr(map_builder, "instance", None)
        if instance is not None and hasattr(instance, "building_coverage"):
            current = float(getattr(instance, "building_coverage", 0.01))
            instance.building_coverage = max(current, float(self.building_coverage))


class QuadrantBuildingsVariant(MachinaArenaVariant):
    name: str = "quadrant_buildings"
    description: str = "Place buildings in the four quadrants of the map."
    building_names: list[str] = ["carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor"]

    @override
    def modify_node(self, node):
        node.building_names = self.building_names

        names = list(node.building_names or self.building_names)
        centers = [
            (0.25, 0.25),  # top-left
            (0.75, 0.25),  # top-right
            (0.25, 0.75),  # bottom-left
            (0.75, 0.75),  # bottom-right
        ]
        dists: dict[str, DistributionConfig] = {}
        for i, name in enumerate(names):
            cx, cy = centers[i % len(centers)]
            dists[name] = DistributionConfig(
                type=DistributionType.NORMAL,
                mean_x=cx,
                mean_y=cy,
                std_x=0.18,
                std_y=0.18,
            )
        node.building_distributions = dists
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


class SingleResourceUniformVariant(MachinaArenaVariant):
    name: str = "single_resource_uniform"
    description: str = "Place only a single building via uniform distribution across the map."
    building_name: str = "oxygen_extractor"

    @override
    def modify_node(self, node):
        # Resolve resource to a concrete building name
        # Restrict building set to only the chosen building and enforce uniform distribution
        node.building_names = [self.building_name]
        node.building_weights = {self.building_name: 1.0}
        node.building_distributions = None
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


class EmptyBaseVariant(BaseHubVariant):
    name: str = "empty_base"
    description: str = "Base hub with extractors removed from the four corners."
    # Extractor object names to remove, e.g., ["oxygen_extractor"]
    missing: list[str] = list(HUB_EXTRACTORS)

    @override
    def modify_node(self, node):
        # Use the default extractor order and blank out any that are missing
        missing_set = set(self.missing or [])
        corner_objects = [name if name not in missing_set else "" for name in HUB_EXTRACTORS]
        node.corner_objects = corner_objects
        node.corner_bundle = "custom"


class AssemblerDrawsFromChestsVariant(BaseHubVariant):
    name: str = "assembler_draws_from_chests"
    description: str = "Assembler draws from chests."

    # It would be better if this were configurable, but we use variants in places where that's hard.
    # This needs to not overlap with the default (heart) chest.
    chest_distance: int = 2

    @override
    def modify_node(self, node):
        node.cross_objects = ["chest_carbon", "chest_oxygen", "chest_germanium", "chest_silicon"]
        node.cross_bundle = "custom"
        node.cross_distance = self.chest_distance

    @override
    def modify_env(self, mission, env):
        super().modify_env(mission, env)
        assembler = env.game.objects["assembler"]
        assert isinstance(assembler, AssemblerConfig)
        assembler.chest_search_distance = self.chest_distance
        chest = env.game.objects["chest"]
        assert isinstance(chest, ChestConfig)
        chest.vibe_transfers = {
            "default": {
                "heart": 255,
            }
        }


class BalancedCornersVariant(MachinaArenaVariant):
    """Enable corner balancing to ensure fair spawn distances."""

    name: str = "balanced_corners"
    description: str = "Balance path distances from center to corners for fair spawns."
    balance_tolerance: float = 1.5
    max_balance_shortcuts: int = 10

    @override
    def modify_node(self, node):
        node.balance_corners = True
        node.balance_tolerance = self.balance_tolerance
        node.max_balance_shortcuts = self.max_balance_shortcuts


class TraderVariant(MissionVariant):
    name: str = "trader"
    description: str = "Agents can trade resources with each other."

    @override
    def modify_env(self, mission, env):
        # Define vibe transfers for trading resources (actor gives, target receives)
        trade_transfers = [
            VibeTransfer(vibe="carbon_a", target={"carbon": 1}, actor={"carbon": -1}),
            VibeTransfer(vibe="carbon_b", target={"carbon": 10}, actor={"carbon": -10}),
            VibeTransfer(vibe="oxygen_a", target={"oxygen": 1}, actor={"oxygen": -1}),
            VibeTransfer(vibe="oxygen_b", target={"oxygen": 10}, actor={"oxygen": -10}),
            VibeTransfer(vibe="germanium_a", target={"germanium": 1}, actor={"germanium": -1}),
            VibeTransfer(vibe="germanium_b", target={"germanium": 4}, actor={"germanium": -4}),
            VibeTransfer(vibe="silicon_a", target={"silicon": 10}, actor={"silicon": -10}),
            VibeTransfer(vibe="silicon_b", target={"silicon": 50}, actor={"silicon": -50}),
            VibeTransfer(vibe="heart_a", target={"heart": 1}, actor={"heart": -1}),
            VibeTransfer(vibe="heart_b", target={"heart": 4}, actor={"heart": -4}),
        ]
        # Enable transfer action with these vibes
        env.game.actions.transfer.enabled = True
        env.game.actions.transfer.vibe_transfers.extend(trade_transfers)


class SharedRewardsVariant(MissionVariant):
    name: str = "shared_rewards"
    description: str = "Rewards for deposited hearts are shared among all agents."

    @override
    def modify_env(self, mission, env):
        num_cogs = mission.num_cogs if mission.num_cogs is not None else mission.site.min_cogs
        rewards = dict(env.game.agent.rewards.stats)
        rewards["chest.heart.deposited_by_agent"] = 0
        rewards["chest.heart.amount"] = 1 / num_cogs
        env.game.agent.rewards.stats = rewards


# TODO - validate that all variant names are unique
VARIANTS: list[MissionVariant] = [
    AssemblerDrawsFromChestsVariant(),
    CavesVariant(),
    ChestHeartTuneVariant(),
    CityVariant(),
    ClipHubStationsVariant(),
    ClipPeriodOnVariant(),
    CompassVariant(),
    CyclicalUnclipVariant(),
    DarkSideVariant(),
    DesertVariant(),
    EmptyBaseVariant(),
    EnergizedVariant(),
    ExtractorHeartTuneVariant(),
    ForestVariant(),
    HeartChorusVariant(),
    InventoryHeartTuneVariant(),
    LonelyHeartVariant(),
    MinedOutVariant(),
    PackRatVariant(),
    QuadrantBuildingsVariant(),
    ResourceBottleneckVariant(),
    RoughTerrainVariant(),
    SharedRewardsVariant(),
    SingleResourceUniformVariant(),
    SingleToolUnclipVariant(),
    Small50Variant(),
    SolarFlareVariant(),
    SuperChargedVariant(),
    TraderVariant(),
    TinyHeartProtocolsVariant(),
    TrainingVariant(),
    VibeCheckMin2Variant(),
    *DIFFICULTY_VARIANTS,
]

# Hidden variants registry: Remains usable but will NOT appear in `cogames variants` listing
HIDDEN_VARIANTS: list[MissionVariant] = [
    # Example: ExperimentalVariant(),  # keep empty by default
]
