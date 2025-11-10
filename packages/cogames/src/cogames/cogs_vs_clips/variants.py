from typing import override

from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_VARIANTS
from cogames.cogs_vs_clips.mission import MissionVariant
from cogames.cogs_vs_clips.procedural import BaseHubVariant, MachinaArenaVariant
from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, ProtocolConfig


class MinedOutVariant(MissionVariant):
    name: str = "mined_out"
    description: str = "Some resources are depleted. You must be efficient to survive."

    @override
    def modify_mission(self, mission):
        mission.carbon_extractor.efficiency -= 50
        mission.oxygen_extractor.efficiency -= 50
        mission.germanium_extractor.efficiency -= 50
        mission.silicon_extractor.efficiency -= 50


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
        mission.assembler.heart_cost = 1

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
        mission.charger.efficiency -= 50


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


class NeutralFacedVariant(MissionVariant):
    name: str = "neutral_faced"
    description: str = "Disable vibe swapping; keep neutral face."

    @override
    def modify_env(self, mission, env):
        change_vibe = env.game.actions.change_vibe
        change_vibe.enabled = False
        change_vibe.number_of_vibes = 1

        neutral_vibe_name = "default"
        env.game.vibe_names = [neutral_vibe_name]
        for name, obj in env.game.objects.items():
            if isinstance(obj, AssemblerConfig) and obj.protocols:
                primary_protocol = obj.protocols[0].model_copy(deep=True)
                primary_protocol.vibes = [neutral_vibe_name]
                obj.protocols = [primary_protocol]
            elif isinstance(obj, ChestConfig) and name == "chest":
                obj.vibe_transfers = {neutral_vibe_name: {"heart": 255}}


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
        env.game.agent.rewards.stats = {
            "heart.gained": 1.0,
            "chest.heart.deposited": 1.0,
            "chest.heart.withdrawn": -1.0,
            "inventory.diversity.ge.2": 0.17,
            "inventory.diversity.ge.3": 0.18,
            "inventory.diversity.ge.4": 0.60,
            "inventory.diversity.ge.5": 0.97,
        }


class VibeCheckMin2Variant(MissionVariant):
    name: str = "vibe_check_min_2"
    description: str = "Require at least 2 heart vibes to craft a heart."

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
            if len(proto.vibes) >= 2 and all(v == "heart" for v in proto.vibes):
                filtered.append(proto)
        assembler.protocols = filtered


class Small50Variant(MissionVariant):
    name: str = "small_50"
    description: str = "Set map size to 50x50 for quick runs."

    def modify_env(self, mission, env) -> None:
        env.game.map_builder = env.game.map_builder.model_copy(update={"width": 50, "height": 50})


class CogToolsOnlyVariant(MissionVariant):
    name: str = "cog_tools_only"
    description: str = "Gear tools (decoder/modulator/scrambler/resonator) require only the 'gear/cog' vibe."

    @override
    def modify_env(self, mission, env) -> None:
        assembler_cfg = env.game.objects["assembler"]
        if not isinstance(assembler_cfg, AssemblerConfig):
            raise TypeError("Expected 'assembler' to be AssemblerConfig")
        gear_outputs = {"decoder", "modulator", "scrambler", "resonator"}
        for protocol in assembler_cfg.protocols:
            if any(k in protocol.output_resources for k in gear_outputs):
                protocol.vibes = ["gear"]


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

        heart_cost = mission.assembler.heart_cost
        per_heart = {
            "carbon": heart_cost * 2,
            "oxygen": heart_cost * 2,
            "germanium": max(heart_cost // 2, 1),
            "silicon": heart_cost * 5,
            "energy": heart_cost * 2,
        }

        if hearts > 0:
            agent_cfg = env.game.agent
            agent_cfg.initial_inventory = dict(agent_cfg.initial_inventory)
            resource_limits = dict(agent_cfg.resource_limits)

            def _limit_for(resource: str) -> int:
                if resource in resource_limits:
                    return int(resource_limits[resource])
                for key, limit in resource_limits.items():
                    if isinstance(key, tuple) and resource in key:
                        return int(limit)
                return int(agent_cfg.default_resource_limit)

            for resource_name, per_heart_value in per_heart.items():
                current = int(agent_cfg.initial_inventory.get(resource_name, 0))
                target = current + per_heart_value * hearts
                cap = _limit_for(resource_name)
                agent_cfg.initial_inventory[resource_name] = min(cap, target)

        if self.heart_capacity is not None:
            agent_cfg = env.game.agent
            limits = dict(agent_cfg.resource_limits)
            limits["heart"] = max(int(limits.get("heart", 0)), int(self.heart_capacity))
            agent_cfg.resource_limits = limits


class ChestHeartTuneVariant(MissionVariant):
    name: str = "chest_heart_tune"
    description: str = "Tune chest starting inventory to N hearts worth of inputs."
    hearts: int = 2

    @override
    def modify_env(self, mission, env) -> None:
        hearts = max(0, int(self.hearts))
        if hearts == 0:
            return
        heart_cost = mission.assembler.heart_cost
        per_heart = {
            "carbon": heart_cost * 2,
            "oxygen": heart_cost * 2,
            "germanium": max(heart_cost // 2, 1),
            "silicon": heart_cost * 5,
        }
        chest_cfg = env.game.objects["chest"]
        if not isinstance(chest_cfg, ChestConfig):
            raise TypeError("Expected 'chest' to be ChestConfig")
        start = dict(chest_cfg.initial_inventory)
        for k, v in per_heart.items():
            start[k] = start.get(k, 0) + v * hearts
        chest_cfg.initial_inventory = start


class ExtractorHeartTuneVariant(MissionVariant):
    name: str = "extractor_heart_tune"
    description: str = "Tune extractors for N hearts production capability."
    hearts: int = 5

    @override
    def modify_mission(self, mission):
        hearts = max(0, int(self.hearts))
        if hearts == 0:
            return
        heart_cost = mission.assembler.heart_cost
        one_heart = {
            "carbon": heart_cost * 2,
            "oxygen": heart_cost * 2,
            "germanium": max(heart_cost // 2, 1),
            "silicon": heart_cost * 5,
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


class ClipBaseExceptCarbonVariant(MissionVariant):
    name: str = "clip_base_except_carbon"
    description: str = "Start base extractors clipped except carbon."

    @override
    def modify_mission(self, mission):
        mission.carbon_extractor.start_clipped = False
        mission.oxygen_extractor.start_clipped = True
        mission.germanium_extractor.start_clipped = True
        mission.silicon_extractor.start_clipped = True


class ClipHubStationsVariant(MissionVariant):
    name: str = "clip_hub_stations"
    description: str = "Start base extractors and charger clipped."

    @override
    def modify_mission(self, mission):
        mission.carbon_extractor.start_clipped = True
        mission.oxygen_extractor.start_clipped = True
        mission.germanium_extractor.start_clipped = True
        mission.silicon_extractor.start_clipped = True
        mission.charger.start_clipped = True


class ClipRateOnVariant(MissionVariant):
    name: str = "clip_rate_on"
    description: str = "Enable global clipping with a small non-zero clip rate."

    @override
    def modify_mission(self, mission):
        mission.clip_rate = 0.02


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


class EmptyBaseVariant(BaseHubVariant):
    name: str = "empty_base"
    description: str = "Empty base with no extractors or chests."

    @override
    def modify_node(self, node):
        # Explicit objects/generators take precedence over bundles in BaseHub.
        # Clear them so the 'none' bundles are respected.
        node.corner_objects = None
        node.corner_generator = None
        node.cross_objects = None
        node.corner_bundle = "none"
        node.cross_bundle = "none"


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


# TODO - validate that all variant names are unique
VARIANTS: list[MissionVariant] = [
    MinedOutVariant(),
    DarkSideVariant(),
    SuperChargedVariant(),
    RoughTerrainVariant(),
    SolarFlareVariant(),
    HeartChorusVariant(),
    VibeCheckMin2Variant(),
    DesertVariant(),
    ForestVariant(),
    CityVariant(),
    CavesVariant(),
    EmptyBaseVariant(),
    LonelyHeartVariant(),
    PackRatVariant(),
    EnergizedVariant(),
    NeutralFacedVariant(),
    CompassVariant(),
    Small50Variant(),
    CogToolsOnlyVariant(),
    InventoryHeartTuneVariant(),
    ChestHeartTuneVariant(),
    ExtractorHeartTuneVariant(),
    ClipBaseExceptCarbonVariant(),
    ClipHubStationsVariant(),
    CyclicalUnclipVariant(),
    ClipRateOnVariant(),
    *DIFFICULTY_VARIANTS,
]
