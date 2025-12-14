from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    MettaGridConfig,
    ProtocolConfig,
    ResourceLimitsConfig,
)
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen

RESOURCE_NAMES: tuple[str, ...] = ("carbon", "oxygen", "germanium", "silicon")

MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "maps"


def get_map(map_name: str) -> MapBuilderConfig:
    """Load a map builder configuration from the local diagnostics directory."""
    normalized = map_name
    if normalized.startswith("evals/"):
        normalized = f"diagnostic_evals/{normalized.split('/', 1)[1]}"
    map_path = MAPS_DIR / normalized
    if not map_path.exists():
        raise FileNotFoundError(f"Diagnostic map not found: {map_path}")
    # Wrap AsciiMapBuilderConfig in MapGen.Config to match standard get_map() behavior
    return MapGen.Config(
        instance=MapBuilderConfig.from_uri(str(map_path)),
        instances=1,  # Force single instance - use spawn points from ASCII map directly
        fixed_spawn_order=False,
        instance_border_width=0,  # Don't add border - maps already have borders built in
    )


def _add_make_env_modifier(mission: Mission, modifier: Callable[[MettaGridConfig], None]) -> Mission:
    """Attach a post-make_env modifier to an instantiated mission."""
    original_make_env = mission.make_env

    def wrapped_make_env() -> MettaGridConfig:
        env_cfg = original_make_env()
        modifier(env_cfg)
        return env_cfg

    object.__setattr__(mission, "make_env", wrapped_make_env)
    return mission


EVALS = Site(
    name="evals",
    description="Diagnostic evaluation arenas.",
    map_builder=get_map("evals/diagnostic_radial.map"),
    min_cogs=1,
    max_cogs=4,
)


class _DiagnosticMissionBase(Mission):
    """Base class for minimal diagnostic evaluation missions."""

    site: Site = EVALS

    map_name: str = Field(default="evals/diagnostic_eval_template.map")
    max_steps: int = Field(default=250)
    required_agents: int | None = Field(default=None)

    inventory_seed: Dict[str, int] = Field(default_factory=dict)
    communal_chest_hearts: int | None = Field(default=None)
    resource_chest_stock: Dict[str, int] = Field(default_factory=dict)
    clip_extractors: set[str] = Field(default_factory=set)
    extractor_max_uses: Dict[str, int] = Field(default_factory=dict)
    assembler_heart_chorus: int = Field(default=1)
    # If True, set assembler heart chorus to the number of agents in the environment
    dynamic_assembler_chorus: bool = Field(default=False)
    # If True, give agents high energy capacity and regen (overridden by specific missions)
    generous_energy: bool = Field(default=True)

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - hook for subclasses
        """Hook for mission-specific environment alterations."""

    def configure(self) -> None:
        # Defaults per spec: large capacities, high regen unless mission disables it
        self.heart_capacity = max(self.heart_capacity, 255)
        self.cargo_capacity = max(self.cargo_capacity, 255)
        self.gear_capacity = max(self.gear_capacity, 255)
        self.energy_capacity = max(self.energy_capacity, 255)
        if self.generous_energy:
            # Full energy each step; effectively "lots of charge"
            self.energy_regen_amount = self.energy_capacity

    def make_env(self) -> MettaGridConfig:
        """Override make_env to use the mission's map_name instead of site.map_builder."""
        forced_map = get_map(self.map_name)
        # Temporarily override site.map_builder so parent make_env uses the correct map
        original_map_builder = self.site.map_builder
        self.site.map_builder = forced_map
        try:
            cfg = super().make_env()
            # Apply diagnostic-specific modifications
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = self.max_steps
            self._apply_inventory_seed(cfg)
            self._apply_communal_chest(cfg)
            self._apply_resource_chests(cfg)
            self._apply_extractor_settings(cfg)
            # Apply assembler requirements (may be overridden by dynamic chorus below)
            self._apply_assembler_requirements(cfg)
            # Zero out cooldowns everywhere to keep interactions snappy
            self._zero_all_protocol_cooldowns(cfg)
            # If required, set heart chorus to the number of agents after env is created
            if self.dynamic_assembler_chorus:
                self.assembler_heart_chorus = max(1, int(cfg.game.num_agents))
                self._apply_assembler_requirements(cfg)
            # Finally, normalize rewards so a single deposited heart yields at most 1 reward.
            self._apply_heart_reward_cap(cfg)
            self.configure_env(cfg)
            return cfg
        finally:
            # Restore original map_builder
            self.site.map_builder = original_map_builder

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
        *,
        cli_override: bool = False,
    ) -> "Mission":
        forced_map = get_map(self.map_name)
        # TODO: Mission doesn't have instantiate() - this code path appears unused
        mission = super().instantiate(forced_map, num_cogs, variant, cli_override=cli_override)  # type: ignore[attr-defined]
        if not cli_override and self.required_agents is not None:
            mission.num_cogs = self.required_agents

        def _post(cfg: MettaGridConfig) -> None:
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = self.max_steps
            self._apply_inventory_seed(cfg)
            self._apply_communal_chest(cfg)
            self._apply_resource_chests(cfg)
            self._apply_extractor_settings(cfg)
            # Apply assembler requirements (may be overridden by dynamic chorus below)
            self._apply_assembler_requirements(cfg)
            # Zero out cooldowns everywhere to keep interactions snappy
            self._zero_all_protocol_cooldowns(cfg)
            # If required, set heart chorus to the number of agents after env is created
            if self.dynamic_assembler_chorus:
                self.assembler_heart_chorus = max(1, int(cfg.game.num_agents))
                self._apply_assembler_requirements(cfg)
            # Finally, normalize rewards so a single deposited heart yields at most 1 reward.
            self._apply_heart_reward_cap(cfg)
            self.configure_env(cfg)

        return _add_make_env_modifier(mission, _post)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_inventory_seed(self, cfg: MettaGridConfig) -> None:
        if not self.inventory_seed:
            return
        seed = dict(cfg.game.agent.inventory.initial)
        seed.update(self.inventory_seed)
        cfg.game.agent.inventory.initial = seed

    def _apply_communal_chest(self, cfg: MettaGridConfig) -> None:
        if self.communal_chest_hearts is None:
            return
        chest = cfg.game.objects.get("communal_chest")
        if isinstance(chest, ChestConfig):
            chest.inventory.initial = {"heart": self.communal_chest_hearts}

    def _apply_resource_chests(self, cfg: MettaGridConfig) -> None:
        if not self.resource_chest_stock:
            return
        for resource, amount in self.resource_chest_stock.items():
            chest_cfg = cfg.game.objects.get(f"chest_{resource}")
            if isinstance(chest_cfg, ChestConfig):
                chest_cfg.inventory.initial = {resource: amount}

    def _apply_extractor_settings(self, cfg: MettaGridConfig) -> None:
        for resource in RESOURCE_NAMES:
            extractor = cfg.game.objects.get(f"{resource}_extractor")
            if not isinstance(extractor, AssemblerConfig):
                continue
            if resource in self.clip_extractors:
                extractor.start_clipped = True
            if resource in self.extractor_max_uses:
                extractor.max_uses = self.extractor_max_uses[resource]

    def _apply_assembler_requirements(self, cfg: MettaGridConfig) -> None:
        assembler = cfg.game.objects.get("assembler")
        if not isinstance(assembler, AssemblerConfig):
            return
        self._ensure_minimal_heart_recipe(assembler)
        if self.assembler_heart_chorus <= 1:
            return
        # Use a valid heart vibe from the CVC vibe set.
        chorus = ["heart_a"] * self.assembler_heart_chorus
        updated: list[ProtocolConfig] = []
        heart_protocol_applied = False
        for proto in assembler.protocols:
            if proto.output_resources.get("heart", 0) > 0:
                if heart_protocol_applied:
                    # Drop duplicate heart protocols to avoid vibe collisions.
                    continue
                updated_proto = proto.model_copy(update={"vibes": chorus})
                updated.append(updated_proto)
                heart_protocol_applied = True
            else:
                updated.append(proto)
        assembler.protocols = updated

    def _zero_all_protocol_cooldowns(self, cfg: MettaGridConfig) -> None:
        # Zero cooldowns on assembler/extractor protocols and unclipping protocols
        for _name, obj in list(cfg.game.objects.items()):
            if not isinstance(obj, AssemblerConfig):
                continue
            updated: list[ProtocolConfig] = []
            for proto in obj.protocols:
                updated.append(proto.model_copy(update={"cooldown": 0}))
            obj.protocols = updated
        if cfg.game.clipper is not None:
            new_up: list[ProtocolConfig] = []
            for proto in cfg.game.clipper.unclipping_protocols:
                new_up.append(proto.model_copy(update={"cooldown": 0}))
            cfg.game.clipper.unclipping_protocols = new_up

    def _apply_heart_reward_cap(self, cfg: MettaGridConfig) -> None:
        """Normalize diagnostics so a single deposited heart yields at most 1 reward per episode.

        - Make each agent-deposited heart worth exactly 1.0 reward (credited only to the depositor).
        - Ensure all chests can store at most 1 heart so total reward per episode cannot exceed 1.
        """
        agent_cfg = cfg.game.agent
        rewards = agent_cfg.rewards
        stats = dict(rewards.stats or {})
        stats["chest.heart.deposited_by_agent"] = 1.0
        agent_cfg.rewards = rewards.model_copy(update={"stats": stats})

        # Cap heart capacity for every chest used in diagnostics (communal or resource-specific).
        for _name, obj in cfg.game.objects.items():
            if not isinstance(obj, ChestConfig):
                continue
            # Find existing heart limit or create new one
            heart_limit = obj.inventory.limits.get("heart", ResourceLimitsConfig(limit=1, resources=["heart"]))
            heart_limit.limit = 1
            obj.inventory.limits["heart"] = heart_limit

    def _ensure_minimal_heart_recipe(self, assembler: AssemblerConfig) -> None:
        minimal_inputs = {
            "carbon": 2,
            "oxygen": 2,
            "germanium": 1,
            "silicon": 3,
            "energy": 2,
        }

        updated_protocols: list[ProtocolConfig] = []
        heart_recipe_applied = False

        for proto in assembler.protocols:
            if proto.output_resources.get("heart", 0) > 0:
                if heart_recipe_applied:
                    # Drop duplicate heart recipes to avoid conflicting requirements.
                    continue
                updated_proto = proto.model_copy(
                    update={
                        "vibes": ["heart_a"],
                        "input_resources": minimal_inputs,
                        "cooldown": 0,
                        "output_resources": {"heart": 1},
                    }
                )
                updated_protocols.append(updated_proto)
                heart_recipe_applied = True
            else:
                updated_protocols.append(proto)

        if not heart_recipe_applied:
            updated_protocols.insert(
                0,
                ProtocolConfig(
                    vibes=["heart_a"],
                    input_resources=minimal_inputs,
                    output_resources={"heart": 1},
                    cooldown=0,
                ),
            )

        assembler.protocols = updated_protocols


# ----------------------------------------------------------------------
# New diagnostics per spec
# ----------------------------------------------------------------------


# Chest navigation: agents start with a heart and must deposit it
class DiagnosticChestNavigation1(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation1"
    description: str = "Navigate to the chest and deposit a heart."
    map_name: str = "evals/diagnostic_chest_navigation1.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=250)
    required_agents: int | None = 1
    # 1-4 agents by default; no forced agent count


class DiagnosticChestNavigation2(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation2"
    description: str = "Navigate through obstacles to deposit a heart."
    map_name: str = "evals/diagnostic_chest_navigation2.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=250)
    required_agents: int | None = 1


class DiagnosticChestNavigation3(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation3"
    description: str = "Navigate obstacles to deposit a heart."
    map_name: str = "evals/diagnostic_chest_navigation3.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=250)
    required_agents: int | None = 1


# Chest deposit: explicitly single-agent defaults
class DiagnosticChestDepositNear(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_near"
    description: str = "Deposit a carried heart into a nearby chest."
    map_name: str = "evals/diagnostic_chest_near.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=250)


class DiagnosticChestDepositSearch(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_search"
    description: str = "Find the chest outside the initial FOV and deposit a heart."
    map_name: str = "evals/diagnostic_chest_search.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=250)


# Assemble seeded: 1-4 agents, assembler requires exactly num agents to chorus
class DiagnosticAssembleSeededNear(_DiagnosticMissionBase):
    name: str = "diagnostic_assemble_seeded_near"
    description: str = "Agents are pre-seeded; chorus glyph HEART near the assembler."
    map_name: str = "evals/diagnostic_assembler_near.map"
    dynamic_assembler_chorus: bool = True
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=50)


class DiagnosticAssembleSeededSearch(_DiagnosticMissionBase):
    name: str = "diagnostic_assemble_seeded_search"
    description: str = "Agents are pre-seeded; locate the assembler and chorus glyph HEART."
    map_name: str = "evals/diagnostic_assembler_search.map"
    dynamic_assembler_chorus: bool = True
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=150)


# Extract mission set: missing one resource; 1-4 agents; chorus required
class DiagnosticExtractMissingCarbon(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_carbon"
    description: str = "All agents start around the assembler; carbon must be extracted."
    map_name: str = "evals/diagnostic_extract_lab.map"
    dynamic_assembler_chorus: bool = True
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"oxygen": 2, "germanium": 1, "silicon": 3})
    max_steps: int = Field(default=130)


class DiagnosticExtractMissingOxygen(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_oxygen"
    description: str = "Gather oxygen from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_extract_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "germanium": 1, "silicon": 3})
    max_steps: int = Field(default=130)


class DiagnosticExtractMissingGermanium(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_germanium"
    description: str = "Gather germanium from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_extract_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "silicon": 3})
    max_steps: int = Field(default=130)


class DiagnosticExtractMissingSilicon(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_silicon"
    description: str = "Gather silicon from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_extract_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1})
    max_steps: int = Field(default=130)


class _UnclipBase(_DiagnosticMissionBase):
    map_name: str = "evals/diagnostic_unclip.map"
    dynamic_assembler_chorus: bool = True

    def configure_env(self, cfg: MettaGridConfig) -> None:
        # Determine how many extractors to clip based on number of agents (1..4)
        num_agents = max(1, int(cfg.game.num_agents))
        resources = list(RESOURCE_NAMES)
        to_clip = resources[: min(num_agents, len(resources))]
        # Clip the chosen extractors
        for res in to_clip:
            station = cfg.game.objects.get(f"{res}_extractor")
            if isinstance(station, AssemblerConfig):
                station.start_clipped = True
        # Configure unclipping to require the other three resources of the clipped station (no gear)
        unclipping_protos: list[ProtocolConfig] = []
        for res in to_clip:
            others = {r: 1 for r in resources if r != res}
            unclipping_protos.append(ProtocolConfig(input_resources=others, cooldown=0))
        if cfg.game.clipper is not None:
            cfg.game.clipper.unclipping_protocols = unclipping_protos

        non_clipped = [res for res in resources if res not in to_clip]

        assembler = cfg.game.objects.get("assembler")
        if isinstance(assembler, AssemblerConfig):
            updated_protocols: list[ProtocolConfig] = []
            for proto in assembler.protocols:
                if proto.output_resources.get("decoder", 0) > 0:
                    inputs = {res: 1 for res in non_clipped}
                    updated_proto = proto.model_copy(update={"vibes": ["gear"], "input_resources": inputs})
                    updated_protocols.append(updated_proto)
                else:
                    updated_protocols.append(proto)
            assembler.protocols = updated_protocols

        agent_cfg = cfg.game.agent
        inventory = dict(agent_cfg.inventory.initial)
        for res in resources:
            inventory.pop(res, None)

        base_amount = 2 if num_agents > 1 else 1
        for res in non_clipped:
            inventory[res] = max(inventory.get(res, 0), base_amount)
        for res in to_clip:
            inventory.pop(res, None)

        if num_agents > 1:
            inventory["decoder"] = max(inventory.get("decoder", 0), len(to_clip))
        else:
            inventory.pop("decoder", None)

        agent_cfg.inventory.initial = inventory


class DiagnosticUnclipCraft(_UnclipBase):
    name: str = "diagnostic_unclip_craft"
    description: str = "Craft to unclip extractors and complete a heart chorus."
    # No preseeded tools; agents have only basic resources below
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}
    )
    max_steps: int = Field(default=250)


class DiagnosticUnclipPreseed(_UnclipBase):
    name: str = "diagnostic_unclip_preseed"
    description: str = "Preseeded for unclipping; number of clipped extractors equals number of agents."
    # Preseed with a mix of resources to allow immediate unclipping
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 2, "silicon": 2}
    )
    max_steps: int = Field(default=250)


class DiagnosticChargeUp(_DiagnosticMissionBase):
    name: str = "diagnostic_charge_up"
    description: str = "Agent starts low on energy and must charge to proceed."
    map_name: str = "evals/diagnostic_charge_up.map"
    required_agents: int | None = 1
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    # Disable generous energy for this eval
    generous_energy: bool = False
    max_steps: int = Field(default=250)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        # Set starting energy to 30 and no regen
        agent = cfg.game.agent
        agent.inventory.initial = dict(agent.inventory.initial)
        agent.inventory.initial["energy"] = 60
        agent.inventory.regen_amounts = {"default": {"energy": 0}}


class DiagnosticAgile(_DiagnosticMissionBase):
    name: str = "diagnostic_agile"
    description: str = "Navigation agility challenge; 1-4 agents."
    map_name: str = "evals/diagnostic_agile.map"
    max_steps: int = Field(default=250)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        required = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
        for resource, needed in required.items():
            station = cfg.game.objects.get(f"{resource}_extractor")
            if isinstance(station, AssemblerConfig):
                station.max_uses = 1
                updated: list[ProtocolConfig] = []
                for proto in station.protocols:
                    outputs = dict(proto.output_resources)
                    if resource in outputs:
                        outputs = {resource: needed}
                        proto = proto.model_copy(update={"output_resources": outputs})
                    updated.append(proto)
                station.protocols = updated


class DiagnosticMemory(_DiagnosticMissionBase):
    name: str = "diagnostic_memory"
    description: str = "Harder memory challenge with longer distance to chest."
    map_name: str = "evals/diagnostic_memory.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=110)


class DiagnosticRadial(_DiagnosticMissionBase):
    name: str = "diagnostic_radial"
    description: str = "Radial resource layout; gather all four ingredients and chorus assemble."
    map_name: str = "evals/diagnostic_radial.map"
    dynamic_assembler_chorus: bool = True
    max_steps: int = Field(default=250)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        agent = cfg.game.agent
        inventory = dict(agent.inventory.initial)
        inventory["energy"] = 255
        agent.inventory.initial = inventory
        agent.inventory.regen_amounts = {"default": {"energy": 255}}


# ----------------------------------------------------------------------
# Hard versions of diagnostics (same maps, more time)
# ----------------------------------------------------------------------


class DiagnosticChestNavigation1Hard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation1_hard"
    description: str = "Navigate to the chest and deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_navigation1_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=350)
    required_agents: int | None = 1


class DiagnosticChestNavigation2Hard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation2_hard"
    description: str = "Navigate through obstacles to deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_navigation2_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=350)
    required_agents: int | None = 1


class DiagnosticChestNavigation3Hard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation3_hard"
    description: str = "Navigate obstacles to deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_navigation3_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=350)
    required_agents: int | None = 1


class DiagnosticChestDepositSearchHard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_search_hard"
    description: str = "Find the chest outside the initial FOV and deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_search_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=350)


class DiagnosticChargeUpHard(_DiagnosticMissionBase):
    name: str = "diagnostic_charge_up_hard"
    description: str = "Agent starts low on energy and must charge to proceed (hard)."
    map_name: str = "evals/diagnostic_charge_up_hard.map"
    required_agents: int | None = 1
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    # Disable generous energy for this eval
    generous_energy: bool = False
    max_steps: int = Field(default=350)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        # Set starting energy to 30 and no regen
        agent = cfg.game.agent
        agent.inventory.initial = dict(agent.inventory.initial)
        agent.inventory.initial["energy"] = 60
        agent.inventory.regen_amounts = {"default": {"energy": 0}}


class DiagnosticMemoryHard(_DiagnosticMissionBase):
    name: str = "diagnostic_memory_hard"
    description: str = "Harder memory challenge with longer distance to chest (hard)."
    map_name: str = "evals/diagnostic_memory_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=170)


class DiagnosticAssembleSeededSearchHard(_DiagnosticMissionBase):
    name: str = "diagnostic_assemble_seeded_search_hard"
    description: str = "Agents are pre-seeded; locate the assembler and chorus glyph HEART (hard)."
    map_name: str = "evals/diagnostic_assembler_search_hard.map"
    dynamic_assembler_chorus: bool = True
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=250)


class DiagnosticExtractMissingCarbonHard(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_carbon_hard"
    description: str = "All agents start around the assembler; carbon must be extracted (hard)."
    map_name: str = "evals/diagnostic_extract_lab_hard.map"
    dynamic_assembler_chorus: bool = True
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"oxygen": 2, "germanium": 1, "silicon": 3})
    max_steps: int = Field(default=230)


class DiagnosticExtractMissingOxygenHard(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_oxygen_hard"
    description: str = "Gather oxygen from the extractor to complete a heart (hard)."
    map_name: str = "evals/diagnostic_extract_lab_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "germanium": 1, "silicon": 3})
    max_steps: int = Field(default=230)


class DiagnosticExtractMissingGermaniumHard(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_germanium_hard"
    description: str = "Gather germanium from the extractor to complete a heart (hard)."
    map_name: str = "evals/diagnostic_extract_lab_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "silicon": 3})
    max_steps: int = Field(default=230)


class DiagnosticExtractMissingSiliconHard(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_silicon_hard"
    description: str = "Gather silicon from the extractor to complete a heart (hard)."
    map_name: str = "evals/diagnostic_extract_lab_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1})
    max_steps: int = Field(default=230)


class DiagnosticAgileHard(_DiagnosticMissionBase):
    name: str = "diagnostic_agile_hard"
    description: str = "Navigation agility challenge; 1-4 agents (hard)."
    map_name: str = "evals/diagnostic_agile_hard.map"
    max_steps: int = Field(default=350)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        required = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
        for resource, needed in required.items():
            station = cfg.game.objects.get(f"{resource}_extractor")
            if isinstance(station, AssemblerConfig):
                station.max_uses = 1
                updated: list[ProtocolConfig] = []
                for proto in station.protocols:
                    outputs = dict(proto.output_resources)
                    if resource in outputs:
                        outputs = {resource: needed}
                        proto = proto.model_copy(update={"output_resources": outputs})
                    updated.append(proto)
                station.protocols = updated


class DiagnosticRadialHard(_DiagnosticMissionBase):
    name: str = "diagnostic_radial_hard"
    description: str = "Radial resource layout; gather all four ingredients and chorus assemble (hard)."
    map_name: str = "evals/diagnostic_radial_hard.map"
    dynamic_assembler_chorus: bool = True
    max_steps: int = Field(default=350)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        agent = cfg.game.agent
        inventory = dict(agent.inventory.initial)
        inventory["energy"] = 255
        agent.inventory.initial = inventory
        agent.inventory.regen_amounts = {"default": {"energy": 255}}


DIAGNOSTIC_EVALS: list[type[_DiagnosticMissionBase]] = [
    DiagnosticChestNavigation1,
    DiagnosticChestNavigation2,
    DiagnosticChestNavigation3,
    DiagnosticChestDepositNear,
    DiagnosticChestDepositSearch,
    DiagnosticChargeUp,
    DiagnosticMemory,
    DiagnosticAssembleSeededNear,
    DiagnosticAssembleSeededSearch,
    DiagnosticExtractMissingCarbon,
    DiagnosticExtractMissingOxygen,
    DiagnosticExtractMissingGermanium,
    DiagnosticExtractMissingSilicon,
    DiagnosticUnclipCraft,
    DiagnosticUnclipPreseed,
    DiagnosticAgile,
    DiagnosticRadial,
    # Hard versions
    DiagnosticChestNavigation1Hard,
    DiagnosticChestNavigation2Hard,
    DiagnosticChestNavigation3Hard,
    DiagnosticChestDepositSearchHard,
    DiagnosticChargeUpHard,
    DiagnosticMemoryHard,
    DiagnosticAssembleSeededSearchHard,
    DiagnosticExtractMissingCarbonHard,
    DiagnosticExtractMissingOxygenHard,
    DiagnosticExtractMissingGermaniumHard,
    DiagnosticExtractMissingSiliconHard,
    DiagnosticAgileHard,
    DiagnosticRadialHard,
]
