from __future__ import annotations

from typing import Dict

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import _add_make_env_modifier, get_map
from cogames.cogs_vs_clips.sites import EVALS, Site
from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, MettaGridConfig, ProtocolConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig

RESOURCE_NAMES: tuple[str, ...] = ("carbon", "oxygen", "germanium", "silicon")


class _DiagnosticMissionBase(Mission):
    """Base class for minimal diagnostic evaluation missions."""

    site: Site = EVALS

    map_name: str = Field(default="evals/diagnostic_eval_template.map")
    max_steps: int = Field(default=100)
    required_agents: int | None = Field(default=None)

    inventory_seed: Dict[str, int] = Field(default_factory=dict)
    communal_chest_hearts: int | None = Field(default=None)
    resource_chest_stock: Dict[str, int] = Field(default_factory=dict)
    clip_extractors: set[str] = Field(default_factory=set)
    extractor_max_uses: Dict[str, int] = Field(default_factory=dict)
    assembler_heart_chorus: int = Field(default=1)

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - hook for subclasses
        """Hook for mission-specific environment alterations."""

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
        *,
        cli_override: bool = False,
    ) -> "Mission":
        forced_map = get_map(self.map_name)
        mission = super().instantiate(forced_map, num_cogs, variant, cli_override=cli_override)
        if not cli_override and self.required_agents is not None:
            mission.num_cogs = self.required_agents

        def _post(cfg: MettaGridConfig) -> None:
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = self.max_steps
            self._apply_inventory_seed(cfg)
            self._apply_communal_chest(cfg)
            self._apply_resource_chests(cfg)
            self._apply_extractor_settings(cfg)
            self._apply_assembler_requirements(cfg)
            self.configure_env(cfg)

        return _add_make_env_modifier(mission, _post)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_inventory_seed(self, cfg: MettaGridConfig) -> None:
        if not self.inventory_seed:
            return
        seed = dict(cfg.game.agent.initial_inventory)
        seed.update(self.inventory_seed)
        cfg.game.agent.initial_inventory = seed

    def _apply_communal_chest(self, cfg: MettaGridConfig) -> None:
        if self.communal_chest_hearts is None:
            return
        chest = cfg.game.objects.get("communal_chest")
        if isinstance(chest, ChestConfig):
            chest.initial_inventory = self.communal_chest_hearts

    def _apply_resource_chests(self, cfg: MettaGridConfig) -> None:
        if not self.resource_chest_stock:
            return
        for resource, amount in self.resource_chest_stock.items():
            chest_cfg = cfg.game.objects.get(f"chest_{resource}")
            if isinstance(chest_cfg, ChestConfig):
                chest_cfg.initial_inventory = amount

    def _apply_extractor_settings(self, cfg: MettaGridConfig) -> None:
        for resource in RESOURCE_NAMES:
            extractor = cfg.game.objects.get(f"{resource}_extractor")
            if extractor is None:
                continue
            if resource in self.clip_extractors and hasattr(extractor, "start_clipped"):
                extractor.start_clipped = True
            if resource in self.extractor_max_uses and hasattr(extractor, "max_uses"):
                extractor.max_uses = self.extractor_max_uses[resource]
            clipped_key = f"clipped_{resource}_extractor"
            if clipped_key in cfg.game.objects:
                cfg.game.objects.pop(clipped_key)

    def _apply_assembler_requirements(self, cfg: MettaGridConfig) -> None:
        assembler = cfg.game.objects.get("assembler")
        if not isinstance(assembler, AssemblerConfig):
            return
        self._ensure_minimal_heart_recipe(assembler)
        if self.assembler_heart_chorus <= 1:
            return
        chorus = ["heart"] * self.assembler_heart_chorus
        updated: list[ProtocolConfig] = []
        heart_protocol_applied = False
        for proto in assembler.protocols:
            if proto.output_resources.get("heart", 0) > 0:
                if heart_protocol_applied:
                    # Drop duplicate heart protocols to avoid vibe collisions.
                    continue
                proto = proto.model_copy(update={"vibes": chorus})
                heart_protocol_applied = True
            updated.append(proto)
        assembler.protocols = updated

    def _ensure_minimal_heart_recipe(self, assembler: AssemblerConfig) -> None:
        minimal_inputs = {
            "carbon": 2,
            "oxygen": 2,
            "germanium": 1,
            "silicon": 3,
            "energy": 2,
        }

        for idx, proto in enumerate(assembler.protocols):
            if proto.output_resources.get("heart", 0) > 0:
                assembler.protocols[idx] = proto.model_copy(
                    update={
                        "input_resources": minimal_inputs,
                        "cooldown": 1,
                        "vibes": ["heart"],
                        "output_resources": {"heart": 1},
                    }
                )
                return

        assembler.protocols = [
            ProtocolConfig(
                vibes=["heart"],
                input_resources=minimal_inputs,
                output_resources={"heart": 1},
                cooldown=1,
            ),
            *assembler.protocols,
        ]


# ----------------------------------------------------------------------
# Single-agent diagnostics
# ----------------------------------------------------------------------


class DiagnosticChestDepositNear(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_near"
    description: str = "Deposit a carried heart into a visible chest."
    map_name: str = "evals/diagnostic_chest_near.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=40)


class DiagnosticChestDepositSearch(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_search"
    description: str = "Find the chest outside the initial FOV and deposit a heart."
    map_name: str = "evals/diagnostic_chest_search.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=60)


class DiagnosticAssembleSeededNear(_DiagnosticMissionBase):
    name: str = "diagnostic_assemble_seeded_near"
    description: str = "Assemble a heart with resources pre-seeded near the assembler."
    map_name: str = "evals/diagnostic_assembler_near.map"
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=50)


class DiagnosticAssembleSeededSearch(_DiagnosticMissionBase):
    name: str = "diagnostic_assemble_seeded_search"
    description: str = "Locate the assembler and craft a heart using seeded resources."
    map_name: str = "evals/diagnostic_assembler_search.map"
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=80)


class DiagnosticExtractMissingCarbon(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_carbon"
    description: str = "Gather carbon from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_resource_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"oxygen": 2, "germanium": 1, "silicon": 3})
    max_steps: int = Field(default=90)


class DiagnosticExtractMissingOxygen(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_oxygen"
    description: str = "Gather oxygen from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_resource_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "germanium": 1, "silicon": 3})
    max_steps: int = Field(default=90)


class DiagnosticExtractMissingGermanium(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_germanium"
    description: str = "Gather germanium from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_resource_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "silicon": 3})
    max_steps: int = Field(default=90)


class DiagnosticExtractMissingSilicon(_DiagnosticMissionBase):
    name: str = "diagnostic_extract_missing_silicon"
    description: str = "Gather silicon from the extractor to complete a heart."
    map_name: str = "evals/diagnostic_resource_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1})
    max_steps: int = Field(default=110)


class DiagnosticFullLoopAllExtractors(_DiagnosticMissionBase):
    name: str = "diagnostic_full_loop_all_extractors"
    description: str = "Execute the full extraction loop with no resources pre-seeded."
    map_name: str = "evals/diagnostic_resource_lab.map"
    max_steps: int = Field(default=160)


class DiagnosticUnclipCraftTool(_DiagnosticMissionBase):
    name: str = "diagnostic_unclip_craft_tool"
    description: str = "Craft an unclipping tool, free the extractor, gather resources, and craft a heart."
    map_name: str = "evals/diagnostic_unclip.map"
    clip_extractors: set[str] = Field(default_factory=lambda: {"carbon"})
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 1, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=170)


class DiagnosticUnclipPreseedTool(_DiagnosticMissionBase):
    name: str = "diagnostic_unclip_preseed_tool"
    description: str = "Use a pre-seeded tool to unclip an extractor and finish the heart loop."
    map_name: str = "evals/diagnostic_unclip.map"
    clip_extractors: set[str] = Field(default_factory=lambda: {"carbon"})
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"decoder": 1, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=130)


class DiagnosticForageBeyondBase(_DiagnosticMissionBase):
    name: str = "diagnostic_forage_beyond_base"
    description: str = "After exhausting base supplies, venture outward to finish the heart recipe."
    map_name: str = "evals/diagnostic_resource_lab.map"
    resource_chest_stock: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 0, "silicon": 1}
    )
    max_steps: int = Field(default=180)


class DiagnosticInventoryPrioritizeMissingOne(_DiagnosticMissionBase):
    name: str = "diagnostic_inventory_prioritize_missing_one"
    description: str = "Start with three ingredients and prioritise the missing resource to craft a heart."
    map_name: str = "evals/diagnostic_resource_lab.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1})
    max_steps: int = Field(default=100)


class DiagnosticInventoryPrioritizeRateLimiter(_DiagnosticMissionBase):
    name: str = "diagnostic_inventory_prioritize_rate_limiter"
    description: str = "Identify the rate limiting resource from chest inventory and prioritise it."
    map_name: str = "evals/diagnostic_resource_lab.map"
    resource_chest_stock: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 8, "oxygen": 8, "germanium": 1, "silicon": 8}
    )
    max_steps: int = Field(default=170)


# ----------------------------------------------------------------------
# Multi-agent chorus diagnostics
# ----------------------------------------------------------------------


class DiagnosticMultiAssembleDuo(_DiagnosticMissionBase):
    name: str = "diagnostic_multi_assemble_two_agents"
    description: str = "Two agents must chorus glyph HEART together."
    map_name: str = "evals/diagnostic_multi_chorus.map"
    required_agents: int | None = 2
    assembler_heart_chorus: int = 2
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=140)


class DiagnosticMultiAssembleTrio(_DiagnosticMissionBase):
    name: str = "diagnostic_multi_assemble_three_agents"
    description: str = "Three agents coordinate to chorus glyph HEART."
    map_name: str = "evals/diagnostic_multi_chorus.map"
    required_agents: int | None = 3
    assembler_heart_chorus: int = 3
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=160)


class DiagnosticMultiAssembleQuartet(_DiagnosticMissionBase):
    name: str = "diagnostic_multi_assemble_four_agents"
    description: str = "Four agents chorus glyph HEART together at the assembler."
    map_name: str = "evals/diagnostic_multi_chorus.map"
    required_agents: int | None = 4
    assembler_heart_chorus: int = 4
    inventory_seed: Dict[str, int] = Field(
        default_factory=lambda: {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}
    )
    max_steps: int = Field(default=180)


DIAGNOSTIC_EVALS: list[type[_DiagnosticMissionBase]] = [
    DiagnosticChestDepositNear,
    DiagnosticChestDepositSearch,
    DiagnosticAssembleSeededNear,
    DiagnosticAssembleSeededSearch,
    DiagnosticExtractMissingCarbon,
    DiagnosticExtractMissingOxygen,
    DiagnosticExtractMissingGermanium,
    DiagnosticExtractMissingSilicon,
    DiagnosticFullLoopAllExtractors,
    DiagnosticUnclipCraftTool,
    DiagnosticUnclipPreseedTool,
    DiagnosticForageBeyondBase,
    DiagnosticInventoryPrioritizeMissingOne,
    DiagnosticInventoryPrioritizeRateLimiter,
    DiagnosticMultiAssembleDuo,
    DiagnosticMultiAssembleTrio,
    DiagnosticMultiAssembleQuartet,
]
