from __future__ import annotations

from pathlib import Path
from typing import Dict

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


MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "maps"


def get_map(map_name: str) -> MapBuilderConfig:
    """Load a map builder configuration from the adversarial_maps directory."""
    normalized = map_name
    if normalized.startswith("adversarial/"):
        normalized = f"adversarial_maps/{normalized.split('/', 1)[1]}"
    map_path = MAPS_DIR / normalized
    if not map_path.exists():
        raise FileNotFoundError(f"Adversarial map not found: {map_path}")
    return MapGen.Config(
        instance=MapBuilderConfig.from_uri(str(map_path)),
        instances=1,
        fixed_spawn_order=False,
        instance_border_width=0,
    )


ADVERSARIAL = Site(
    name="adversarial",
    description="Adversarial evaluation arenas.",
    map_builder=get_map("adversarial/adversarial_crowded.map"),
    min_cogs=1,
    max_cogs=4,
)


class _AdversarialMissionBase(Mission):
    """Base class for adversarial evaluation missions."""

    site: Site = ADVERSARIAL

    map_name: str = Field(default="adversarial/adversarial_crowded.map")
    max_steps: int = Field(default=1000)
    required_agents: int | None = Field(default=3)

    inventory_seed: Dict[str, int] = Field(default_factory=dict)
    dynamic_assembler_chorus: bool = Field(default=True)
    generous_energy: bool = Field(default=True)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        """Hook for mission-specific environment alterations."""

    def configure(self) -> None:
        self.heart_capacity = max(self.heart_capacity, 255)
        self.cargo_capacity = max(self.cargo_capacity, 255)
        self.gear_capacity = max(self.gear_capacity, 255)
        self.energy_capacity = max(self.energy_capacity, 255)
        if self.generous_energy:
            self.energy_regen_amount = self.energy_capacity

    def make_env(self) -> MettaGridConfig:
        """Override make_env to use the mission's map_name instead of site.map_builder."""
        forced_map = get_map(self.map_name)
        original_map_builder = self.site.map_builder
        self.site.map_builder = forced_map
        try:
            cfg = super().make_env()
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = self.max_steps
            self._apply_inventory_seed(cfg)
            self._apply_chest_capacity(cfg)
            self._apply_extractor_outputs(cfg)
            if self.dynamic_assembler_chorus:
                self._apply_dynamic_chorus(cfg)
            self.configure_env(cfg)
            return cfg
        finally:
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
        mission = super().instantiate(forced_map, num_cogs, variant, cli_override=cli_override)
        if not cli_override and self.required_agents is not None:
            mission.num_cogs = self.required_agents

        def _post(cfg: MettaGridConfig) -> None:
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = self.max_steps
            self._apply_inventory_seed(cfg)
            self._apply_chest_capacity(cfg)
            self._apply_extractor_outputs(cfg)
            if self.dynamic_assembler_chorus:
                self._apply_dynamic_chorus(cfg)
            self.configure_env(cfg)

        return _add_make_env_modifier(mission, _post)

    def _apply_inventory_seed(self, cfg: MettaGridConfig) -> None:
        if not self.inventory_seed:
            return
        seed = dict(cfg.game.agent.initial_inventory)
        seed.update(self.inventory_seed)
        cfg.game.agent.initial_inventory = seed

    def _apply_chest_capacity(self, cfg: MettaGridConfig) -> None:
        """Limit all chests to hold at most 1 heart."""
        for _name, obj in cfg.game.objects.items():
            if not isinstance(obj, ChestConfig):
                continue
            heart_limit = obj.resource_limits.get("heart", ResourceLimitsConfig(limit=1, resources=["heart"]))
            heart_limit.limit = 1
            obj.resource_limits["heart"] = heart_limit

    def _apply_extractor_outputs(self, cfg: MettaGridConfig) -> None:
        """Set each extractor to output exactly what's needed for a heart, with cooldown."""
        # Get heart recipe from assembler's first heart protocol
        assembler = cfg.game.objects.get("assembler")
        if not isinstance(assembler, AssemblerConfig):
            return
        heart_recipe = {}
        for proto in assembler.protocols:
            if proto.output_resources.get("heart", 0) == 1:  # Single heart protocol
                heart_recipe = dict(proto.input_resources)
                heart_recipe.pop("energy", None)  # Energy comes from agent, not extractors
                break

        for resource, amount in heart_recipe.items():
            extractor = cfg.game.objects.get(f"{resource}_extractor")
            if not isinstance(extractor, AssemblerConfig):
                continue
            updated: list[ProtocolConfig] = []
            for proto in extractor.protocols:
                if resource in proto.output_resources:
                    updated_proto = proto.model_copy(update={
                        "output_resources": {resource: amount},
                        "cooldown": 20,
                    })
                    updated.append(updated_proto)
                else:
                    updated.append(proto)
            extractor.protocols = updated

    def _apply_dynamic_chorus(self, cfg: MettaGridConfig) -> None:
        num_agents = max(1, int(cfg.game.num_agents))
        assembler = cfg.game.objects.get("assembler")
        if not isinstance(assembler, AssemblerConfig):
            return
        chorus = ["heart_a"] * num_agents
        updated: list[ProtocolConfig] = []
        heart_protocol_applied = False
        for proto in assembler.protocols:
            if proto.output_resources.get("heart", 0) > 0:
                if heart_protocol_applied:
                    continue
                updated_proto = proto.model_copy(update={"vibes": chorus})
                updated.append(updated_proto)
                heart_protocol_applied = True
            else:
                updated.append(proto)
        assembler.protocols = updated


def _add_make_env_modifier(mission: Mission, modifier) -> Mission:
    """Attach a post-make_env modifier to an instantiated mission."""
    original_make_env = mission.make_env

    def wrapped_make_env() -> MettaGridConfig:
        env_cfg = original_make_env()
        modifier(env_cfg)
        return env_cfg

    object.__setattr__(mission, "make_env", wrapped_make_env)
    return mission


# ----------------------------------------------------------------------
# Adversarial missions
# ----------------------------------------------------------------------


class AdversarialAsymmetric(_AdversarialMissionBase):
    name: str = "adversarial_asymmetric"
    description: str = "Asymmetric resource layout with complex navigation."
    map_name: str = "adversarial/adversarial_asymmetric.map"
    required_agents: int | None = 3


class AdversarialBlockedAssembler(_AdversarialMissionBase):
    name: str = "adversarial_blocked_assembler"
    description: str = "Large map with blocked paths to assembler and scattered resources."
    map_name: str = "adversarial/adversarial_blocked_assembler.map"
    required_agents: int | None = 3


class AdversarialBottleneck(_AdversarialMissionBase):
    name: str = "adversarial_bottleneck"
    description: str = "Map with bottleneck passages creating competition points."
    map_name: str = "adversarial/adversarial_bottleneck.map"
    required_agents: int | None = 3


class AdversarialCrowded(_AdversarialMissionBase):
    name: str = "adversarial_crowded"
    description: str = "Crowded central area with symmetric resource quadrants."
    map_name: str = "adversarial/adversarial_crowded.map"
    required_agents: int | None = 3


ADVERSARIAL_EVALS: list[type[_AdversarialMissionBase]] = [
    AdversarialAsymmetric,
    AdversarialBlockedAssembler,
    AdversarialBottleneck,
    AdversarialCrowded,
]

