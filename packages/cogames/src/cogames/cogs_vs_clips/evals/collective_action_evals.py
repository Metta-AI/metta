"""Collective action evaluation scenarios.

This module encodes the full specification supplied for the collective-action
evaluation suite.  It focuses on recording the exact scenario structure,
background-behaviour mixes, mission-level metrics, and success criteria so that
downstream evaluation harnesses can consume the metadata consistently.

Wherever possible we also preconfigure the base environment (episode length,
reward shaping) to match the global protocol.  Mission-specific mechanics that
require engine-side support (e.g. waste dynamics, door locks, dual-operator
press logic) are represented as structured metadata under
``cfg.game.params["collective_action"]``.  This keeps the mission definitions
faithful to the design document while allowing future engine work to hook into
the declared rules without modifying the mission registry again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import _add_make_env_modifier, get_map
from cogames.cogs_vs_clips.sites import EVALS, Site
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    AgentRewards,
    AssemblerConfig,
    MettaGridConfig,
    ProtocolConfig,
)

# ---------------------------------------------------------------------------
# Background behaviour palette
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackgroundBehaviourSpec:
    """Describes an immutable background policy template."""

    name: str
    display_name: str
    description: str


BACKGROUND_BEHAVIOURS: Dict[str, BackgroundBehaviourSpec] = {
    "pacifist_greedy": BackgroundBehaviourSpec(
        name="pacifist_greedy",
        display_name="Pacifist-Greedy",
        description=(
            "Never jam or attack. Prioritises the nearest extractor or assembler "
            "and disregards sustainability or cleanup obligations."
        ),
    ),
    "zap_greedy": BackgroundBehaviourSpec(
        name="zap_greedy",
        display_name="Zap-Greedy",
        description=(
            "Harvests aggressively. If a non-teammate is within Manhattan distance "
            "≤ 1, or if a doorway blocks progress, issues jam/disable before "
            "continuing. Otherwise identical to Pacifist-Greedy."
        ),
    ),
    "qos_cleaner": BackgroundBehaviourSpec(
        name="qos_cleaner",
        display_name="QoS-Cleaner",
        description=(
            "Monitors local station waste. When any visible station reports "
            "state.waste > 0.5, paths to the nearest Janitor station and interacts "
            "until the reading falls to ≤ 0.2. Falls back to Pacifist-Greedy "
            "production otherwise."
        ),
    ),
    "always_producer": BackgroundBehaviourSpec(
        name="always_producer",
        display_name="Always-Producer",
        description=(
            "Pure free-rider. Never performs cleaning actions; otherwise identical to Pacifist-Greedy production."
        ),
    ),
    "stickler_a": BackgroundBehaviourSpec(
        name="stickler_a",
        display_name="Stickler-A",
        description=("Commits to Assembler A regardless of alternative yields. Ignores all other assemblers."),
    ),
    "stickler_b": BackgroundBehaviourSpec(
        name="stickler_b",
        display_name="Stickler-B",
        description=("Commits exclusively to Assembler B (yield 2). Ignores the remaining options."),
    ),
    "stickler_c": BackgroundBehaviourSpec(
        name="stickler_c",
        display_name="Stickler-C",
        description=("Commits exclusively to Assembler C (yield 3). Ignores the remaining options."),
    ),
    "best_responder": BackgroundBehaviourSpec(
        name="best_responder",
        display_name="Best-Responder",
        description=(
            "Observes declared assembler yields in view and selects the highest. Breaks ties by nearest distance."
        ),
    ),
    "lazy_partner": BackgroundBehaviourSpec(
        name="lazy_partner",
        display_name="Lazy-Partner",
        description=(
            "Aims to occupy dual-operator pads. Each tick has a 25% chance to "
            "abandon the pad for ≤ 3 ticks before returning, creating trust stress."
        ),
    ),
    "sustainable_zapper": BackgroundBehaviourSpec(
        name="sustainable_zapper",
        display_name="Sustainable-Zapper",
        description=(
            "Targets extractors with state.stock > 1. Refuses to take the last unit. "
            "If blocked at doorways or an enemy is adjacent, issues jam; otherwise "
            "harvests sustainably."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Scenario metadata helpers
# ---------------------------------------------------------------------------


GLOBAL_PROTOCOL_METADATA: Dict[str, Any] = {
    "episode_length": 1000,
    "seeds": [1, 2, 3],
    "reward_model": {
        "primary_stat": "heart.gained",
        "description": "Sparse reward only: +1 for each crafted heart; no auxiliary shaping.",
    },
    "primary_metric": {
        "name": "focal_per_capita_reward",
        "description": "Episode mean of focal agents' rewards, averaged over seeds.",
    },
    "secondary_metrics": [
        {
            "name": "background_per_capita_reward",
            "description": "Mean reward across background agents (per-episode, seed-averaged).",
        },
        {
            "name": "focal_reward_gini",
            "description": "Gini inequality index computed on focal-agent rewards.",
        },
    ],
    "reporting": {
        "resident_label": "resident",
        "visitor_label": "visitor",
        "success_field": "passes_success_criteria",
    },
}


BackgroundMix = Tuple[str, int]


class _CollectiveActionMissionBase(Mission):
    """Shared glue for collective-action evaluation missions."""

    site: Site = EVALS

    # Scenario metadata ------------------------------------------------------------------
    map_name: str = Field()
    scenario_name: str = Field()
    scenario_mode: str = Field(description="Either 'resident' or 'visitor'.")
    focal_agents: int = Field(ge=1)
    background_mix: Tuple[BackgroundMix, ...] = Field(default_factory=tuple)
    externality_metrics: Tuple[Mapping[str, Any], ...] = Field(default_factory=tuple)
    success_criteria: Mapping[str, Any] = Field(default_factory=dict)
    scenario_notes: Mapping[str, Any] = Field(default_factory=dict)

    def _total_agents(self) -> int:
        return self.focal_agents + sum(count for _, count in self.background_mix)

    # ------------------------------------------------------------------
    # Mission integration
    # ------------------------------------------------------------------

    def instantiate(
        self,
        map_builder,
        num_cogs: int,
        variant: MissionVariant | None = None,
        *,
        cli_override: bool = False,
    ) -> "Mission":
        forced_map = get_map(self.map_name)
        mission = super().instantiate(
            forced_map,
            num_cogs=self._total_agents(),
            variant=variant,
            cli_override=True,
        )

        mission.num_cogs = self._total_agents()

        def _post(cfg: MettaGridConfig) -> None:
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = GLOBAL_PROTOCOL_METADATA["episode_length"]

            # Enforce sparse reward model across all scenarios.
            agent_cfg: AgentConfig = cfg.game.agent
            agent_cfg.rewards = AgentRewards(stats={"heart.gained": 1.0})

            params = dict(cfg.game.params or {})
            params["collective_action"] = {
                "global_protocol": GLOBAL_PROTOCOL_METADATA,
                "scenario": self._scenario_metadata(),
            }
            cfg.game.params = params

            self.configure_env(cfg)

        return _add_make_env_modifier(mission, _post)

    # ------------------------------------------------------------------
    # Metadata generation helpers
    # ------------------------------------------------------------------

    def _scenario_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.scenario_name,
            "mode": self.scenario_mode,
            "focal_agents": self.focal_agents,
            "background_agents": self._background_metadata(),
            "externality_metrics": list(self.externality_metrics),
            "success_criteria": dict(self.success_criteria),
            "notes": dict(self.scenario_notes),
        }

    def _background_metadata(self) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for behaviour, count in self.background_mix:
            spec = BACKGROUND_BEHAVIOURS.get(behaviour)
            entry = {"behaviour": behaviour, "count": count}
            if spec is not None:
                entry["display_name"] = spec.display_name
                entry["description"] = spec.description
            metadata.append(entry)
        return metadata

    def configure_env(self, cfg: MettaGridConfig) -> None:
        """Hook for subclasses to mutate the environment configuration."""


# ---------------------------------------------------------------------------
# Mission 1 — Sustainable Mining & Territory Defense
# ---------------------------------------------------------------------------


SUSTAINABLE_BASE_NOTES: Mapping[str, Any] = {
    "map_layout": {
        "carbon": "West wing behind a short corridor.",
        "oxygen": "East wing behind a short corridor.",
        "germanium": "North alcove just above the assembler.",
        "silicon": "South alcove just below the assembler.",
    },
    "extractor_settings": {
        "carbon_max_uses": 18,
        "oxygen_max_uses": 12,
        "germanium_max_uses": 6,
        "silicon_max_uses": 8,
        "shared_cooldown_ticks": 6,
    },
    "team_guidance": (
        "Split roles across the four wings, keep the assembler fed, and leave a buffer so stocks can recover."
    ),
}


class CollectiveSustainableLoop(_CollectiveActionMissionBase):
    name: str = "collective_sustainable_loop"
    description: str = "Four-agent sustainability loop: fan out to each extractor, rotate before stocks run dry."
    map_name: str = "evals/collective_action_sustainable_base.map"
    scenario_name: str = "Sustainable Loop"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {
            "name": "hearts_crafted",
            "description": "HEARTs completed by the focal agents.",
        },
        {
            "name": "carbon_uses",
            "description": "Total carbon extractor activations (all teams).",
        },
        {
            "name": "oxygen_uses",
            "description": "Total oxygen extractor activations (all teams).",
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "carbon_use_ceiling": 18,
        "oxygen_use_ceiling": 12,
    }
    scenario_notes: Mapping[str, Any] = SUSTAINABLE_BASE_NOTES

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        carbon = cfg.game.objects.get("carbon_extractor")
        oxygen = cfg.game.objects.get("oxygen_extractor")
        germanium = cfg.game.objects.get("germanium_extractor")
        silicon = cfg.game.objects.get("silicon_extractor")

        for extractor, max_uses in (
            (carbon, 18),
            (oxygen, 12),
            (germanium, 6),
            (silicon, 8),
        ):
            if extractor is None:
                continue
            if hasattr(extractor, "max_uses"):
                extractor.max_uses = max_uses
            if getattr(extractor, "protocols", None):
                for protocol in extractor.protocols:
                    protocol.cooldown = max(protocol.cooldown, 6)

        # Reset communal chest to empty to force the loop
        chest = cfg.game.objects.get("chest")
        if chest is not None and hasattr(chest, "initial_inventory"):
            chest.initial_inventory = 0


SUSTAINABLE_PARTNERSHIP_NOTES: Mapping[str, Any] = {
    **SUSTAINABLE_BASE_NOTES,
    "clipped_extractors": ["carbon", "oxygen"],
    "starter_tools": {"decoder_per_agent": 1, "modulator_per_agent": 1},
    "team_guidance": (
        "Two agents handle unclipping and feeder extractors while the others keep germanium/silicon stocked "
        "and craft HEARTs in batches."
    ),
}


class CollectiveUnclipStartup(CollectiveSustainableLoop):
    name: str = "collective_unclip_startup"
    description: str = "Clipped basics require coordinated unclipping before the team can run the loop."
    map_name: str = "evals/collective_action_sustainable_partnership.map"
    scenario_name: str = "Unclip Startup"
    background_mix: Tuple[BackgroundMix, ...] = ()
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 2,
        "unclip_success_min": 2,
    }
    scenario_notes: Mapping[str, Any] = SUSTAINABLE_PARTNERSHIP_NOTES

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        super().configure_env(cfg)
        for name, extractor in cfg.game.objects.items():
            if name.startswith(("carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor")):
                if hasattr(extractor, "start_clipped"):
                    extractor.start_clipped = True

        agent_cfg = cfg.game.agent
        agent_cfg.initial_inventory = dict(agent_cfg.initial_inventory)
        agent_cfg.initial_inventory["decoder"] = max(agent_cfg.initial_inventory.get("decoder", 0), 1)
        agent_cfg.initial_inventory.pop("modulator", None)


# ---------------------------------------------------------------------------
# Mission 2 — Divide and Deliver
# ---------------------------------------------------------------------------


SPLIT_NOTES: Mapping[str, Any] = {
    "map_layout": {
        "corners": "Carbon, oxygen, germanium, and silicon sit in the four corners of the map.",
        "assembly": "Charger, assembler, and communal chest occupy the central plaza.",
    },
    "team_guidance": (
        "Split into corner specialists and runners; keep the center stocked while respecting extractor cooldowns."
    ),
    "extractor_settings": {
        "corner_distance": "~10 tiles from center to each corner extractor.",
        "max_uses": {"carbon": 16, "oxygen": 16, "germanium": 10, "silicon": 10},
        "cooldown_ticks": 8,
    },
}


class CollectiveSplitHarvest(_CollectiveActionMissionBase):
    name: str = "collective_split_harvest"
    description: str = "Divide the squad across corner extractors and shuttle resources back to center."
    map_name: str = "evals/collective_action_split.map"
    scenario_name: str = "Corner Split"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs completed at the assembler."},
        {"name": "corner_visits", "description": "Number of corner extractor activations."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "corner_extractors_visited": 4,
    }
    scenario_notes: Mapping[str, Any] = SPLIT_NOTES

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        extractor_targets = {
            "carbon_extractor": 16,
            "oxygen_extractor": 16,
            "germanium_extractor": 10,
            "silicon_extractor": 10,
        }
        for name, uses in extractor_targets.items():
            extractor = cfg.game.objects.get(name)
            if extractor is None:
                continue
            if hasattr(extractor, "max_uses"):
                extractor.max_uses = uses
            if getattr(extractor, "protocols", None):
                for protocol in extractor.protocols:
                    protocol.cooldown = max(protocol.cooldown, 8)

        # Empty the communal chest and standardize starting inventory
        chest = cfg.game.objects.get("chest")
        if chest is not None and hasattr(chest, "initial_inventory"):
            chest.initial_inventory = 0
        agent_cfg = cfg.game.agent
        agent_cfg.initial_inventory = {"energy": 100}


class CollectiveCornerCircuit(_CollectiveActionMissionBase):
    name: str = "collective_corner_circuit"
    description: str = (
        "Long-haul circuit: silicon and germanium recharge slowly, so agents must stagger trips and share the charger."
    )
    map_name: str = "evals/collective_action_split.map"
    scenario_name: str = "Corner Circuit"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs completed by the team."},
        {"name": "charger_visits", "description": "Trips to the central charger."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 2,
        "silicon_cycles": 4,
        "germanium_cycles": 4,
    }
    scenario_notes: Mapping[str, Any] = {
        "map_layout": SPLIT_NOTES["map_layout"],
        "team_guidance": (
            "Plan a relay so one pair harvests silicon/germanium while the other pair ferries carbon/oxygen "
            "and keeps the charger free."
        ),
        "extractor_settings": {
            "silicon_cooldown": 12,
            "germanium_cooldown": 12,
            "carbon_cooldown": 6,
            "oxygen_cooldown": 8,
        },
        "energy_guidance": "Agents begin at 70 energy, so charger usage must be scheduled.",
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        tuning = {
            "carbon_extractor": 6,
            "oxygen_extractor": 8,
            "germanium_extractor": 12,
            "silicon_extractor": 12,
        }
        for name, cooldown in tuning.items():
            extractor = cfg.game.objects.get(name)
            if extractor is None:
                continue
            if hasattr(extractor, "max_uses"):
                extractor.max_uses = 0  # unlimited but rely on cooldown
            if getattr(extractor, "protocols", None):
                for protocol in extractor.protocols:
                    protocol.cooldown = max(protocol.cooldown, cooldown)

        chest = cfg.game.objects.get("chest")
        if chest is not None and hasattr(chest, "initial_inventory"):
            chest.initial_inventory = 0

        agent_cfg = cfg.game.agent
        agent_cfg.initial_inventory = {"energy": 70}


DECLIPPING_NOTES: Mapping[str, Any] = {
    "map_layout": {
        "assembler": "Central hub with charger west and chest east for short handoffs.",
        "extractors": "Five of each resource are scattered in arcs across the map, all starting clipped.",
    },
    "role_guidance": (
        "Agents 0-1 begin with decoder stacks. They should fan out to unclip extractors while teammates gather"
        " and ferry resources back to the assembler."
    ),
    "extractor_settings": {
        "count_per_resource": 5,
        "start_clipped": True,
    },
}


class _CollectiveDeclippingBase(_CollectiveActionMissionBase):
    decoders_per_specialist: int = 10
    specialists: int = 2
    assembler_chorus: int = 2

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        # Ensure every extractor instance starts clipped and prune placeholder clipped_* entries.
        for name in list(cfg.game.objects.keys()):
            obj = cfg.game.objects[name]
            if name.startswith("clipped_") and name.endswith("_extractor"):
                cfg.game.objects.pop(name)
                continue
            if name.endswith("_extractor") and hasattr(obj, "start_clipped"):
                obj.start_clipped = True

        assembler = next((o for o in cfg.game.objects.values() if isinstance(o, AssemblerConfig)), None)
        if assembler is not None:
            self._configure_assembler(assembler)

        base_agent = cfg.game.agent
        base_inventory = dict(base_agent.initial_inventory)
        base_inventory.setdefault("energy", 100)
        base_inventory.pop("decoder", None)
        base_agent.initial_inventory = base_inventory

        total_agents = self._total_agents()
        agents: List[AgentConfig] = []
        for idx in range(total_agents):
            agent_cfg = base_agent.model_copy(deep=True)
            inventory = dict(agent_cfg.initial_inventory)
            if idx < min(self.specialists, total_agents):
                inventory["decoder"] = max(inventory.get("decoder", 0), self.decoders_per_specialist)
            else:
                inventory.pop("decoder", None)
            agent_cfg.initial_inventory = inventory
            agents.append(agent_cfg)
        cfg.game.agents = agents

        # Reintroduce clipped_* placeholders so the renderer finds atlas sprites.
        if cfg.game._resolved_type_ids:
            cfg.game._resolved_type_ids = False

        for resource in ("carbon", "oxygen", "germanium", "silicon"):
            base_name = f"{resource}_extractor"
            clipped_name = f"{resource}_extractor.clipped"
            extractor = cfg.game.objects.get(base_name)
            if extractor is not None and clipped_name not in cfg.game.objects:
                clone = extractor.model_copy()
                clone.name = clipped_name
                clone.type_id = None
                cfg.game.objects[clipped_name] = clone

    def _configure_assembler(self, assembler: AssemblerConfig) -> None:
        minimal_inputs = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3, "energy": 2}
        updated: List[ProtocolConfig] = []
        heart_protocol_applied = False
        for proto in assembler.protocols:
            if proto.output_resources.get("heart", 0) > 0:
                if heart_protocol_applied:
                    continue
                updated.append(
                    proto.model_copy(
                        update={
                            "input_resources": minimal_inputs,
                            "output_resources": {"heart": 1},
                            "cooldown": max(1, proto.cooldown),
                            "vibes": ["heart"] * self.assembler_chorus,
                        }
                    )
                )
                heart_protocol_applied = True
            else:
                updated.append(proto)

        if not heart_protocol_applied:
            updated.insert(
                0,
                ProtocolConfig(
                    vibes=["heart"] * self.assembler_chorus,
                    input_resources=minimal_inputs,
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            )

        assembler.protocols = updated


class CollectiveDeclippingFour(_CollectiveDeclippingBase):
    name: str = "collective_declipping_four"
    description: str = "Four-agent declipping relay: two specialists free the economy while two harvest."
    map_name: str = "evals/collective_action_declippers.map"
    scenario_name: str = "Declipping Relay (4)"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs completed after the declipping phase."},
        {"name": "extractors_unclipped", "description": "Count of extractor unclipping events."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 2,
        "unclipped_fraction_min": 1.0,
    }
    scenario_notes: Mapping[str, Any] = DECLIPPING_NOTES


class CollectiveDeclippingSix(_CollectiveDeclippingBase):
    name: str = "collective_declipping_six"
    description: str = "Six-agent variant emphasising division of labour once extractors are online."
    map_name: str = "evals/collective_action_declippers.map"
    scenario_name: str = "Declipping Relay (6)"
    scenario_mode: str = "resident"
    focal_agents: int = 6
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced after coordination."},
        {"name": "extractors_unclipped", "description": "Total unclipping interactions across all resources."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "unclipped_fraction_min": 1.0,
    }
    scenario_notes: Mapping[str, Any] = {
        **DECLIPPING_NOTES,
        "coordination": "Extra agents should rotate between harvesting and delivery once all extractors are active.",
    }


class CollectiveDeclippingEight(_CollectiveDeclippingBase):
    name: str = "collective_declipping_eight"
    description: str = "Eight-agent large-team declipping stressing routing discipline across the scattered map."
    map_name: str = "evals/collective_action_declippers.map"
    scenario_name: str = "Declipping Relay (8)"
    scenario_mode: str = "resident"
    focal_agents: int = 8
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEART throughput after the economy spins up."},
        {"name": "extractors_unclipped", "description": "Aggregate unclipping events."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 4,
        "unclipped_fraction_min": 1.0,
    }
    scenario_notes: Mapping[str, Any] = {
        **DECLIPPING_NOTES,
        "coordination": (
            "Two decoder specialists keep roaming to relight extractors; the remaining six split into quadrant"
            " harvesting and logistics crews."
        ),
    }


SPECIALITY_NOTES: Mapping[str, Any] = {
    "map_layout": {
        "carbon_lane": "North-west tunnels packed with carbon extractors and limited chargers.",
        "oxygen_lane": "North-east gallery with dense oxygen vents and dual chargers mid-lane.",
        "germanium_lane": "South-west caverns holding clustered germanium veins near tight corridors.",
        "silicon_lane": "South-east foundry stocked with silicon beds leading back to the assembler hub.",
    },
    "team_guidance": (
        "Each cog spawns in its speciality lane with a starter payload. Stay in-lane to keep throughput high,"
        " hand off at the central assembler, and avoid abandoning your sector unless a lane collapses."
    ),
}


class CollectiveSpecialityFocus(_CollectiveActionMissionBase):
    name: str = "collective_speciality_focus"
    description: str = "Four specialists begin seeded in unique lanes; success requires sticking with assigned roles."
    map_name: str = "evals/collective_speciality.map"
    scenario_name: str = "Lane Specialisation"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced via coordinated specialisation."},
        {
            "name": "lane_switches",
            "description": (
                "Count of extractor interactions a focal makes outside their spawn lane. "
                "Instrumentation required downstream."
            ),
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "specialist_lane_integrity": (
            "Post-hoc check: majority of each agent's extractor pulls occur within their spawn lane."
        ),
    }
    scenario_notes: Mapping[str, Any] = SPECIALITY_NOTES

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        chest = cfg.game.objects.get("chest")
        if chest is not None and hasattr(chest, "initial_inventory"):
            chest.initial_inventory = 0

        base_agent = cfg.game.agent
        base_inventory = dict(base_agent.initial_inventory)
        base_inventory.setdefault("energy", 100)
        cfg.game.agent.initial_inventory = base_inventory

        loadouts = [
            {"carbon": 1},
            {"oxygen": 1},
            {"germanium": 1},
            {"silicon": 1},
        ]

        agents: List[AgentConfig] = []
        for idx in range(self._total_agents()):
            agent_cfg = base_agent.model_copy(deep=True)
            inventory = dict(base_inventory)
            if idx < len(loadouts):
                for resource, amount in loadouts[idx].items():
                    inventory[resource] = amount
            agent_cfg.initial_inventory = inventory
            agents.append(agent_cfg)

        cfg.game.agents = agents


GERANIUM_NOTES: Mapping[str, Any] = {
    "map_layout": {
        "germanium_field": "Large northern trench of germanium extractors with a single corridor to the hub.",
        "support_extractors": "Carbon, silicon, and oxygen clusters sit closer to the assembler for quick refills.",
        "assembler_plaza": "Mid-map assembler with adjacent chest; two operators can remain stationed there.",
    },
    "team_guidance": (
        "Two operators stay on assembler duty (chorus length two) while the remaining specialists run a geranium"
        " relay from the northern vein back to base. Keep trips staggered to avoid idle assembler time."
    ),
}


class CollectiveGeraniumRelay(_CollectiveActionMissionBase):
    name: str = "collective_geranium_relay"
    description: str = (
        "Geranium bottleneck scenario: two agents staff the assembler while teammates maintain a constant ore relay."
    )
    map_name: str = "evals/collective_geranium.map"
    scenario_name: str = "Geranium Relay"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced once the relay is humming."},
        {
            "name": "germanium_runs",
            "description": "Number of germanium extractor activations across the team (proxy for relay pressure).",
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "germanium_runs_min": 12,
        "assembler_idle_ceiling": "Instrumentation: assembler idle < 20% of ticks post-first heart.",
    }
    scenario_notes: Mapping[str, Any] = GERANIUM_NOTES

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        assembler = next((o for o in cfg.game.objects.values() if isinstance(o, AssemblerConfig)), None)
        if assembler is not None:
            minimal_inputs = {"carbon": 2, "oxygen": 2, "germanium": 2, "silicon": 2, "energy": 2}
            updated: List[ProtocolConfig] = []
            heart_protocol_applied = False
            for proto in assembler.protocols:
                if proto.output_resources.get("heart", 0) > 0:
                    if heart_protocol_applied:
                        continue
                    updated.append(
                        proto.model_copy(
                            update={
                                "input_resources": minimal_inputs,
                                "output_resources": {"heart": 1},
                                "cooldown": max(1, proto.cooldown),
                                "vibes": ["heart", "heart"],
                            }
                        )
                    )
                    heart_protocol_applied = True
                else:
                    updated.append(proto)
            if not heart_protocol_applied:
                updated.insert(
                    0,
                    ProtocolConfig(
                        vibes=["heart", "heart"],
                        input_resources=minimal_inputs,
                        output_resources={"heart": 1},
                        cooldown=1,
                    ),
                )
            assembler.protocols = updated

        for name, obj in cfg.game.objects.items():
            if not hasattr(obj, "max_uses"):
                continue
            if name.startswith("germanium_extractor"):
                obj.max_uses = max(3, getattr(obj, "max_uses", 0) or 3)
                if getattr(obj, "protocols", None):
                    for proto in obj.protocols:
                        proto.cooldown = max(proto.cooldown, 5)
            elif name.startswith(("carbon_extractor", "oxygen_extractor", "silicon_extractor")):
                obj.max_uses = 0

        chest = cfg.game.objects.get("chest")
        if chest is not None and hasattr(chest, "initial_inventory"):
            chest.initial_inventory = 0

        base_agent = cfg.game.agent
        base_inventory = dict(base_agent.initial_inventory)
        base_inventory.setdefault("energy", 100)
        base_agent.initial_inventory = base_inventory

        loadouts = [
            {"carbon": 2, "oxygen": 2, "silicon": 1},
            {"carbon": 1, "oxygen": 1, "silicon": 1},
            {},
            {},
        ]

        agents: List[AgentConfig] = []
        for idx in range(self._total_agents()):
            agent_cfg = base_agent.model_copy(deep=True)
            inventory = dict(base_inventory)
            if idx < len(loadouts):
                for resource, amount in loadouts[idx].items():
                    inventory[resource] = amount
            agent_cfg.initial_inventory = inventory
            agents.append(agent_cfg)

        cfg.game.agents = agents


class CollectiveUnclipScatter(_CollectiveDeclippingBase):
    name: str = "collective_unclip_scatter"
    description: str = (
        "Clipped extractors scattered across the arena; two decoder specialists fan out while teammates ferry loot."
    )
    map_name: str = "evals/collective_unclip.map"
    scenario_name: str = "Unclip Scatter"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    decoders_per_specialist: int = 20
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced after the arena is fully online."},
        {
            "name": "extractors_unclipped",
            "description": "Total unclipping interactions across all extractor types.",
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "unclipped_fraction_min": 1.0,
        "decoder_usage_min": "At least 15 decoder spends recorded across the team.",
    }
    scenario_notes: Mapping[str, Any] = {
        "map_layout": {
            "scatter": (
                "Ten extractors of each resource are spread across a wide arena with chargers embedded throughout."
            ),
            "hub": "Assembler and chest sit centrally; decoder specialists start adjacent for quick dispatch.",
        },
        "role_guidance": (
            "Agents 1-2 carry decoder stacks and should sprint to new extractors; agents 3-4 trail to harvest once"
            " stations come online."
        ),
    }


class CollectiveUnclipBuddy(_CollectiveDeclippingBase):
    name: str = "collective_unclip_buddy"
    description: str = "Buddy unclipping: one decoder specialist unlocks the map while teammates harvest immediately."
    map_name: str = "evals/collective_unclip_buddy.map"
    scenario_name: str = "Unclip Buddy"
    scenario_mode: str = "resident"
    focal_agents: int = 2
    background_mix: Tuple[BackgroundMix, ...] = ()
    specialists: int = 1
    decoders_per_specialist: int = 20
    assembler_chorus: int = 1
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced once the lone specialist keeps stations online."},
        {
            "name": "decoder_spend",
            "description": "Decoder usages recorded across the episode (proxy for unclipping diligence).",
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 2,
        "decoder_spend_min": 10,
        "unclipped_fraction_min": 1.0,
    }
    scenario_notes: Mapping[str, Any] = {
        "map_layout": {
            "lanes": "Specialist starts adjacent to silicon while partner spawns near germanium; corridor stays tight.",
        },
        "role_guidance": (
            "Agent 1 carries a decoder stack to roam and unlock extractors; Agent 2 should trail closely and harvest"
            " immediately before clips re-spread."
        ),
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        super().configure_env(cfg)

        assembler = next((o for o in cfg.game.objects.values() if isinstance(o, AssemblerConfig)), None)
        if assembler is not None:
            heart_inputs = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3, "energy": 2}
            decoder_inputs = {"carbon": 1}
            additional = [
                proto
                for proto in assembler.protocols
                if proto.output_resources.get("heart", 0) == 0 and proto.output_resources.get("decoder", 0) == 0
            ]
            assembler.protocols = [
                ProtocolConfig(
                    vibes=["heart"],
                    input_resources=heart_inputs,
                    output_resources={"heart": 1},
                    cooldown=0,
                ),
                ProtocolConfig(
                    vibes=[],
                    input_resources=decoder_inputs,
                    output_resources={"decoder": 1},
                    cooldown=0,
                ),
                *additional,
            ]

        if cfg.game.clipper is not None:
            cfg.game.clipper.clip_rate = 0.3

        # Enforce explicit inventories: specialist (index 0) gets decoder stack, partner does not.
        decoder_cap = max(self.decoders_per_specialist, 20)
        base_limits = dict(cfg.game.agent.resource_limits)
        base_limits["decoder"] = decoder_cap
        cfg.game.agent.resource_limits = base_limits

        updated_agents: List[AgentConfig] = []
        for idx, agent_cfg in enumerate(cfg.game.agents):
            inventory = dict(agent_cfg.initial_inventory)
            limits = dict(agent_cfg.resource_limits)
            if idx == 0:  # first spawn (Agent 1) carries full decoder stack
                limits["decoder"] = decoder_cap
                inventory["decoder"] = self.decoders_per_specialist
            else:  # second spawn (Agent 2) should have none
                limits["decoder"] = 0
                inventory.pop("decoder", None)
            agent_cfg.initial_inventory = inventory
            agent_cfg.resource_limits = limits
            updated_agents.append(agent_cfg)
        cfg.game.agents = updated_agents

        base_inventory = dict(cfg.game.agent.initial_inventory)
        base_inventory.pop("decoder", None)
        cfg.game.agent.initial_inventory = base_inventory

        for name, obj in cfg.game.objects.items():
            if name.endswith("_extractor") and hasattr(obj, "start_clipped"):
                obj.start_clipped = name.startswith("silicon")

        # Reintroduce clipped-* placeholders so the renderer finds atlas sprites.
        for resource in ("carbon", "oxygen", "germanium", "silicon"):
            base_name = f"{resource}_extractor"
            clipped_name = f"{resource}_extractor.clipped"
            extractor = cfg.game.objects.get(base_name)
            if extractor is not None and clipped_name not in cfg.game.objects:
                clone = extractor.model_copy()
                clone.name = clipped_name
                clone.type_id = None  # allow type-id reallocation to avoid clashes
                cfg.game.objects[clipped_name] = clone


class CollectiveTrafficJam(_CollectiveActionMissionBase):
    name: str = "collective_traffic_jam"
    description: str = "Choke-point assembler with unlimited extractors; manage traffic flow for sustained throughput."
    map_name: str = "evals/collective_traffic_jam.map"
    scenario_name: str = "Traffic Jam"
    scenario_mode: str = "resident"
    focal_agents: int = 5
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced across the extended episode."},
        {
            "name": "lane_block_ticks",
            "description": "Ticks where the central corridor contains >2 agents (requires downstream logging).",
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 6,
        "max_lane_block_ticks": "Instrumentation target: congestion over threshold < 25% of episode.",
    }
    scenario_notes: Mapping[str, Any] = {
        "layout": (
            "Two narrow corridors feed a central assembler room. Extractors are plentiful, so scheduling the queue is"
            " key."
        ),
        "guidance": (
            "Assign two agents to assembler duty and stagger runners so the corridor stays clear despite the long"
            " episode."
        ),
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        cfg.game.max_steps = 3000
        for name, obj in cfg.game.objects.items():
            if not hasattr(obj, "max_uses"):
                continue
            if name.startswith(("carbon_extractor", "oxygen_extractor", "silicon_extractor", "germanium_extractor")):
                obj.max_uses = 0
                if getattr(obj, "protocols", None):
                    for proto in obj.protocols:
                        proto.cooldown = 0


class CollectiveRoundabout(_CollectiveActionMissionBase):
    name: str = "collective_roundabout"
    description: str = "Roundabout loop: agents must circulate in sync to avoid congestion and keep assemblies flowing."
    map_name: str = "evals/collective_roundabout.map"
    scenario_name: str = "Roundabout"
    scenario_mode: str = "resident"
    focal_agents: int = 6
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced while maintaining free-flow traffic."},
        {
            "name": "lane_conflicts",
            "description": "Ticks where multiple agents occupy the same corridor tile (requires downstream logging).",
        },
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 4,
        "lane_conflicts_max": "Instrumentation target: conflicts under 20% of episode ticks.",
    }
    scenario_notes: Mapping[str, Any] = {
        "layout": "Single-file roundabout with resource alcoves feeding into the ring.",
        "guidance": (
            "Pick a shared direction (clockwise) and keep moving to avoid mutual blockages at the assembler bays."
        ),
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        heart_inputs = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3, "energy": 2}
        for name, obj in cfg.game.objects.items():
            if name.startswith("assembler") and isinstance(obj, AssemblerConfig):
                obj.protocols = [
                    ProtocolConfig(
                        vibes=["heart"],
                        input_resources=heart_inputs,
                        output_resources={"heart": 1},
                        cooldown=0,
                    )
                ]
            if name.endswith("_extractor") and hasattr(obj, "max_uses"):
                obj.max_uses = 0
                if getattr(obj, "protocols", None):
                    for proto in obj.protocols:
                        proto.cooldown = 0


class CollectiveChooseRoom(_CollectiveActionMissionBase):
    name: str = "collective_choose_room"
    description: str = "Pick a room and commit: the assembler needs the whole squad while resources stay plentiful."
    map_name: str = "evals/collective_choose_a_room.map"
    scenario_name: str = "Choose a Room"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced after the team converges."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
    }
    scenario_notes: Mapping[str, Any] = {
        "layout": "Multiple wings feed a single assembler chamber; doors encourage splitting.",
        "guidance": (
            "Coordinate to enter the same room; assembler requires all four agents while extractors never run dry."
        ),
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        heart_inputs = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3, "energy": 2}
        for name, obj in cfg.game.objects.items():
            if name.startswith("assembler") and isinstance(obj, AssemblerConfig):
                obj.protocols = [
                    ProtocolConfig(
                        vibes=["heart", "heart", "heart", "heart"],
                        input_resources=heart_inputs,
                        output_resources={"heart": 1},
                        cooldown=0,
                    )
                ]
            if name.endswith("_extractor") and hasattr(obj, "max_uses"):
                obj.start_clipped = False
                obj.max_uses = 0
                if getattr(obj, "protocols", None):
                    for proto in obj.protocols:
                        proto.cooldown = 0


class CollectiveRadial(_CollectiveActionMissionBase):
    name: str = "collective_radial"
    description: str = "Radial corridors force a room choice; assembler demands the full team while resources overflow."
    map_name: str = "evals/collective_radial.map"
    scenario_name: str = "Radial Assembly"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced once agents coordinate on a radial room."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
    }
    scenario_notes: Mapping[str, Any] = {
        "layout": "Oxygen and carbon wings oppose germanium and silicon; assembler sits at the hub.",
        "guidance": "Move as a pack to the same radial room; assembler chorus length four, stations never exhaust.",
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        heart_inputs = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3, "energy": 2}
        for name, obj in cfg.game.objects.items():
            if name.startswith("assembler") and isinstance(obj, AssemblerConfig):
                obj.protocols = [
                    ProtocolConfig(
                        vibes=["heart", "heart", "heart", "heart"],
                        input_resources=heart_inputs,
                        output_resources={"heart": 1},
                        cooldown=0,
                    )
                ]
            if name.endswith("_extractor") and hasattr(obj, "max_uses"):
                obj.start_clipped = False
                obj.max_uses = 0
                if getattr(obj, "protocols", None):
                    for proto in obj.protocols:
                        proto.cooldown = 0


class CollectiveAlcove(_CollectiveActionMissionBase):
    name: str = "collective_alcove"
    description: str = "Alcove split: resources flow freely but assemblers sit in secluded pockets waiting for runners."
    map_name: str = "evals/collective_alcove.map"
    scenario_name: str = "Alcove Logistics"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced by shuttling between alcoves."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
    }
    scenario_notes: Mapping[str, Any] = {
        "layout": "Two assemblers sit in opposite alcoves fed by long corridors; resources are plentiful throughout.",
        "guidance": (
            "Keep runners moving between alcoves; assemblers only need one agent but require all four resources."
        ),
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        heart_inputs = {"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3, "energy": 2}
        for name, obj in cfg.game.objects.items():
            if name.startswith("assembler") and isinstance(obj, AssemblerConfig):
                obj.protocols = [
                    ProtocolConfig(
                        vibes=["heart"],
                        input_resources=heart_inputs,
                        output_resources={"heart": 1},
                        cooldown=0,
                    )
                ]
            if name.endswith("_extractor") and hasattr(obj, "max_uses"):
                obj.start_clipped = False
                obj.max_uses = 0
                if getattr(obj, "protocols", None):
                    for proto in obj.protocols:
                        proto.cooldown = 0


class CollectiveUnclip(_CollectiveDeclippingBase):
    name: str = "collective_unclip"
    description: str = (
        "Baseline unclipping exercise: all extractors spawn clipped and every agent begins with a decoder."
    )
    map_name: str = "evals/collective_unclip.map"
    scenario_name: str = "Collective Unclip"
    scenario_mode: str = "resident"
    focal_agents: int = 4
    background_mix: Tuple[BackgroundMix, ...] = ()
    specialists: int = 4  # everyone receives the decoder allotment
    decoders_per_specialist: int = 20
    assembler_chorus: int = 1
    externality_metrics: Tuple[Mapping[str, Any], ...] = (
        {"name": "hearts_crafted", "description": "HEARTs produced once stations are kept online."},
        {"name": "extractors_unclipped", "description": "Total unclipping interactions across all resources."},
    )
    success_criteria: Mapping[str, Any] = {
        "minimum_hearts": 3,
        "unclipped_fraction_min": 1.0,
    }
    scenario_notes: Mapping[str, Any] = {
        "layout": "Large arena with scattered extractors and chargers; no resource limits aside from clipping.",
        "guidance": (
            "Each agent starts stocked with decoders. Fan out to keep extractors online, then regroup to sustain heart"
            " throughput."
        ),
    }

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - configuration hook
        super().configure_env(cfg)

        # Ensure everyone retains a sizable decoder stack and appropriate limit.
        decoder_cap = max(self.decoders_per_specialist, 20)
        base_limits = dict(cfg.game.agent.resource_limits)
        base_limits["decoder"] = decoder_cap
        cfg.game.agent.resource_limits = base_limits

        updated_agents: List[AgentConfig] = []
        for agent_cfg in cfg.game.agents:
            inventory = dict(agent_cfg.initial_inventory)
            limits = dict(agent_cfg.resource_limits)
            limits["decoder"] = decoder_cap
            inventory["decoder"] = self.decoders_per_specialist
            agent_cfg.initial_inventory = inventory
            agent_cfg.resource_limits = limits
            updated_agents.append(agent_cfg)
        cfg.game.agents = updated_agents

        base_inventory = dict(cfg.game.agent.initial_inventory)
        base_inventory["decoder"] = self.decoders_per_specialist
        cfg.game.agent.initial_inventory = base_inventory


COLLECTIVE_ACTION_EVALS: List[type[_CollectiveActionMissionBase]] = [
    CollectiveSustainableLoop,
    CollectiveUnclipStartup,
    CollectiveSplitHarvest,
    CollectiveCornerCircuit,
    CollectiveDeclippingFour,
    CollectiveDeclippingSix,
    CollectiveDeclippingEight,
    CollectiveSpecialityFocus,
    CollectiveGeraniumRelay,
    CollectiveUnclip,
    CollectiveUnclipScatter,
    CollectiveUnclipBuddy,
    CollectiveTrafficJam,
    CollectiveRoundabout,
    CollectiveChooseRoom,
    CollectiveRadial,
    CollectiveAlcove,
]
