"""A Cogs vs Clips version of the arena recipe - STABLE

This is meant as a basic testbed for CvC buildings / mechanics.
This recipe is automatically validated in CI and release processes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.stations import (
    CvCStationConfig,
    CvCWallConfig,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    DiscreteRandomConfig,
)
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AlignActionConfig,
    AOEEffectConfig,
    AssemblerConfig,
    ChangeVibeActionConfig,
    ChestConfig,
    CommonsChestConfig,
    CommonsConfig,
    DamageConfig,
    GameConfig,
    GlobalObsConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ProtocolConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.vibes import Vibe
from mettagrid.mapgen.mapgen import MapGen

resources = [
    "energy",
    "carbon",
    "oxygen",
    "germanium",
    "silicon",
    "heart",
    "weapon",
    "shield",
    "battery",
    "gear",
    "damage",
    "support",
]

vibes = [
    Vibe("😐", "default"),
    Vibe("⚔️", "weapon"),
    Vibe("🛡️", "shield"),
    Vibe("🔋", "battery"),
    Vibe("⚙️", "gear"),
    Vibe("❤️", "heart"),
]


class PylonConfig(GridObjectConfig):
    radius: int


class RoleStationConfig(GridObjectConfig):
    role: str


class ExtractorConfig(GridObjectConfig):
    pass


class ColonyConfig(GridObjectConfig):
    pass


def support_aoe(range: int = 10) -> AOEEffectConfig:
    """AOE effect that provides support to nearby agents."""
    return AOEEffectConfig(
        range=range,
        resource_deltas={"support": 10, "energy": 100, "damage": -100},
        members_only=True,
    )


def supply_depot_config(map_name: str, team: Optional[str] = None, has_aoe: bool = True) -> CommonsChestConfig:
    """Supply depot that receives element resources via default vibe into commons."""
    return CommonsChestConfig(
        name="supply_depot",
        map_name=map_name,
        render_symbol="📦",
        commons=team,
        vibe_transfers={"default": {"carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255}},
<<<<<<< HEAD
        aoes=[support_aoe()],
=======
        aoe=support_aoe() if has_aoe else None,
>>>>>>> 16a1dec145 (cp)
    )


def main_nexus_config(map_name: str) -> AssemblerConfig:
    """Main nexus assembler with support AOE effect."""
    return AssemblerConfig(
        name="main_nexus",
        map_name=map_name,
        render_symbol="🏛️",
        clip_immune=True,
        chest_search_distance=10,
        commons="cogs",
        protocols=[
            ProtocolConfig(
                vibes=["heart"],
                input_resources={
                    "carbon": 10,
                    "oxygen": 10,
                    "germanium": 10,
                    "silicon": 10,
                },
                output_resources={"heart": 1},
            ),
        ],
        aoes=[support_aoe()],
    )


def resource_chest_config(map_name: str, resource: str, amount: int = 100) -> ChestConfig:
    """Chest containing a resource that can be withdrawn via default vibe."""
    return ChestConfig(
        name=f"{resource}_chest",
        map_name=map_name,
        render_symbol="📦",
        inventory=InventoryConfig(
            limits={resource: ResourceLimitsConfig(limit=amount, resources=[resource])},
            initial={resource: amount},
        ),
        vibe_transfers={"default": {resource: -255}},  # negative = withdraw
    )


class CogAssemblerConfig(CvCStationConfig):
    def station_cfg(self) -> AssemblerConfig:
        # gear = [("carbon", "decoder"), ("oxygen", "modulator"), ("germanium", "scrambler"), ("silicon", "resonator")]
        return AssemblerConfig(
            name="assembler",
            render_symbol="",
            clip_immune=True,
            chest_search_distance=10,
            protocols=[
                ProtocolConfig(
                    vibes=["heart"],
                    input_resources={
                        "carbon": 10,
                        "oxygen": 10,
                        "germanium": 10,
                        "silicon": 10,
                    },
                    output_resources={"heart": 1},
                ),
            ],
        )


def make_env(num_agents: int = 10) -> MettaGridConfig:
    map_builder = MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=num_agents,
            building_coverage=0.1,
        ),
    )
    vibe_names = [vibe.name for vibe in vibes]
    game = GameConfig(
        map_builder=map_builder,
        max_steps=1000,
        num_agents=num_agents,
        resource_names=resources,
        vibe_names=vibe_names,
        global_obs=GlobalObsConfig(),
        actions=ActionsConfig(
            move=MoveActionConfig(
                consumed_resources={"energy": 3},
            ),
            noop=NoopActionConfig(),
            change_vibe=ChangeVibeActionConfig(vibes=vibes),
            align=AlignActionConfig(
                vibe="heart",
                cost={"heart": 1},
                required_resources={"support": 1},
            ),
            scramble=AlignActionConfig(
                vibe="weapon",
                set_to_none=True,
                required_resources={"weapon": 1},
            ),
        ),
        agent=AgentConfig(
            commons="cogs",
            inventory=InventoryConfig(
                limits={
                    "heart": ResourceLimitsConfig(limit=20000, resources=["heart"]),
                    "energy": ResourceLimitsConfig(limit=0, resources=["energy"], modifiers={"battery": 25}),
                    "cargo": ResourceLimitsConfig(limit=255, resources=["carbon", "oxygen", "germanium", "silicon"]),
                    "gear": ResourceLimitsConfig(
                        limit=0, resources=["weapon", "shield", "battery"], modifiers={"gear": 1}
                    ),
                    "support": ResourceLimitsConfig(limit=10, resources=["support"]),
                },
                initial={
                    "gear": 10,
                    "battery": 4,
                    "energy": 100,
                    "weapon": 1,
                    "shield": 0,
                    "silicon": 50,
                    "oxygen": 50,
                    "carbon": 50,
                    "germanium": 50,
                    "heart": 5,
                },
                regen_amounts={
                    "default": {
                        "energy": 1,
                        "damage": 1,
                        "support": -1,
                    },
                },
            ),
            rewards=AgentRewards(
                inventory={
                    "heart": 1,
                    "carbon": 0.001,
                    "oxygen": 0.001,
                    "germanium": 0.001,
                    "silicon": 0.001,
                    "weapon": 0.01,
                    "shield": 0.01,
                    "battery": 0.01,
                    "gear": 0.1,
                },
                commons_inventory={
                    "carbon": 0.01,
                    "oxygen": 0.01,
                    "germanium": 0.01,
                    "silicon": 0.01,
                },
            ),
            damage=DamageConfig(
                threshold={"damage": 200},
                resources={"battery": 1, "weapon": 0, "shield": 0},  # resource -> minimum (won't go below)
            ),
            freeze_duration=5,
        ),
        inventory_regen_interval=1,
        objects={
            "wall": CvCWallConfig().station_cfg(),
            "assembler": main_nexus_config("assembler"),
            "charger": supply_depot_config("charger", "clips"),
            "chest": supply_depot_config("chest", "cogs", has_aoe=False),
            "carbon_extractor": resource_chest_config("carbon_extractor", "carbon"),
            "oxygen_extractor": resource_chest_config("oxygen_extractor", "oxygen"),
            "germanium_extractor": resource_chest_config("germanium_extractor", "germanium"),
            "silicon_extractor": resource_chest_config("silicon_extractor", "silicon"),
        },
        commons=[
            CommonsConfig(
                name="cogs",
                inventory=InventoryConfig(
                    limits={
                        "resources": ResourceLimitsConfig(
                            limit=10000, resources=["carbon", "oxygen", "germanium", "silicon"]
                        ),
                        "hearts": ResourceLimitsConfig(limit=65535, resources=["heart"]),
                    },
                    initial={
                        "carbon": 0,
                        "oxygen": 0,
                        "germanium": 0,
                        "silicon": 0,
                        "heart": 0,
                    },
                ),
            ),
            CommonsConfig(
                name="clips",
            ),
        ],
    )

    env = MettaGridConfig(game=game)
    return env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or make_env()

    arena_tasks = cc.bucketed(arena_env)

    # for item in ["ore_red", "battery_red", "laser", "armor"]:
    #     arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
    #     arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.max_steps", [1000, 5000, 10000])
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.energy", [7, 8, 9, 10])
    arena_tasks.add_bucket("game.actions.attack.weapon_resources.weapon", [9, 10, 11])
    arena_tasks.add_bucket("game.actions.attack.armor_resources.shield", [12, 13, 14])
    arena_tasks.add_bucket("game.agent.inventory.initial.weapon", [0, 1, 2, 3])
    arena_tasks.add_bucket("game.agent.inventory.initial.shield", [0, 1, 2, 3])
    arena_tasks.add_bucket("game.agent.inventory.initial.battery", [3, 4, 5, 6])
    arena_tasks.add_bucket("game.agent.inventory.regen_amounts.default.damage", [1, 2, 3])
    arena_tasks.add_bucket("game.agent.inventory.regen_amounts.default.energy", [1, 2, 3])

    if algorithm_config is None:
        # algorithm_config = LearningProgressConfig(
        #     use_bidirectional=True,
        #     ema_timescale=0.001,
        #     exploration_bonus=0.1,
        #     max_memory_tasks=2000,
        #     max_slice_axes=4,
        #     enable_detailed_slice_logging=True,
        # )
        algorithm_config = DiscreteRandomConfig()

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    basic_env = env or make_env()

    return [
        SimulationConfig(suite="cog_arena", name="basic", env=basic_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
    resolved_curriculum = curriculum or make_curriculum()
    trainer_cfg = TrainerConfig()
    evaluator_cfg = EvaluatorConfig(simulations=simulations())

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    resolved_policy_uris: str | list[str]
    if policy_uris is None:
        resolved_policy_uris = []
    elif isinstance(policy_uris, str):
        resolved_policy_uris = policy_uris
    else:
        resolved_policy_uris = list(policy_uris)
    return EvaluateTool(
        simulations=simulations(),
        policy_uris=resolved_policy_uris,
    )


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with a policy."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Generate replay from a policy."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)
