"""A Cogs vs Clips version of the arena recipe - STABLE

This is meant as a basic testbed for CvC buildings / mechanics.
This recipe is automatically validated in CI and release processes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.stations import (
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
    ActivationHandler,
    ActorCommonsHas,
    ActorHas,
    AgentConfig,
    AgentRewards,
    Align,
    AOEEffectConfig,
    AssemblerConfig,
    ChangeVibeActionConfig,
    ChestConfig,
    ClearInventoryMutation,
    CommonsChestConfig,
    CommonsConfig,
    CommonsDeposit,
    CommonsWithdraw,
    GameConfig,
    GlobalObsConfig,
    HealthConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    RemoveAlignment,
    ResourceLimitsConfig,
    TargetCommonsUpdate,
    UpdateActor,
    Withdraw,
    isAligned,
    isEnemy,
    isNeutral,
)
from mettagrid.config.vibes import Vibe
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHubConfig

gear = ["aligner", "scrambler", "miner", "scout"]
elements = ["oxygen", "carbon", "germanium", "silicon"]


def neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


HEART_COST = {e: 1 for e in elements}

ALIGN_COST = {"heart": 1}
SCRAMBLE_COST = {"heart": 1}

gear_costs = {
    "aligner": {"carbon": 3, "oxygen": 1, "germanium": 1, "silicon": 1},
    "scrambler": {"carbon": 1, "oxygen": 3, "germanium": 1, "silicon": 1},
    "miner": {"carbon": 1, "oxygen": 1, "germanium": 3, "silicon": 1},
    "scout": {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 3},
}

resources = [
    "energy",
    "heart",
    "hp",
    "influence",
    *elements,
    *gear,
]

vibes = [
    Vibe("😐", "default"),
    Vibe("❤️", "heart"),
]


def influence_aoe(range: int = 10) -> AOEEffectConfig:
    """AOE effect that provides influence to nearby agents in the same commons."""
    return AOEEffectConfig(
        range=range,
        resource_deltas={"influence": 10, "energy": 100, "hp": 100},
        filters=[isAligned()],
    )


def attack_aoe(range: int = 10) -> AOEEffectConfig:
    """AOE effect that attacks nearby agents not in the same commons."""
    return AOEEffectConfig(
        range=range,
        resource_deltas={"hp": -1, "influence": -100},
        filters=[isEnemy()],
    )


def supply_depot_config(map_name: str, team: Optional[str] = None, has_aoe: bool = True) -> CommonsChestConfig:
    """Supply depot that receives element resources via default vibe into commons."""
    aoes = [influence_aoe(), attack_aoe()]
    return CommonsChestConfig(
        name="supply_depot",
        map_name=map_name,
        render_symbol="📦",
        commons=team,
        aoes=aoes if has_aoe else [],
        handlers=[
            ActivationHandler(
                name="deposit",
                filters=[isAligned()],
                mutations=[CommonsDeposit({resource: 100 for resource in elements})],
            ),
            ActivationHandler(
                name="align",
                filters=[isNeutral(), ActorHas({"aligner": 1, "influence": 1, **ALIGN_COST})],
                mutations=[UpdateActor(neg(ALIGN_COST)), Align()],
            ),
            # Scramble handler: remove this depot's commons alignment (only if aligned)
            ActivationHandler(
                name="scramble",
                filters=[isEnemy(), ActorHas({"scrambler": 1, **SCRAMBLE_COST})],
                mutations=[RemoveAlignment(), UpdateActor(neg(SCRAMBLE_COST))],
            ),
        ],
    )


def nexus(map_name: str) -> AssemblerConfig:
    """Main nexus assembler with influence AOE effect."""
    return AssemblerConfig(
        name="main_nexus",
        map_name=map_name,
        render_symbol="🏛️",
        clip_immune=True,
        chest_search_distance=10,
        commons="cogs",
        aoes=[influence_aoe(), attack_aoe()],
        handlers=[
            ActivationHandler(
                name="deposit",
                filters=[isAligned()],
                mutations=[CommonsDeposit({resource: 100 for resource in elements})],
            ),
        ],
    )


def chest() -> AssemblerConfig:
    """Main nexus assembler with influence AOE effect."""
    return AssemblerConfig(
        name="chest",
        map_name="chest",
        render_symbol="📦",
        commons="cogs",
        handlers=[
            ActivationHandler(
                name="get_heart",
                filters=[isAligned()],
                mutations=[CommonsWithdraw({"heart": 1})],
            ),
            ActivationHandler(
                name="make_heart",
                filters=[isAligned(), ActorCommonsHas(HEART_COST)],
                mutations=[
                    TargetCommonsUpdate(neg(HEART_COST)),
                    UpdateActor({"heart": 1}),
                ],
            ),
        ],
    )


def extractor(resource: str, amount: int = 100) -> ChestConfig:
    """Chest (mine) containing a resource with alignment-based extraction.

    - If aligned to actor's commons: add 10 resource to commons + cooldown
    - If not aligned: add to agent's inventory
    """
    return ChestConfig(
        name=f"{resource}_chest",
        map_name=f"{resource}_extractor",
        render_symbol="📦",
        inventory=InventoryConfig(
            limits={resource: ResourceLimitsConfig(limit=amount, resources=[resource])},
            initial={resource: amount},
        ),
        handlers=[
            # Extract handler: transfer resource from mine to agent
            ActivationHandler(
                name="extract_big",
                filters=[ActorHas({"miner": 1})],
                mutations=[Withdraw({resource: 10})],
            ),
            ActivationHandler(
                name="extract_small",
                filters=[],
                mutations=[Withdraw({resource: 1})],
            ),
        ],
    )


# Gear station symbols
GEAR_SYMBOLS = {
    "aligner": "🔗",
    "scrambler": "🌀",
    "miner": "⛏️",
    "scout": "🔭",
}


def gear_station(gear_type: str) -> AssemblerConfig:
    """Create a gear station that clears all gear and adds the specified gear type."""
    return AssemblerConfig(
        name=f"{gear_type}_station",
        map_name=f"{gear_type}_station",
        render_symbol=GEAR_SYMBOLS.get(gear_type, "⚙️"),
        commons="cogs",
        handlers=[
            ActivationHandler(
                name="keep_gear",
                filters=[isAligned(), ActorHas({gear_type: 1})],
                mutations=[],
            ),
            ActivationHandler(
                name="change_gear",
                filters=[isAligned(), ActorCommonsHas(gear_costs[gear_type])],
                mutations=[
                    ClearInventoryMutation(target="actor", limit_name="gear"),
                    TargetCommonsUpdate(neg(gear_costs[gear_type])),
                    UpdateActor({gear_type: 1}),
                ],
            ),
        ],
    )


def make_env(num_agents: int = 10) -> MettaGridConfig:
    # Configure hub with gear stations
    hub_config = BaseHubConfig(
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=7,
        stations=[
            "aligner_station",
            "scrambler_station",
            "miner_station",
            "scout_station",
        ],
    )
    map_builder = MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=num_agents,
            building_coverage=0.1,
            hub=hub_config,
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
        ),
        agent=AgentConfig(
            commons="cogs",
            inventory=InventoryConfig(
                limits={
                    "gear": ResourceLimitsConfig(limit=1, resources=gear),
                    "hp": ResourceLimitsConfig(limit=100, resources=["hp"], modifiers={"scout": 400, "scrambler": 200}),
                    "heart": ResourceLimitsConfig(limit=10, resources=["heart"]),
                    "energy": ResourceLimitsConfig(limit=10, resources=["energy"], modifiers={"scout": 100}),
                    "cargo": ResourceLimitsConfig(limit=4, resources=elements, modifiers={"miner": 40}),
                    "influence": ResourceLimitsConfig(limit=0, resources=["influence"], modifiers={"aligner": 20}),
                },
                initial={
                    "energy": 100,
                    "heart": 5,
                    "hp": 50,
                },
                regen_amounts={
                    "default": {
                        "energy": 1,
                        "hp": -1,
                        "influence": -1,
                    },
                },
            ),
            rewards=AgentRewards(
                commons_inventory={
                    # "carbon": 0.001,
                    # "oxygen": 0.001,
                    # "germanium": 0.001,
                    # "silicon": 0.001,
                },
                commons_stats={
                    "aligned.charger.held": 0.001,
                },
            ),
            health=HealthConfig(
                health_resource="hp",
                on_damage=[
                    ClearInventoryMutation(target="actor", limit_name="gear"),
                    UpdateActor({"hp": 500}),
                ],
            ),
        ),
        inventory_regen_interval=1,
        objects={
            "wall": CvCWallConfig().station_cfg(),
            "assembler": nexus("assembler"),
            "charger": supply_depot_config("charger", "clips"),
            "chest": chest(),
            **{f"{resource}_extractor": extractor(resource) for resource in elements},
            **{f"{gear}_station": gear_station(gear) for gear in gear},
        },
        commons=[
            CommonsConfig(
                name="cogs",
                inventory=InventoryConfig(
                    limits={
                        "resources": ResourceLimitsConfig(limit=10000, resources=elements),
                        "hearts": ResourceLimitsConfig(limit=65535, resources=["heart"]),
                    },
                    initial={
                        "carbon": 10,
                        "oxygen": 10,
                        "germanium": 10,
                        "silicon": 10,
                        "heart": 5,
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
