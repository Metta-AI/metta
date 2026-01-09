"""A Cogs vs Clips version of the arena recipe - STABLE

This is meant as a basic testbed for CvC buildings / mechanics.
This recipe is automatically validated in CI and release processes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.missions import make_cogsguard_mission
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
from mettagrid.config.mettagrid_config import MettaGridConfig


def make_env(num_agents: int = 10, max_steps: int = 1000) -> MettaGridConfig:
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
            "chest",
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
        max_steps=max_steps,
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
            collective="cogs",
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
                collective_inventory={
                    # "carbon": 0.001,
                    # "oxygen": 0.001,
                    # "germanium": 0.001,
                    # "silicon": 0.001,
                },
                collective_stats={
                    "aligned.charger.held": 1.0 / max_steps,
                },
            ),
        ),
        inventory_regen_interval=1,
        objects={
            "wall": CvCWallConfig().station_cfg(),
            "assembler": nexus("assembler"),
            "charger": supply_depot_config("charger", "clips"),
            "chest": chest(),
            **{f"{resource}_extractor": extractor(resource) for resource in elements},
            **{f"{g}_station": gear_station(g) for g in gear},
        },
        collectives={
            "cogs": CollectiveConfig(
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
            "clips": CollectiveConfig(
                name="clips",
            ),
        },
    )

    env = MettaGridConfig(game=game)
    return env


def make_curriculum(
    env: Optional[MettaGridConfig] = None,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    env = env or make_env()

    tasks = cc.bucketed(env)

    # for item in ["ore_red", "battery_red", "laser", "armor"]:
    #     arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
    #     arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    tasks.add_bucket("game.max_steps", [1000, 5000, 10000])
    tasks.add_bucket("game.agent.inventory.initial.heart", [0, 1, 2, 3])

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

    return tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    env = env or make_env()

    return [
        SimulationConfig(suite="cogsguard", name="basic", env=env),
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
