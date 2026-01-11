from __future__ import annotations

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.map.terrain_from_numpy import NavigationFromNumpy
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import AsciiMapBuilder, MettaGridConfig
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance
from recipes.experiment.architectures import get_architecture
from recipes.experiment.cfg import NAVIGATION_EVALS
import metta.tools as tools

def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_nav_ascii_env(
    name: str,
    max_steps: int,
    num_agents=1,
    num_instances=4,
    border_width: int = 6,
    instance_border_width: int = 3,
) -> MettaGridConfig:
    # we re-use nav sequence maps, but replace all objects with assemblers
    path = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"

    env = eb.make_navigation(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps

    map_instance = AsciiMapBuilder.Config.from_uri(path)

    # Replace objects with assemblers by setting char_to_map_name (char -> map_name, the stable ASCII map key).
    map_instance.char_to_map_name["n"] = "assembler"
    map_instance.char_to_map_name["m"] = "assembler"

    env.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=border_width,
        instance_border_width=instance_border_width,
        instance=map_instance,
    )

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> MettaGridConfig:
    env = eb.make_navigation(num_agents=4)
    env.game.max_steps = 300
    env.game.map_builder = MapGen.Config(
        instances=4,
        instance=MapGen.Config(
            width=60,
            height=60,
            border_width=3,
            instance=MeanDistance.Config(
                mean_distance=30,
                objects={"assembler": 3},
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_navigation_eval_suite() -> list[SimulationConfig]:
    evals = [
        SimulationConfig(
            suite="navigation",
            name=eval["name"],
            env=make_nav_ascii_env(
                name=eval["name"],
                max_steps=eval["max_steps"],
                num_agents=eval["num_agents"],
                num_instances=eval["num_instances"],
            ),
        )
        for eval in NAVIGATION_EVALS
    ] + [
        SimulationConfig(
            suite="navigation",
            name="emptyspace_sparse",
            env=make_emptyspace_sparse_env(),
        )
    ]
    return evals


def mettagrid(num_agents: int = 1, num_instances: int = 4) -> MettaGridConfig:
    nav = eb.make_navigation(num_agents=num_agents * num_instances)

    nav.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=6,
        instance_border_width=3,
        instance=NavigationFromNumpy.Config(
            agents=num_agents,
            objects={"assembler": 10},
            dir="varied_terrain/dense_large",
        ),
    )
    return nav


def simulations() -> list[SimulationConfig]:
    return list(make_navigation_eval_suite())


def make_curriculum(
    nav_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    nav_env = nav_env or mettagrid()

    # make a set of training tasks for navigation
    dense_tasks = cc.bucketed(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance.dir", maps)
    dense_tasks.add_bucket("game.map_builder.instance.objects.assembler", [Span(3, 50)])

    # sparse environments are just random maps
    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = RandomMapBuilder.Config(
        agents=4,
        objects={"assembler": 10},
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.assembler", [Span(1, 10)])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.006,  # Tuned via sweep prashant.lp_sweep.12_10_2 (was 0.001)
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=3,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return nav_tasks.to_curriculum(
        num_active_tasks=1000,  # Smaller pool for navigation tasks
        algorithm_config=algorithm_config,
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    arch_type: str = "vit",
) -> tools.TrainTool:
    resolved_curriculum = curriculum or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)

    evaluator_cfg = EvaluatorConfig(
        simulations=make_navigation_eval_suite(),
    )

    return tools.TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
        policy_architecture=get_architecture(arch_type),
    )


def evaluate(
    policy_uris: list[str] | str,
) -> tools.EvaluateTool:
    return tools.EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
    )


def play_training_env(policy_uri: Optional[str] = None) -> tools.PlayTool:
    env = mettagrid()
    return tools.PlayTool(
        sim=SimulationConfig(suite="navigation", name="training_env", env=env),
        policy_uri=policy_uri,
    )


def play(policy_uri: Optional[str] = None) -> tools.PlayTool:
    return tools.PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> tools.ReplayTool:
    return tools.ReplayTool(sim=simulations()[0], policy_uri=policy_uri)
