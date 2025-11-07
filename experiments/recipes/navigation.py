# ruff: noqa: E501
import typing

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.cogworks.curriculum.task_generator
import metta.map.terrain_from_numpy
import metta.rl.training
import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import mettagrid.config.mettagrid_config
import mettagrid.map_builder.random
import mettagrid.mapgen.mapgen

import experiments.evals.navigation


def mettagrid(
    num_agents: int = 1, num_instances: int = 4
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    nav = eb.make_navigation(num_agents=num_agents * num_instances)

    nav.game.map_builder = mettagrid.mapgen.mapgen.MapGen.Config(
        instances=num_instances,
        border_width=6,
        instance_border_width=3,
        instance=metta.map.terrain_from_numpy.NavigationFromNumpy.Config(
            agents=num_agents,
            objects={"altar": 10},
            dir="varied_terrain/dense_large",
        ),
    )
    return nav


def simulations() -> list[metta.sim.simulation_config.SimulationConfig]:
    return list(experiments.evals.navigation.make_navigation_eval_suite())


def make_curriculum(
    nav_env: typing.Optional[mettagrid.config.mettagrid_config.MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig
    ] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    nav_env = nav_env or mettagrid()

    # make a set of training tasks for navigation
    dense_tasks = cc.bucketed(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance.dir", maps)
    dense_tasks.add_bucket(
        "game.map_builder.instance.objects.altar",
        [metta.cogworks.curriculum.task_generator.Span(3, 50)],
    )

    # sparse environments are just random maps
    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = (
        mettagrid.map_builder.random.RandomMapBuilder.Config(
            agents=4,
            objects={"altar": 10},
        )
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket(
        "game.map_builder.width",
        [metta.cogworks.curriculum.task_generator.Span(60, 120)],
    )
    sparse_tasks.add_bucket(
        "game.map_builder.height",
        [metta.cogworks.curriculum.task_generator.Span(60, 120)],
    )
    sparse_tasks.add_bucket(
        "game.map_builder.objects.altar",
        [metta.cogworks.curriculum.task_generator.Span(1, 10)],
    )

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
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
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
) -> metta.tools.train.TrainTool:
    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    evaluator_cfg = metta.rl.training.EvaluatorConfig(
        simulations=experiments.evals.navigation.make_navigation_eval_suite(),
    )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(
            curriculum=resolved_curriculum
        ),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: str | typing.Sequence[str] | None = None,
) -> metta.tools.eval.EvaluateTool:
    return metta.tools.eval.EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
    )


def play_training_env(
    policy_uri: typing.Optional[str] = None,
) -> metta.tools.play.PlayTool:
    env = mettagrid()
    return metta.tools.play.PlayTool(
        sim=metta.sim.simulation_config.SimulationConfig(
            suite="navigation", name="training_env", env=env
        ),
        policy_uri=policy_uri,
    )


def play(policy_uri: typing.Optional[str] = None) -> metta.tools.play.PlayTool:
    return metta.tools.play.PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: typing.Optional[str] = None) -> metta.tools.replay.ReplayTool:
    return metta.tools.replay.ReplayTool(sim=simulations()[0], policy_uri=policy_uri)
