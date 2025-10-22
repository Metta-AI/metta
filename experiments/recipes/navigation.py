import random
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import (
    AnyTaskGeneratorConfig,
    Span,
    TaskGenerator,
    TaskGeneratorConfig,
)
from metta.map.terrain_from_numpy import NavigationFromNumpy
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen

from experiments.evals.navigation import make_navigation_eval_suite


class NavigationTaskGenerator(TaskGenerator):
    """Custom task generator for navigation that sets dynamic labels based on task parameters."""

    class Config(TaskGeneratorConfig["NavigationTaskGenerator"]):
        """Configuration for NavigationTaskGenerator."""

        child_generator: AnyTaskGeneratorConfig

    def __init__(self, config: "NavigationTaskGenerator.Config"):
        super().__init__(config)
        self._config = config
        self._child_generator = config.child_generator.create()

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Generate task and set dynamic label based on task parameters."""
        env_cfg = self._child_generator.get_task(task_id)

        # Get the bucket values that were sampled
        bucket_values = getattr(self._child_generator, "_last_bucket_values", {})

        # Extract key parameters for labeling
        map_dir = bucket_values.get("game.map_builder.instance.dir", "")
        width = bucket_values.get("game.map_builder.width")
        height = bucket_values.get("game.map_builder.height")

        # Create label based on task type
        if map_dir:
            # Dense task - use terrain directory
            # Extract just the terrain name from path like "varied_terrain/dense_large"
            terrain_name = map_dir.split("/")[-1] if "/" in map_dir else map_dir
            label = terrain_name
        elif width and height:
            # Sparse task - use dimensions
            label = f"random_{width}x{height}"
        else:
            label = "navigation"

        env_cfg.label = label
        return env_cfg


def mettagrid(num_agents: int = 1, num_instances: int = 4) -> MettaGridConfig:
    nav = eb.make_navigation(num_agents=num_agents * num_instances)

    nav.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=6,
        instance_border_width=3,
        instance=NavigationFromNumpy.Config(
            agents=num_agents,
            objects={"altar": 10},
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
    dense_tasks.add_bucket("game.map_builder.instance.objects.altar", [Span(3, 50)])

    # sparse environments are just random maps
    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = RandomMapBuilder.Config(
        agents=4,
        objects={"altar": 10},
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [Span(1, 10)])

    # Wrap in NavigationTaskGenerator to add dynamic labels
    nav_tasks_config = cc.merge([dense_tasks, sparse_tasks])
    nav_tasks_with_labels = NavigationTaskGenerator.Config(
        child_generator=nav_tasks_config
    )

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_slice_axes=3,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return CurriculumConfig(
        task_generator=nav_tasks_with_labels,
        num_active_tasks=1000,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=make_navigation_eval_suite(),
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    return EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
    )


def play_training_env(policy_uri: Optional[str] = None) -> PlayTool:
    env = mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="navigation", name="training_env", env=env),
        policy_uri=policy_uri,
    )


def play(policy_uri: Optional[str] = None) -> PlayTool:
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)
