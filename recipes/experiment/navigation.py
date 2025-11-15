import random
from typing import Optional, Sequence

import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import (
    TaskGenerator,
    TaskGeneratorConfig,
)
from metta.map.terrain_from_numpy import NavigationFromNumpy
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import AsciiMapBuilder, MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance
from recipes.experiment.cfg import NAVIGATION_EVALS


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
    # we re-use nav sequence maps, but replace all objects with altars
    path = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"

    env = eb.make_navigation(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps

    map_instance = AsciiMapBuilder.Config.from_uri(path)

    # Replace objects with altars by setting char_to_map_name (char -> map_name, the stable ASCII map key).
    map_instance.char_to_map_name["n"] = "altar"
    map_instance.char_to_map_name["m"] = "altar"

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
                objects={"altar": 3},
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


class NavigationTaskGenerator(TaskGenerator):
    """Custom task generator for navigation that creates map_builder configs directly."""

    class Config(TaskGeneratorConfig["NavigationTaskGenerator"]):
        """Configuration for NavigationTaskGenerator."""

        base_env: MettaGridConfig
        num_instances: int = 4
        num_agents_per_instance: int = 1

    def __init__(self, config: "NavigationTaskGenerator.Config"):
        super().__init__(config)
        self._config = config

        # Build list of dense terrain maps
        self._dense_maps = ["terrain_maps_nohearts"]
        for size in ["large", "medium", "small"]:
            for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
                self._dense_maps.append(f"varied_terrain/{terrain}_{size}")

        # Ranges for variation
        self._altar_range = (3, 50)
        self._sparse_width_range = (60, 120)
        self._sparse_height_range = (60, 120)
        self._sparse_altar_range = (1, 10)

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Generate task by constructing full map_builder config."""
        env_cfg = self._config.base_env.model_copy(deep=True)

        # 50% chance of dense terrain, 50% chance of sparse random map
        if rng.random() < 0.5:
            # Dense terrain task
            map_dir = rng.choice(self._dense_maps)
            altar_count = rng.randint(*self._altar_range)

            env_cfg.game.map_builder = MapGen.Config(
                instances=self._config.num_instances,
                border_width=6,
                instance_border_width=3,
                instance=NavigationFromNumpy.Config(
                    agents=self._config.num_agents_per_instance,
                    objects={"altar": altar_count},
                    dir=map_dir,
                ),
            )

            # Set label based on terrain
            terrain_name = map_dir.split("/")[-1] if "/" in map_dir else map_dir
            env_cfg.label = terrain_name
        else:
            # Sparse random map task
            width = rng.randint(*self._sparse_width_range)
            height = rng.randint(*self._sparse_height_range)
            altar_count = rng.randint(*self._sparse_altar_range)

            env_cfg.game.map_builder = RandomMapBuilder.Config(
                agents=self._config.num_instances * self._config.num_agents_per_instance,
                width=width,
                height=height,
                objects={"altar": altar_count},
            )

            env_cfg.label = "random"

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
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    num_instances: int = 4,
    num_agents_per_instance: int = 1,
) -> CurriculumConfig:
    nav_env = nav_env or mettagrid(num_agents=num_agents_per_instance, num_instances=num_instances)

    # Use custom task generator that creates map_builder configs directly
    nav_tasks_config = NavigationTaskGenerator.Config(
        base_env=nav_env,
        num_instances=num_instances,
        num_agents_per_instance=num_agents_per_instance,
    )

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            num_active_tasks=256,
            slow_timescale_factor=0.2,
            rand_task_rate=0.01,
            exploration_bonus=0.1,
            min_samples_for_lp=10,  # Use exploration bonus for first 10 samples
            lp_score_temperature=0.0,  # Z-score normalization for relative LP comparison
            z_score_amplification=50.0,  # Amplification after z-score (only when temp=0)
            show_curriculum_troubleshooting_logging=True,  # Enable per-task metrics for debugging
            early_progress_amplification=0.5,  # 0.5 = OFF, low values (0.05) amplify unsolved tasks
        )

    return CurriculumConfig(
        task_generator=nav_tasks_config,
        num_active_tasks=50,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
    curriculum = curriculum or make_curriculum()

    evaluator_cfg = EvaluatorConfig(
        simulations=make_navigation_eval_suite(),
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
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
