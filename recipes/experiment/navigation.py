from typing import Optional, Sequence

from cogames.cogs_vs_clips.mission import Mission, Site
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from metta.cogworks.curriculum import bucketed, merge
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.map.terrain_from_numpy import NavigationFromNumpy
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import AsciiMapBuilder, MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance
from recipes.experiment.cfg import NAVIGATION_EVALS


def make_cogames_nav_env(map_builder: MapGen.Config, num_agents: int, reward_scale: float = 1.0) -> MettaGridConfig:
    """Build a navigation environment that follows Cogames/CvC station rules."""
    site = Site(
        name="navigation_site",
        description="Navigation map aligned with Cogames rules",
        map_builder=map_builder,
        min_cogs=num_agents,
        max_cogs=num_agents,
    )

    mission = Mission(
        name="navigation",
        description="Navigation mission using converters and chests",
        site=site,
        num_cogs=num_agents,
        assembler=CvCAssemblerConfig(),
        chest=CvCChestConfig(),
        charger=ChargerConfig(),
        carbon_extractor=CarbonExtractorConfig(),
        oxygen_extractor=OxygenExtractorConfig(),
        germanium_extractor=GermaniumExtractorConfig(),
        silicon_extractor=SiliconExtractorConfig(),
        wall=CvCWallConfig(),
    )

    env = mission.make_env()
    env.game.agent.rewards.stats["chest.heart.amount"] = reward_scale / num_agents
    return env


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.stats["chest.heart.amount"] = 0.333 / env.game.num_agents
    return env


def make_nav_ascii_env(
    name: str,
    max_steps: int,
    num_agents=1,
    num_instances=4,
    border_width: int = 6,
    instance_border_width: int = 3,
) -> MettaGridConfig:
    # Re-use nav sequence maps but remap objects to Cogames convertors/chests
    path = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"

    map_instance = AsciiMapBuilder.Config.from_uri(path)

    # Replace nav map objects with Cogames stations to match mission rules.
    map_instance.char_to_map_name["n"] = "assembler"
    map_instance.char_to_map_name["m"] = "chest"
    map_instance.char_to_map_name["_"] = "empty"  # Remove altars (not supported in C++ simulator)

    map_builder = MapGen.Config(
        instances=num_instances,
        border_width=border_width,
        instance_border_width=instance_border_width,
        instance=map_instance,
    )

    env = make_cogames_nav_env(map_builder=map_builder, num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> MettaGridConfig:
    map_builder = MapGen.Config(
        instances=4,
        instance=MapGen.Config(
            width=60,
            height=60,
            border_width=3,
            instance=MeanDistance.Config(
                mean_distance=30,
                objects={
                    "assembler": 2,
                    "chest": 2,
                    "charger": 2,
                    "carbon_extractor": 2,
                },
            ),
        ),
    )
    env = make_cogames_nav_env(map_builder=map_builder, num_agents=4)
    env.game.max_steps = 300
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
    map_builder = MapGen.Config(
        instances=num_instances,
        border_width=6,
        instance_border_width=3,
        instance=NavigationFromNumpy.Config(
            agents=num_agents,
            objects={
                "assembler": 6,
                "chest": 4,
                "charger": 4,
                "carbon_extractor": 4,
                "oxygen_extractor": 2,
            },
            dir="varied_terrain/dense_large",
            remove_altars=True,
        ),
    )
    return make_cogames_nav_env(map_builder=map_builder, num_agents=num_agents * num_instances)


def simulations() -> list[SimulationConfig]:
    return list(make_navigation_eval_suite())


def make_curriculum(
    nav_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    nav_env = nav_env or mettagrid()

    # make a set of training tasks for navigation
    dense_tasks = bucketed(nav_env)

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
        objects={
            "assembler": 6,
            "chest": 4,
            "charger": 4,
            "carbon_extractor": 4,
            "oxygen_extractor": 2,
        },
    )
    sparse_tasks = bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.assembler", [Span(1, 10)])

    nav_tasks = merge([dense_tasks, sparse_tasks])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
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
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    resolved_curriculum = curriculum or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)

    evaluator_cfg = EvaluatorConfig(
        simulations=make_navigation_eval_suite(),
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: Optional[Sequence[str] | str] = None,
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


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Sweep-optimized evaluation using fewer episodes and a shorter time budget."""
    sweep_simulations = [
        SimulationConfig(
            suite="navigation_sweep",
            name=sim.name,
            env=sim.env,
            num_episodes=1,
            max_time_s=240,
        )
        for sim in make_navigation_eval_suite()
    ]

    return EvaluateTool(
        simulations=sweep_simulations,
        policy_uris=[policy_uri],
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    return StubTool()


def sweep(sweep_name: str) -> SweepTool:
    """
    Prototypical sweep function. Override SweepTool parameters via the CLI if needed.

    Example usage:
        `uv run ./tools/run.py recipes.experiment.navigation.sweep \
            sweep_name=\"ak.nav.12345678\" -- gpus=4 nodes=2`
    We recommend running using local_test=True before running the sweep on the remote:
        `uv run ./tools/run.py recipes.experiment.navigation.sweep \
            sweep_name=\"ak.nav.12345678.local_test\" -- local_test=True`

    This will run a quick local sweep and allow you to catch configuration bugs
    (NB: Unless those bugs are related to batch_size, minibatch_size, or hardware config).
    If this runs smoothly, you must launch the sweep on a remote sandbox
    (otherwise sweep progress will halt when you close your computer).

    Running on the remote:
        1 - Start a sweep controller sandbox: `./devops/skypilot/sandbox.py new --sweep-controller`, and ssh into it.
        2 - Clean git pollution: `git clean -df && git stash`
        3 - Ensure your sky credentials are present: `sky status` -- if not, follow the instructions on screen.
        4 - Install tmux on the sandbox `apt install tmux`
        5 - Launch tmux session: `tmux new -s sweep`
        6 - Launch the sweep:
            `uv run ./tools/run.py recipes.experiment.navigation.sweep \
                sweep_name=\"ak.nav.12345678\" -- gpus=4 nodes=2`
        7 - Detach when you want: CTRL+B then d
        8 - Attach to look at status/output: `tmux attach -t sweep_configs`

    Please tag Axel (akerbec@softmax.ai) on any bug report.
    """
    parameters = [
        SP.LEARNING_RATE,
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.ADAM_EPS,
        SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=7.5e8,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.navigation",
        train_entrypoint="train",
        eval_entrypoint="evaluate_stub",
        objective="experience/rewards",
        parameters=parameters,
        max_trials=80,
        num_parallel_trials=4,
    )
