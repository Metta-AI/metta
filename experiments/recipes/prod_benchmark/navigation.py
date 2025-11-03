from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.map.terrain_from_numpy import NavigationFromNumpy
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import make_sweep, SweepParameters as SP, Distribution as D
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen

from experiments.evals.navigation import make_navigation_eval_suite


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

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

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
    use_curriculum: bool = False,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with fixed environment (no curriculum by default for benchmarking).

    Set use_curriculum=True to enable curriculum learning.
    """
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=make_navigation_eval_suite(),
    )

    # For benchmarking, use fixed environment by default (no curriculum)
    if use_curriculum:
        curriculum = make_curriculum(
            enable_detailed_slice_logging=enable_detailed_slice_logging
        )
        training_env = TrainingEnvironmentConfig(curriculum=curriculum)
    else:
        # Fixed environment for benchmarking
        training_env = TrainingEnvironmentConfig(env=mettagrid())

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
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


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Evaluation tool for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    NB: Please note that this function takes a **single** policy_uri. This is the expected signature in our sweeps.
    Additional arguments are supported through eval_overrides.
    """
    # Create sweep-optimized versions of the standard evaluations
    # Use a dedicated suite name to control the metric namespace in WandB
    sweep_simulations = [
        SimulationConfig(
            suite="sweep",
            name=sim.name,
            env=sim.env,
            num_episodes=10,  # 10 episodes for statistical reliability
            max_time_s=240,  # 4 minutes max per simulation
        )
        for sim in simulations()
    ]

    return EvaluateTool(
        simulations=sweep_simulations,
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> SweepTool:
    """Prototypical sweep function.

    In your own recipe, you likely only every need this. You can override other SweepTool parameters in the CLI.

    Example usage:
        `uv run ./tools/run.py experiments.recipes.benchmarks.navigation.sweep sweep_name="nav.sweep.10081528" -- gpus=4 nodes=2`

    We recommend running using local_test=True before running the sweep on the remote:
        `uv run ./tools/run.py experiments.recipes.benchmarks.navigation.sweep sweep_name="nav.sweep.10081528.local_test" -- local_test=True`
    This will run a quick local sweep and allow you to catch configuration bugs (NB: Unless those bugs are related to batch_size, minibatch_size, or hardware configuration).
    If this runs smoothly, you must launch the sweep on a remote sandbox (otherwise sweep progress will halt when you close your computer).

    Running on the remote:
        1 - Start a sweep controller sandbox: `./devops/skypilot/sandbox.py --sweep-controller`, and ssh into it.
        2 - Clean git pollution: `git clean -df && git stash`
        3 - Ensure your sky credentials are present: `sky status` -- if not, follow the instructions on screen.
        4 - Install tmux on the sandbox `apt install tmux`
        5 - Launch tmux session: `tmux new -s sweep`
        6 - Launch the sweep: `uv run ./tools/run.py experiments.recipes.benchmarks.navigation.sweep sweep_name="nav.sweep.10081528" -- gpus=4 nodes=2`
        7 - Detach when you want: CTRL+B then d
        8 - Attach to look at status/output: `tmux attach -t sweep`

    Please tag Axel (akerbec@softmax.ai) on any bug report.
    """

    # Common parameters are accessible via SP (SweepParameters).
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
        recipe="experiments.recipes.benchmarks.navigation",
        train_entrypoint="train",
        # NB: You MUST use a specific sweep eval suite, different than those in training.
        # Besides this being a recommended practice, using the same eval suite in both
        # training and scoring will lead to key conflicts that will lock the sweep.
        eval_entrypoint="evaluate_in_sweep",
        # Typically, "evaluator/eval_{suite}/score"
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        num_trials=80,
        # Default value is 1. We don't recommend going higher than 4.
        # The faster each individual trial, the lower you should set this number.
        num_parallel_trials=4,
    )
