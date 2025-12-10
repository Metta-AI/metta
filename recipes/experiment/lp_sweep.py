"""Learning Progress Hyperparameter Sweep Recipe.

This recipe sweeps over learning progress algorithm parameters to find optimal
settings for curriculum-based training.

The key LP parameters being swept:
- lp_gain: Gain factor for performance bonus (z-score gain) - MOST IMPORTANT per Matt
- ema_timescale: Timescale for fast EMA (sets the low-frequency component)
- slow_timescale_factor: Factor for slow EMA (sets the width of frequency window)
- progress_smoothing: Nonlinear score transformation factor (NOT the same as Matt's
  "prioritization rescaling factor" which would control low vs high task priority)

NOTE: Matt mentioned a "prioritization rescaling factor" (sweep 0.4-0.6, where 0.5 is
neutral, <0.5 prioritizes low-scoring tasks, >0.5 prioritizes high-scoring). This
does NOT appear to exist in the current LP implementation. The `progress_smoothing`
parameter is a different kind of rescaling. You may need to implement the
prioritization factor separately or clarify with Matt what he meant.

Environment: Navigation tasks (high dynamic range, shows reward within 50-150M steps)

Usage:
    # Local test first
    uv run tools/run.py recipes.experiment.lp_sweep.sweep \
        sweep_name="prashant.lp_sweep_test" -- local_test=True

    # Run on sandbox (after local test passes)
    uv run tools/run.py recipes.experiment.lp_sweep.sweep \
        sweep_name="prashant.lp_sweep.12_09" -- gpus=4 nodes=1
"""

from typing import Optional

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import (
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
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.builder import envs as eb
from mettagrid.config.mettagrid_config import AsciiMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance
from recipes.experiment.cfg import NAVIGATION_EVALS

# ============================================================================
# Environment setup - Navigation (high dynamic range, fast reward signal)
# ============================================================================


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_nav_ascii_env(
    name: str,
    max_steps: int,
    num_agents: int = 1,
    num_instances: int = 4,
    border_width: int = 6,
    instance_border_width: int = 3,
) -> MettaGridConfig:
    path = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"
    env = eb.make_navigation(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps
    map_instance = AsciiMapBuilder.Config.from_uri(path)
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
            name=eval_cfg["name"],
            env=make_nav_ascii_env(
                name=eval_cfg["name"],
                max_steps=eval_cfg["max_steps"],
                num_agents=eval_cfg["num_agents"],
                num_instances=eval_cfg["num_instances"],
            ),
        )
        for eval_cfg in NAVIGATION_EVALS
    ] + [
        SimulationConfig(
            suite="navigation",
            name="emptyspace_sparse",
            env=make_emptyspace_sparse_env(),
        )
    ]
    return evals


def mettagrid(num_agents: int = 1, num_instances: int = 4) -> MettaGridConfig:
    """Create base navigation environment."""
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


# ============================================================================
# Curriculum with LP parameters exposed
# ============================================================================


def make_curriculum(
    nav_env: Optional[MettaGridConfig] = None,
    # LP parameters to sweep
    ema_timescale: float = 0.001,
    slow_timescale_factor: float = 0.2,
    progress_smoothing: float = 0.05,
    lp_gain: float = 0.1,
    exploration_bonus: float = 0.1,
    enable_detailed_slice_logging: bool = False,
) -> CurriculumConfig:
    """Create curriculum with configurable LP parameters."""
    nav_env = nav_env or mettagrid()

    # Build task variants
    dense_tasks = cc.bucketed(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance.dir", maps)
    dense_tasks.add_bucket("game.map_builder.instance.objects.assembler", [Span(3, 50)])

    # Create LP algorithm config with exposed parameters
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=ema_timescale,
        slow_timescale_factor=slow_timescale_factor,
        progress_smoothing=progress_smoothing,
        lp_gain=lp_gain,
        exploration_bonus=exploration_bonus,
        max_memory_tasks=1000,
        max_slice_axes=3,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    return dense_tasks.to_curriculum(
        num_active_tasks=1000,
        algorithm_config=algorithm_config,
    )


# ============================================================================
# Train function with LP parameters exposed for sweep
# ============================================================================


def train(
    # LP parameters - these are the parameters we sweep over
    ema_timescale: float = 0.001,
    slow_timescale_factor: float = 0.2,
    progress_smoothing: float = 0.05,
    lp_gain: float = 0.1,
    exploration_bonus: float = 0.1,
    # Training configuration
    enable_detailed_slice_logging: bool = False,
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
    """Train function with LP parameters exposed as CLI arguments.

    For sweeps, the sweep infrastructure will call this function with different
    parameter values, which are automatically parsed from CLI.

    LP Parameters being swept:
        ema_timescale: Timescale for fast EMA (sets the low-frequency component)
            Recommended range: 0.0001 to 0.01, search_center=0.001
        slow_timescale_factor: Factor for slow EMA (width of frequency window)
            Recommended range: 0.05 to 0.5, search_center=0.2
        progress_smoothing: Prioritization rescaling factor
            Recommended range: 0.01 to 0.2 (Matt suggested 0.4-0.6 but current default is 0.05)
        lp_gain: Gain factor for performance bonus (z-score gain)
            Recommended range: 0.01 to 0.5, search_center=0.1
    """
    # Use provided curriculum or create one with the LP parameters
    resolved_curriculum = curriculum or make_curriculum(
        ema_timescale=ema_timescale,
        slow_timescale_factor=slow_timescale_factor,
        progress_smoothing=progress_smoothing,
        lp_gain=lp_gain,
        exploration_bonus=exploration_bonus,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=make_navigation_eval_suite(),
        epoch_interval=300,  # Evaluate every 300 epochs
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    """Stub evaluation for sweep (using training metrics instead)."""
    return StubTool()


# ============================================================================
# Sweep configuration
# ============================================================================


def sweep(sweep_name: str) -> SweepTool:
    """Learning Progress hyperparameter sweep.

    Sweeps over 4 key LP parameters (keeping dimensionality under 8 as recommended):
    1. lp_gain (z-score gain) - Most important per Matt
    2. ema_timescale (sets the low-frequency component)
    3. slow_timescale_factor (width of the frequency window)
    4. progress_smoothing (score transformation, NOT prioritization direction)

    Usage:
        # Local test first (CRITICAL - catches config bugs before remote)
        uv run tools/run.py recipes.experiment.lp_sweep.sweep \
            sweep_name="prashant.lp_sweep_test" -- local_test=True

        # Launch on sandbox (after setting up tmux)
        uv run tools/run.py recipes.experiment.lp_sweep.sweep \
            sweep_name="prashant.lp_sweep.12_09" -- gpus=4 nodes=1

    Target: 60+ trials to see clustering at top of score range.
    """

    # Define LP parameter search space (4 parameters as recommended)
    parameters = [
        # 1. lp_gain (z-score gain) - Most important per Matt
        # Controls how much performance bonus affects LP score
        SP.param(
            "lp_gain",
            D.LOG_NORMAL,
            min=0.01,
            max=0.5,
            search_center=0.1,
        ),
        # 2. ema_timescale (sets the low-frequency component)
        # Higher = faster adaptation to new data
        SP.param(
            "ema_timescale",
            D.LOG_NORMAL,
            min=0.0001,
            max=0.01,
            search_center=0.001,
        ),
        # 3. slow_timescale_factor (width of the frequency window)
        # Factor applied to ema_timescale for slow EMA
        # Lower = wider window between fast and slow EMAs
        SP.param(
            "slow_timescale_factor",
            D.UNIFORM,
            min=0.05,
            max=0.5,
            search_center=0.2,
        ),
        # 4. progress_smoothing (nonlinear score transformation)
        # NOTE: This is NOT the "prioritization rescaling factor" Matt mentioned
        # (that would control low vs high task priority with 0.5 neutral).
        # This parameter applies a nonlinear transformation to LP scores.
        # Current default is 0.05, keeping similar range.
        SP.param(
            "progress_smoothing",
            D.UNIFORM,
            min=0.01,
            max=0.2,
            search_center=0.05,
        ),
        # Fixed training length - NOT swept because "longer = better" isn't useful
        # 150M steps should be enough for navigation to show differentiation
        {"trainer.total_timesteps": 150_000_000},  # 150M fixed
    ]

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.lp_sweep",
        train_entrypoint="train",
        eval_entrypoint="evaluate_stub",
        # Optimize for actual learning performance, not LP signal itself
        # experience/rewards measures how well the agent is learning
        metric_key="experience/rewards",
        search_space=parameters,
        max_trials=80,  # 60+ trials recommended
        num_parallel_trials=4,  # 4 GPUs = 4x faster at same cost
    )
