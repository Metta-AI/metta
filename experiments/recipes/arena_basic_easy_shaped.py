from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.protein_config import ParameterConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepSchedulerType, SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig

from experiments.sweeps.protein_configs import PPO_CORE, make_custom_protein_config


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)

    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # sometimes add initial_items to the buildings
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena", name="basic", env=basic_env),
        SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = simulations()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=eval_simulations),
        policy_architecture=policy_architecture,
        torch_profiler=TorchProfilerConfig(),
    )


def evaluate(policy_uris: Optional[Sequence[str]] = None) -> EvaluateTool:
    """Evaluate policies on arena simulations."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with a policy."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Generate replay from a policy."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate_in_sweep(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> EvaluateTool:
    """Evaluation function optimized for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    """
    if simulations is None:
        # Create sweep-optimized versions of the standard evaluations
        # Use a dedicated suite name to control the metric namespace in WandB
        basic_env = mettagrid()
        basic_env.game.actions.attack.consumed_resources["laser"] = 100

        combat_env = basic_env.model_copy()
        combat_env.game.actions.attack.consumed_resources["laser"] = 1

        simulations = [
            SimulationConfig(
                suite="sweep",
                name="basic",
                env=basic_env,
                num_episodes=10,  # 10 episodes for statistical reliability
                max_time_s=240,  # 4 minutes max per simulation
            ),
            SimulationConfig(
                suite="sweep",
                name="combat",
                env=combat_env,
                num_episodes=10,
                max_time_s=240,
            ),
        ]

    return EvaluateTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def sweep_async_progressive(
    min_timesteps: int,
    max_timesteps: int,
    initial_timesteps: int,
    max_concurrent_evals: int = 1,
    liar_strategy: str = "best",
) -> SweepTool:
    """Async-capped sweep that also sweeps over total timesteps.

    Args:
        min_timesteps: Minimum trainer.total_timesteps to consider.
        max_timesteps: Maximum trainer.total_timesteps to consider.
        initial_timesteps: Initial/mean value for trainer.total_timesteps.
        max_concurrent_evals: Max number of concurrent evals (default: 1).
        liar_strategy: Constant Liar strategy (best|mean|worst).

    Returns:
        SweepTool configured for async-capped scheduling and progressive timesteps.
    """

    protein_cfg = make_custom_protein_config(
        PPO_CORE,
        {
            "trainer.total_timesteps": ParameterConfig(
                min=min_timesteps,
                max=max_timesteps,
                distribution="int_uniform",
                mean=initial_timesteps,
                scale="auto",
            )
        },
    )

    # Ensure the optimizer reads from the sweep-specific metric namespace
    protein_cfg.metric = "evaluator/eval_sweep/score"

    return SweepTool(
        # Protein with swept timesteps
        protein_config=protein_cfg,
        # Recipe entrypoints
        recipe_module="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        # Async scheduler selection + knobs
        scheduler_type=SweepSchedulerType.ASYNC_CAPPED,
        max_concurrent_evals=max_concurrent_evals,
        liar_strategy=liar_strategy,
    )
