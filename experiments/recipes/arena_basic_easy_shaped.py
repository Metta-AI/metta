from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from experiments.sweeps.protein_configs import make_custom_protein_config, PPO_BASIC
from experiments.sweeps.standard import protein_sweep
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.protein_config import ParameterConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
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
    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

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


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena/basic", env=basic_env),
        SimulationConfig(name="arena/combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    enable_contrastive: bool = True,
) -> TrainTool:
    trainer_cfg = TrainerConfig(
        losses=LossConfig(enable_contrastive=enable_contrastive),
        curriculum=curriculum
        or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging),
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(
                    name="arena/basic", env=eb.make_arena(num_agents=24, combat=False)
                ),
                SimulationConfig(
                    name="arena/combat", env=eb.make_arena(num_agents=24, combat=True)
                ),
            ],
            evaluate_remote=True,
            evaluate_local=False,
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def train_with_contrastive(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    contrastive_temperature: float = 0.07,
    contrastive_coef: float = 0.1,
) -> TrainTool:
    """Train with contrastive loss using specified hyperparameters."""
    # Create loss config with contrastive enabled and configured
    loss_config = LossConfig(enable_contrastive=True)
    loss_config.loss_configs["contrastive"] = ContrastiveConfig(
        temperature=contrastive_temperature,
        contrastive_coef=contrastive_coef,
    )

    trainer_cfg = TrainerConfig(
        losses=loss_config,
        curriculum=curriculum
        or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging),
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(
                    name="arena/basic", env=eb.make_arena(num_agents=24, combat=False)
                ),
                SimulationConfig(
                    name="arena/combat", env=eb.make_arena(num_agents=24, combat=True)
                ),
            ],
            evaluate_remote=True,
            evaluate_local=False,
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(sim=SimulationConfig(env=eval_env, name="arena"))


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="arena"))


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def evaluate_in_sweep(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    """Evaluation function optimized for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    """
    if simulations is None:
        # Create sweep-optimized versions of the standard evaluations
        basic_env = make_mettagrid()
        basic_env.game.actions.attack.consumed_resources["laser"] = 100

        combat_env = basic_env.model_copy()
        combat_env.game.actions.attack.consumed_resources["laser"] = 1

        simulations = [
            SimulationConfig(
                name="arena/basic",
                env=basic_env,
                num_episodes=10,  # 10 episodes for statistical reliability
                max_time_s=240,  # 4 minutes max per simulation
            ),
            SimulationConfig(
                name="arena/combat",
                env=combat_env,
                num_episodes=10,
                max_time_s=240,
            ),
        ]

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def sweep_contrastive(
    max_trials: int = 300,
    max_parallel_jobs: int = 6,
    max_timesteps: int = 1000000,
    gpus: int = 1,
    batch_size: int = 4,
    local_test: bool = False,
) -> SweepTool:
    """Sweep for contrastive loss hyperparameters.

    Sweeps over:
    - temperature: [0, 0.5] with uniform distribution
    - contrastive_coef: [0, 1] with uniform distribution

    Args:
        max_trials: Maximum number of trials to run
        max_parallel_jobs: Maximum number of parallel jobs
        max_timesteps: Maximum timesteps per trial
        gpus: Number of GPUs per job
        batch_size: Batch size multiplier
        local_test: Whether to run locally for testing
    """
    # Create a custom sweep configuration for contrastive loss
    contrastive_protein_config = make_custom_protein_config(
        base_config=PPO_BASIC,
        parameters={
            # Temperature parameter for contrastive loss
            "contrastive_temperature": ParameterConfig(
                distribution="uniform",
                min=0.0,
                max=0.5,
                mean=0.25,
                scale="auto",
            ),
            # Coefficient for contrastive loss
            "contrastive_coef": ParameterConfig(
                distribution="uniform",
                min=0.0,
                max=1.0,
                mean=0.5,
                scale="auto",
            ),
        },
    )

    # Set the metric for optimization
    contrastive_protein_config.metric = "experience/rewards"

    # Remove batch_size from parameters if it exists (we'll use the batch_size argument instead)
    contrastive_protein_config.parameters.pop("trainer.batch_size", None)

    return protein_sweep(
        recipe="experiments.recipes.arena_basic_easy_shaped",
        train="train_with_contrastive",  # Use the custom train function that accepts contrastive params
        eval="evaluate_in_sweep",
        protein_config=contrastive_protein_config,
        max_parallel_jobs=max_parallel_jobs,
        max_timesteps=max_timesteps,
        max_trials=max_trials,
        gpus=gpus,
        batch_size=batch_size,
        local_test=local_test,
    )
