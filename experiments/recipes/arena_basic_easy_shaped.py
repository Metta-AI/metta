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
from metta.sweep.core import SweepParameters as SP
from metta.sweep.ray.ray_controller import SweepConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.ray_sweep import RaySweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


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


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Evaluation tool for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    NB: Please note that this function takes a **single** policy_uri. This is the expected signature in our sweeps.
    Additional arguments are supported through eval_overrides.
    """

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


def sweep_muon(sweep_name: str) -> RaySweepTool:
    """
    Sweep using Muon optimizer for large-scale training.

    Muon is designed for training large models and uses momentum-based optimization.
    """
    all_specs = [
        *SP.muon_optimizer_hypers(),  # Muon-specific parameters
        *SP.ppo_loss_hypers(include_advanced=False),  # Basic PPO for simplicity
    ]

    search_space = {spec.path: spec.space for spec in all_specs}
    search_space["trainer.total_timesteps"] = 2_000_000_000
    # IMPORTANT: Set optimizer type to muon
    search_space["trainer.optimizer.type"] = "muon"

    sweep_config = SweepConfig(
        sweep_id=sweep_name,
        recipe_module="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        num_samples=50,
        gpus_per_trial=4,
        max_concurrent_trials=4,
    )

    return RaySweepTool(
        sweep_config=sweep_config,
        search_space=search_space,
    )


def sweep_minimal(sweep_name: str) -> RaySweepTool:
    """
    Minimal sweep focusing on the most impactful hyperparameters.

    Uses only basic PPO parameters without advanced features like V-trace or PER.
    """
    all_specs = [
        *SP.adam_optimizer_hypers(),
        *SP.ppo_loss_hypers(include_advanced=False),  # Basic PPO only
    ]

    search_space = {spec.path: spec.space for spec in all_specs}
    search_space["trainer.total_timesteps"] = (
        500_000_000  # Shorter runs for quick iteration
    )

    sweep_config = SweepConfig(
        sweep_id=sweep_name,
        recipe_module="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        num_samples=50,  # Fewer samples for faster exploration
        gpus_per_trial=2,  # Less GPU per trial
        max_concurrent_trials=8,  # More concurrent trials
    )

    return RaySweepTool(
        sweep_config=sweep_config,
        search_space=search_space,
    )


def sweep_full(sweep_name: str) -> RaySweepTool:
    """
    Comprehensive Ray sweep covering TrainerConfig and PPOConfig hyperparameters.
    """

    # Use canonical parameter sets from SweepParameters
    all_specs = [
        *SP.adam_optimizer_hypers(),  # Adam-specific parameters only
        *SP.ppo_loss_hypers(include_advanced=True),
        # Uncomment to include training hyperparameters:
        # *SP.training_hypers(),
    ]

    search_space = {spec.path: spec.space for spec in all_specs}
    search_space["trainer.total_timesteps"] = 2_000_000_000

    sweep_config = SweepConfig(
        sweep_id=sweep_name,
        recipe_module="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        # No evals yet
        eval_entrypoint="evaluate_in_sweep",
        # No score key yet
        num_samples=100,
        gpus_per_trial=4,
        max_concurrent_trials=4,
    )

    return RaySweepTool(
        sweep_config=sweep_config,
        search_space=search_space,
    )
