"""Arena recipe with medium shaped rewards: intermediate shaping for ore, battery, laser, and armor."""

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
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import make_sweep, SweepParameters as SP, Distribution as D
from metta.tools.eval import EvaluateTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create arena environment with medium shaped rewards.

    Progression: moderate rewards for key resources including combat items.
    """
    arena_env = eb.make_arena(num_agents=num_agents)

    # Medium shaped rewards: moderate rewards for key resources
    arena_env.game.agent.rewards.inventory = {
        "heart": 1.0,
        "ore_red": 0.08,  # Slightly higher reward for mining
        "battery_red": 0.5,  # Moderate reward for battery
        "laser": 0.3,  # Reward for combat capability
        "armor": 0.3,  # Reward for defense
        "blueprint": 0.0,  # No reward for blueprint yet
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
    """Create curriculum with medium shaped reward environment."""
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # Vary intermediate rewards in curriculum
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.3, 0.5, 0.8]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # Enable/disable attacks for variety
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Vary initial resources in buildings
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation environments with medium shaped rewards."""
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena_medium_shaped", name="basic", env=basic_env),
        SimulationConfig(suite="arena_medium_shaped", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    enable_contrastive: bool = True,
    policy_architecture: Optional[PolicyArchitecture] = None,
    # These parameters can now be swept over.
    temperature: float = 0.07,
    contrastive_coef: float = 0.1,
) -> TrainTool:
    """Train with medium shaped rewards and optional contrastive loss."""
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    contrastive_config = ContrastiveConfig(
        temperature=temperature,
        contrastive_coef=contrastive_coef,
        embedding_dim=128,
        use_projection_head=True,
    )

    ppo_config = PPOConfig()

    loss_configs = {"ppo": ppo_config}
    if enable_contrastive:
        loss_configs["contrastive"] = contrastive_config

    trainer_config = TrainerConfig(
        losses=LossConfig(
            enable_contrastive=enable_contrastive,
            loss_configs=loss_configs,
        )
    )

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=policy_architecture,
    )


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with medium shaped reward environment."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Replay with medium shaped reward environment."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate(
    policy_uris: Sequence[str] | str | None = None,
    eval_simulations: Optional[Sequence[SimulationConfig]] = None,
) -> EvaluateTool:
    """Evaluate with medium shaped reward environments."""
    sims = list(eval_simulations) if eval_simulations is not None else simulations()

    if policy_uris is None:
        normalized_policy_uris: list[str] = []
    elif isinstance(policy_uris, str):
        normalized_policy_uris = [policy_uris]
    else:
        normalized_policy_uris = list(policy_uris)

    return EvaluateTool(
        simulations=sims,
        policy_uris=normalized_policy_uris,
    )


def evaluate_remote(
    policy_uri: str,
    eval_simulations: Optional[Sequence[SimulationConfig]] = None,
) -> EvalRemoteTool:
    """Remote evaluation with medium shaped reward environments."""
    sims = list(eval_simulations) if eval_simulations is not None else simulations()
    return EvalRemoteTool(
        simulations=sims,
        policy_uri=policy_uri,
    )


# Sweep section

SWEEP_EVAL_SUITE = "sweep_arena_medium_shaped"


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    basic_env = mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    simulations_list = [
        SimulationConfig(suite=SWEEP_EVAL_SUITE, name="basic", env=basic_env),
        SimulationConfig(suite=SWEEP_EVAL_SUITE, name="combat", env=combat_env),
    ]

    return EvaluateTool(
        simulations=simulations_list,
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> SweepTool:
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
        # These two custom parameters are handled by the train function of this recipe
        SP.param("temperature", D.UNIFORM, min=0, max=0.4, search_center=0.07),
        SP.param("contrastive_coef", D.UNIFORM, min=0.0001, max=1, search_center=0.2),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="experiments.recipes.arena_medium_shaped",
        train_entrypoint="train",
        train_overrides={"enable_contrastive": True},
        eval_entrypoint="evaluate_in_sweep",
        objective=f"evaluator/eval_{SWEEP_EVAL_SUITE}/score",
        parameters=parameters,
        num_trials=80,
        num_parallel_trials=4,
    )
