"""Arena recipe with contrastive loss enabled and sparse rewards: ore -> battery -> heart."""

from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)    # Create trainer config with contrastive loss enabled + PPO for actions
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.ppo import PPOConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.sweep import SweepTool, SweepSchedulerType
from experiments.sweeps.protein_configs import PPO_BASIC, make_custom_protein_config
from metta.sweep.protein_config import ParameterConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create arena environment with sparse rewards: only heart gives reward."""
    arena_env = eb.make_arena(num_agents=num_agents)

    # Sparse rewards: only final objective (heart) gives reward
    # Remove all intermediate rewards
    arena_env.game.agent.rewards.inventory["ore_red"] = 0.0
    arena_env.game.agent.rewards.inventory["battery_red"] = 0.0
    arena_env.game.agent.rewards.inventory["laser"] = 0.0
    arena_env.game.agent.rewards.inventory["armor"] = 0.0
    arena_env.game.agent.rewards.inventory["blueprint"] = 0.0

    # Only heart gives reward (final objective)
    arena_env.game.agent.rewards.inventory["heart"] = 1.0
    arena_env.game.agent.rewards.inventory_max["heart"] = 100  # Allow accumulation

    # Set up the resource chain: ore -> battery -> heart
    # Ensure converter processes: mine ore, use ore to make battery, use battery to make heart
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1  # battery -> heart conversion

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create curriculum with sparse reward environment."""
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # Only vary heart rewards (final objective) in curriculum
    arena_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.5, 1.0, 2.0])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.heart", [10, 50, 100])

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


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation environments with sparse rewards."""
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena_sparse", name="basic", env=basic_env),
        SimulationConfig(suite="arena_sparse", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    losses: Optional[LossConfig] = None,
    enable_contrastive: bool = False,
) -> TrainTool:
    """Train with sparse rewards and optional contrastive loss."""
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )
    if not losses:
        contrastive_config = ContrastiveConfig(
            temperature=0.07,
            contrastive_coef=0.1,
            embedding_dim=128,
            use_projection_head=True,
            log_similarities=True,
            log_frequency=1,  # Log every epoch instead of every 100 epochs
        )

        ppo_config = PPOConfig()  # Default PPO config for action generation

        loss_configs = {"ppo": ppo_config}  # PPO generates actions
        if enable_contrastive:
            loss_configs["contrastive"] = (
                contrastive_config  # Only add contrastive if enabled
            )
        losses = LossConfig(
            enable_contrastive=enable_contrastive,
            loss_configs=loss_configs,
        )

    trainer_config = TrainerConfig(
        losses=LossConfig(
            enable_contrastive=enable_contrastive,
            loss_configs=loss_configs,
        )
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Interactive play with sparse reward environment."""
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="arena_sparse", env=eval_env, name="eval")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay with sparse reward environment."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="arena_sparse", env=eval_env, name="eval")
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
    ) -> SimTool:
    """Evaluate with sparse reward environments."""
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def evaluate_remote(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> EvalRemoteTool:
    """Remote evaluation with sparse reward environments."""
    simulations = simulations or make_evals()
    return EvalRemoteTool(
        simulations=simulations,
        policy_uri=policy_uri,
    )

def sweep_async_progressive(
    min_timesteps: int,
    max_timesteps: int,
    initial_timesteps: int,
    max_concurrent_evals: int = 1,
    liar_strategy: str = "best",
    enable_contrastive: bool = True,
) -> SweepTool:
    """Async-capped sweep with progressive timesteps for sparse + contrastive arena.

    - Sweeps over trainer.total_timesteps from [min_timesteps, max_timesteps],
      starting around initial_timesteps (log-normal).
    - Uses AsyncCapped scheduler: fills training slots, serializes evals,
      applies Constant-Liar fantasies.
    - Enables contrastive loss by default.
    """

    protein_cfg = make_custom_protein_config(
        PPO_BASIC,
        {
            # Progressive training horizon
            "trainer.total_timesteps": ParameterConfig(
                min=min_timesteps,
                max=max_timesteps,
                distribution="log_normal",
                mean=initial_timesteps,
                scale="auto",
            ),
            # Contrastive hyperparameters
            "losses.loss_configs.contrastive.temperature": ParameterConfig(
                min=0.02,
                max=0.5,
                distribution="log_normal",
                mean=0.1,
                scale="auto",
            ),
            "losses.loss_configs.contrastive.contrastive_coef": ParameterConfig(
                min=0.01,
                max=1.0,
                distribution="log_normal",
                mean=0.35,
                scale="auto",
            ),
        },
    )

    return SweepTool(
        protein_config=protein_cfg,
        recipe_module="experiments.recipes.arena_with_contrastive_sparse_rewards",
        train_entrypoint="train",
        eval_entrypoint="evaluate_remote",
        # Turn on contrastive loss during training; specifics remain default unless swept
        train_overrides={
            "losses.enable_contrastive": "True",
        },
        scheduler_type=SweepSchedulerType.ASYNC_CAPPED,
        max_concurrent_evals=max_concurrent_evals,
        liar_strategy=liar_strategy,
    )
