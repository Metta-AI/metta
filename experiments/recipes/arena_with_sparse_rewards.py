"""Arena recipe with contrastive loss enabled and sparse rewards: ore -> battery -> heart."""

from typing import Optional, Sequence

from experiments.sweeps.protein_configs import PPO_CORE, make_custom_protein_config
import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.protein_config import ParameterConfig
from metta.tools.eval import EvaluateTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool, SweepSchedulerType
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.ppo import PPOConfig


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
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
    arena_env = arena_env or mettagrid()

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


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation environments with sparse rewards."""
    basic_env = env or mettagrid()
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
    enable_contrastive: bool = True,
    temperature: float = 0.07,
    contrastive_coef: float = 0.1,
) -> TrainTool:
    """Train with sparse rewards and optional contrastive loss."""
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    contrastive_config = ContrastiveConfig(
        temperature=temperature,
        contrastive_coef=contrastive_coef,
        embedding_dim=128,
        use_projection_head=True,
    )

    ppo_config = PPOConfig()  # Default PPO config for action generation

    loss_configs = {"ppo": ppo_config}
    if enable_contrastive:
        loss_configs["contrastive"] = contrastive_config

    trainer_config = TrainerConfig(
        losses=LossConfig(
            enable_contrastive=enable_contrastive,
            loss_configs=loss_configs,
        )
    )

    return TrainTool(
        trainer=trainer_config,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=simulations()),
    )


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with sparse reward environment."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Replay with sparse reward environment."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate(
    policy_uris: Sequence[str] | str | None = None,
    eval_simulations: Optional[Sequence[SimulationConfig]] = None,
) -> EvaluateTool:
    """Evaluate with sparse reward environments."""
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
    """Remote evaluation with sparse reward environments."""
    sims = list(eval_simulations) if eval_simulations is not None else simulations()
    return EvalRemoteTool(
        simulations=sims,
        policy_uri=policy_uri,
    )


def sweep_async_progressive(
    min_timesteps: int,
    max_timesteps: int,
    initial_timesteps: int,
    max_concurrent_evals: int = 5,
    liar_strategy: str = "best",
) -> None:
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
    print("This function is deperecated and must be refactored. \n Please look at the sweep function in basic_easy_shaped for reference.")
    return

    # DEPRECATED
    protein_cfg = make_custom_protein_config(
        base_config=PPO_CORE,
        parameters={
            "trainer.total_timesteps": ParameterConfig(
                min=min_timesteps,
                max=max_timesteps,
                distribution="int_uniform",
                mean=initial_timesteps,
                scale="auto",
            ),
            "temperature": ParameterConfig(
                min=0,
                max=0.5,
                distribution="uniform",
                mean=0.07,
                scale="auto",
            ),
            "contrastive_coef": ParameterConfig(
                min=0,
                max=1.0,
                distribution="uniform",
                mean=0.5,
                scale="auto",
            ),
        },
    )
    protein_cfg.metric = "evaluator/eval_arena_sparse/score"
    return SweepTool(
        protein_config=protein_cfg,
        recipe_module="experiments.recipes.arena_with_sparse_rewards",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        scheduler_type=SweepSchedulerType.ASYNC_CAPPED,
        max_concurrent_evals=max_concurrent_evals,
        liar_strategy=liar_strategy,
    )
