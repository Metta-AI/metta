"""Arena recipe with contrastive loss enabled and sparse rewards: ore -> battery -> heart."""

import typing

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.rl.loss.loss_config
import mettagrid.base_config
import metta.rl.loss.contrastive_config
import metta.rl.loss.ppo
import metta.rl.trainer_config
import metta.rl.training
import metta.sim.simulation_config
import metta.sweep.core
import metta.tools.eval
import metta.tools.eval_remote
import metta.tools.play
import metta.tools.replay
import metta.tools.sweep
import metta.tools.train
import mettagrid


def mettagrid(num_agents: int = 24) -> mettagrid.MettaGridConfig:
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

    return arena_env


def make_curriculum(
    arena_env: typing.Optional[mettagrid.MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig
    ] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    """Create curriculum with sparse reward environment."""
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # Only vary heart rewards (final objective) in curriculum
    arena_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.5, 1.0, 2.0])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.heart", [10, 50, 100])

    # Enable/disable attacks for variety
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(
    env: typing.Optional[mettagrid.MettaGridConfig] = None,
) -> list[metta.sim.simulation_config.SimulationConfig]:
    """Create evaluation environments with sparse rewards."""
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        metta.sim.simulation_config.SimulationConfig(
            suite="arena_sparse", name="basic", env=basic_env
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="arena_sparse", name="combat", env=combat_env
        ),
    ]


def train(
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
    enable_contrastive: bool = True,
    # These parameters can now be swept over.
    temperature: float = 0.07,
    contrastive_coef: float = 0.1,
) -> metta.tools.train.TrainTool:
    """Train with sparse rewards and optional contrastive loss."""
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    contrastive_config = metta.rl.loss.contrastive_config.ContrastiveConfig(
        temperature=temperature,
        contrastive_coef=contrastive_coef,
        embedding_dim=128,
        use_projection_head=True,
    )

    ppo_config = (
        metta.rl.loss.ppo.PPOConfig()
    )  # Default PPO config for action generation

    loss_configs: dict[str, mettagrid.base_config.Config] = {"ppo": ppo_config}
    if enable_contrastive:
        loss_configs["contrastive"] = contrastive_config

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        losses=metta.rl.loss.loss_config.LossConfig(
            enable_contrastive=enable_contrastive,
            loss_configs=loss_configs,
        )
    )

    return metta.tools.train.TrainTool(
        trainer=trainer_config,
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=metta.rl.training.EvaluatorConfig(simulations=simulations()),
    )


def play(policy_uri: typing.Optional[str] = None) -> metta.tools.play.PlayTool:
    """Interactive play with sparse reward environment."""
    return metta.tools.play.PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: typing.Optional[str] = None) -> metta.tools.replay.ReplayTool:
    """Replay with sparse reward environment."""
    return metta.tools.replay.ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate(
    policy_uris: typing.Sequence[str] | str | None = None,
    eval_simulations: typing.Optional[
        typing.Sequence[metta.sim.simulation_config.SimulationConfig]
    ] = None,
) -> metta.tools.eval.EvaluateTool:
    """Evaluate with sparse reward environments."""
    sims = list(eval_simulations) if eval_simulations is not None else simulations()

    if policy_uris is None:
        normalized_policy_uris: list[str] = []
    elif isinstance(policy_uris, str):
        normalized_policy_uris = [policy_uris]
    else:
        normalized_policy_uris = list(policy_uris)

    return metta.tools.eval.EvaluateTool(
        simulations=sims,
        policy_uris=normalized_policy_uris,
    )


def evaluate_remote(
    policy_uri: str,
    eval_simulations: typing.Optional[
        typing.Sequence[metta.sim.simulation_config.SimulationConfig]
    ] = None,
) -> metta.tools.eval_remote.EvalRemoteTool:
    """Remote evaluation with sparse reward environments."""
    sims = list(eval_simulations) if eval_simulations is not None else simulations()
    return metta.tools.eval_remote.EvalRemoteTool(
        simulations=sims,
        policy_uri=policy_uri,
    )


# Sweep section

SWEEP_EVAL_SUITE = "sweep_arena_sparse"


def evaluate_in_sweep(policy_uri: str) -> metta.tools.eval.EvaluateTool:
    basic_env = mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    simulations = [
        metta.sim.simulation_config.SimulationConfig(
            suite=SWEEP_EVAL_SUITE, name="basic", env=basic_env
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite=SWEEP_EVAL_SUITE, name="combat", env=combat_env
        ),
    ]

    return metta.tools.eval.EvaluateTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> metta.tools.sweep.SweepTool:
    parameters = [
        metta.sweep.core.SweepParameters.LEARNING_RATE,
        metta.sweep.core.SweepParameters.PPO_CLIP_COEF,
        metta.sweep.core.SweepParameters.PPO_GAE_LAMBDA,
        metta.sweep.core.SweepParameters.PPO_VF_COEF,
        metta.sweep.core.SweepParameters.ADAM_EPS,
        metta.sweep.core.SweepParameters.param(
            "trainer.total_timesteps",
            metta.sweep.core.Distribution.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=7.5e8,
        ),
        # These two custom parameters are handled by the train function of this recipe,
        # and are therefore sweepable.
        metta.sweep.core.SweepParameters.param(
            "temperature",
            metta.sweep.core.Distribution.UNIFORM,
            min=0,
            max=0.4,
            search_center=0.07,
        ),
        metta.sweep.core.SweepParameters.param(
            "contrastive_coef",
            metta.sweep.core.Distribution.UNIFORM,
            min=0.0001,
            max=1,
            search_center=0.2,
        ),
    ]

    return metta.sweep.core.make_sweep(
        name=sweep_name,
        recipe="experiments.recipes.arena_with_sparse_rewards",
        train_entrypoint="train",
        # We can set global overrides for training here.
        # These are passed via the CLI
        train_overrides={"enable_contrastive": True},
        eval_entrypoint="evaluate_in_sweep",
        objective=f"evaluator/eval_{SWEEP_EVAL_SUITE}/score",
        parameters=parameters,
        max_trials=80,
        num_parallel_trials=4,
    )
