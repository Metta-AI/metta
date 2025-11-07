# ruff: noqa: E501
"""Arena recipe with ScheduleFree AdamW optimizer for comparison testing."""

import typing

import metta.rl.trainer_config
import metta.rl.training
import metta.sim.simulation_config
import metta.sweep.core
import metta.tools.eval
import metta.tools.sweep
import metta.tools.train

# Import everything from the base arena recipe
import experiments.recipes.arena


def train(
    enable_detailed_slice_logging: bool = False,
) -> metta.tools.train.TrainTool:
    """Train with ScheduleFree AdamW optimizer.

    This uses the same configuration as the base arena recipe but with
    ScheduleFree AdamW optimizer instead of regular Adam.
    """
    curriculum = experiments.recipes.arena.make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    # Configure ScheduleFree AdamW optimizer
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.001153637,  # Same as default
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,  # Small weight decay for AdamW
        warmup_steps=1000,  # Warmup steps for ScheduleFree
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=metta.rl.training.EvaluatorConfig(
            simulations=experiments.recipes.arena.simulations()
        ),
    )


def train_shaped(rewards: bool = True) -> metta.tools.train.TrainTool:
    """Train with ScheduleFree AdamW optimizer on shaped rewards task.

    This provides easier training with reward shaping.
    """

    # Get the base shaped training tool
    base_tool = experiments.recipes.arena.train_shaped(rewards=rewards)

    # Configure ScheduleFree AdamW optimizer (using native implementation)
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,  # Same as default
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,  # Small weight decay for AdamW
        warmup_steps=2000,  # Warmup steps for ScheduleFree
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    # Return a new TrainTool with the shaped environment but ScheduleFree optimizer
    return metta.tools.train.TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
    )


def evaluate(
    policy_uris: typing.Optional[typing.Sequence[str]] = None,
) -> metta.tools.eval.EvaluateTool:
    """Evaluate policies on arena simulations."""
    return metta.tools.eval.EvaluateTool(
        simulations=experiments.recipes.arena.simulations(),
        policy_uris=policy_uris or [],
    )


def evaluate_in_sweep(policy_uri: str) -> metta.tools.eval.EvaluateTool:
    """Evaluation tool for sweep runs with ScheduleFree optimizer.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    """
    # Create sweep-optimized versions of the standard evaluations
    basic_env = experiments.recipes.arena.mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    simulations_list = [
        metta.sim.simulation_config.SimulationConfig(
            suite="sweep",
            name="basic",
            env=basic_env,
            num_episodes=10,
            max_time_s=240,
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="sweep",
            name="combat",
            env=combat_env,
            num_episodes=10,
            max_time_s=240,
        ),
    ]

    return metta.tools.eval.EvaluateTool(
        simulations=simulations_list,
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> metta.tools.sweep.SweepTool:
    """Hyperparameter sweep for ScheduleFree optimizer.

    This sweep explores ScheduleFree-specific parameters along with standard RL hyperparameters.

    Example usage:
        # Local test first:
        uv run ./tools/run.py losses.schedulefree.sweep \\
            sweep_name="schedulefree.test" -- local_test=True

        # Production sweep:
        uv run ./tools/run.py losses.schedulefree.sweep \\
            sweep_name="schedulefree.production" -- gpus=4 nodes=2

    Key ScheduleFree parameters:
        - warmup_steps: Number of warmup steps for the learning rate schedule
        - weight_decay: L2 regularization strength
        - learning_rate: Base learning rate
    """
    # ScheduleFree-specific and general RL parameters
    parameters = [
        # Learning rate - critical for ScheduleFree
        metta.sweep.core.SweepParameters.LEARNING_RATE,
        # Weight decay - important for AdamW variant
        metta.sweep.core.SweepParameters.param(
            "trainer.optimizer.weight_decay",
            metta.sweep.core.Distribution.LOG_NORMAL,
            min=1e-4,
            max=1e-1,
            search_center=1e-2,
        ),
        # Warmup steps - ScheduleFree specific
        metta.sweep.core.SweepParameters.param(
            "trainer.optimizer.warmup_steps",
            metta.sweep.core.Distribution.INT_UNIFORM,
            min=500,
            max=5000,
            search_center=1000,
        ),
        # Standard PPO parameters
        metta.sweep.core.SweepParameters.PPO_CLIP_COEF,
        metta.sweep.core.SweepParameters.PPO_GAE_LAMBDA,
        metta.sweep.core.SweepParameters.PPO_VF_COEF,
        metta.sweep.core.SweepParameters.ADAM_EPS,
        # Training duration
        metta.sweep.core.SweepParameters.param(
            "trainer.total_timesteps",
            metta.sweep.core.Distribution.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=7.5e8,
        ),
    ]

    return metta.sweep.core.make_sweep(
        name=sweep_name,
        recipe="experiments.recipes.losses.schedulefree",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        max_trials=80,
        num_parallel_trials=4,
    )
