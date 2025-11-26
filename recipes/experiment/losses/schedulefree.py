"""Arena recipe with ScheduleFree AdamW optimizer for comparison testing."""

from typing import Optional, Sequence

from metta.rl.trainer_config import OptimizerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.eval import EvaluateTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from recipes.experiment.arena import mettagrid, simulations
from recipes.prod.arena_basic_easy_shaped import BASELINE as ARENA_BASELINE


def train(
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with ScheduleFree AdamW optimizer.

    This uses the same configuration as the base arena recipe but with
    ScheduleFree AdamW optimizer instead of regular Adam.
    """
    return ARENA_BASELINE.model_copy(deep=True)


def train_shaped(rewards: bool = True) -> TrainTool:
    """Train with ScheduleFree AdamW optimizer on shaped rewards task.

    This provides easier training with reward shaping.
    """
    from recipes.experiment.arena import train_shaped as base_train_shaped

    baseline = ARENA_BASELINE.model_copy(deep=True)
    base_tool = base_train_shaped(rewards=rewards)
    baseline.training_env.curriculum = base_tool.training_env.curriculum

    baseline.trainer.optimizer = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=2000,
    )

    return baseline


def evaluate(policy_uris: Optional[Sequence[str]] = None) -> EvaluateTool:
    """Evaluate policies on arena simulations."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Evaluation tool for sweep runs with ScheduleFree optimizer.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    """
    # Create sweep-optimized versions of the standard evaluations
    basic_env = mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    simulations_list = [
        SimulationConfig(
            suite="sweep",
            name="basic",
            env=basic_env,
            num_episodes=10,
            max_time_s=240,
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
        simulations=simulations_list,
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> SweepTool:
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
        SP.LEARNING_RATE,
        # Weight decay - important for AdamW variant
        SP.param(
            "trainer.optimizer.weight_decay",
            D.LOG_NORMAL,
            min=1e-4,
            max=1e-1,
            search_center=1e-2,
        ),
        # Warmup steps - ScheduleFree specific
        SP.param(
            "trainer.optimizer.warmup_steps",
            D.INT_UNIFORM,
            min=500,
            max=5000,
            search_center=1000,
        ),
        # Standard PPO parameters
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.ADAM_EPS,
        # Training duration
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
        recipe="recipes.experiment.losses.schedulefree",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        max_trials=80,
        num_parallel_trials=4,
    )
