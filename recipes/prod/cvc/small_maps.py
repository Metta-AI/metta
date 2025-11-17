"""Small-map CoGs vs Clips entry points."""

from typing import Optional, Sequence

from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_small_maps as train
from recipes.experiment.cogs_v_clips import train_small_maps


def train_sweep():
    """Training configuration optimized for sweeps with evaluations disabled.

    This is identical to train() but configured for hyperparameter sweeps with
    fixed num_cogs=4 and variants for CvC small maps.
    Evaluations are performed separately after training completes.
    """
    # Fixed configuration for CvC small map sweeps
    return train_small_maps(
        num_cogs=4,
        variants=["heart_chorus", "pack_rat", "lonely_heart"],
        eval_variants=None,
        eval_difficulty="standard",
        mission=None,
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    """Stub evaluator for sweep runs.

    During sweeps, we disable in-training evaluations for speed.
    Actual evaluation happens through the sweep orchestrator.
    """
    return StubTool()


def sweep(sweep_name: str):
    """Hyperparameter sweep for CvC small maps with balanced parameter set.

    This is the main sweep function with a carefully selected set of important
    hyperparameters. For a more comprehensive exploration, use sweep_comprehensive.

    CONSTRAINTS:
    - minibatch_size must be <= batch_size
    - batch_size must be divisible by minibatch_size
    - Using power-of-2 distributions helps ensure divisibility
    - Invalid combinations will fail quickly and Bayesian optimization will learn to avoid them

    Example usage:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep \
            sweep_name="user.cvc_small.20241117" -- gpus=4 nodes=2

    Local testing:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep \
            sweep_name="user.cvc_small.test" -- local_test=True
    """

    # Balanced set of important parameters
    parameters = [
        # Core learning parameters
        SP.LEARNING_RATE,

        # Training dynamics - using power-of-2 to help ensure divisibility constraints
        SP.param(
            "trainer.batch_size",
            D.UNIFORM_POW2,  # Powers of 2 help ensure divisibility
            min=8192,  # Raised minimum to reduce chance of minibatch > batch
            max=524288,
            search_center=65536,
        ),
        SP.param(
            "trainer.minibatch_size",
            D.UNIFORM_POW2,  # Powers of 2 help ensure batch % minibatch == 0
            min=512,
            max=32768,  # Reduced max to stay below typical batch sizes
            search_center=4096,
        ),

        # PPO-specific parameters
        SP.PPO_CLIP_COEF,
        SP.PPO_ENT_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.param(
            "trainer.losses.ppo.gamma",
            D.LOGIT_NORMAL,
            min=0.8,
            max=0.9999,
            search_center=0.98,
        ),

        # Optimizer parameters (for schedule_free optimizer)
        SP.ADAM_EPS,
        SP.param(
            "trainer.optimizer.beta1",
            D.LOGIT_NORMAL,
            min=0.5,
            max=0.999,
            search_center=0.9,
        ),
    ]

    # Disable in-training evaluations for sweep efficiency
    train_overrides = {
        'evaluator.epoch_interval': 0,
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.cvc.small_maps",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",  # Using stub for now
        objective="env_agent/heart.gained",  # Primary optimization metric
        cost_metric="metric/agent_step",  # Cost tracking metric
        parameters=parameters,
        max_trials=80,
        num_parallel_trials=2,
        train_overrides=train_overrides,
    )


def sweep_quick(sweep_name: str):
    """Quick sweep for testing configurations with shorter training runs.

    Uses reduced timesteps and a minimal parameter set for rapid iteration
    and testing of hyperparameter configurations. Good for verifying sweep
    setup before running full trials.

    Example usage:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep_quick \
            sweep_name="user.cvc_small.quick" -- local_test=True
    """

    # Minimal but important parameter space for quick testing
    parameters = [
        # Core parameters
        SP.LEARNING_RATE,
        SP.PPO_CLIP_COEF,
        SP.PPO_ENT_COEF,
        SP.PPO_GAE_LAMBDA,

        # Add one batch size parameter to test training dynamics
        SP.param(
            "trainer.minibatch_size",
            D.UNIFORM_POW2,
            min=1024,
            max=8192,
            search_center=4096,
        ),
    ]

    # Configuration with reduced training time for testing
    train_overrides = {
        'trainer.total_timesteps': 100_000,  # Quick training for testing
        'evaluator.epoch_interval': 0,  # Disable in-training evaluations
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.cvc.small_maps",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        objective="env_agent/heart.gained",
        cost_metric="metric/agent_step",
        parameters=parameters,
        max_trials=10,  # Fewer trials for quick testing
        num_parallel_trials=1,
        train_overrides=train_overrides,
    )


def sweep_comprehensive(sweep_name: str):
    """Comprehensive hyperparameter sweep with extensive parameter exploration.

    This sweep explores a very large parameter space, similar to the reference
    configuration. Use this for thorough hyperparameter optimization when you
    have significant compute resources available.

    CONSTRAINTS & CONSIDERATIONS:
    - minibatch_size must be <= batch_size and divide it evenly
    - batch_size should be divisible by (num_envs / num_workers)
    - bptt_horizon affects memory usage and gradient quality
    - Using power-of-2 distributions increases compatibility
    - The Bayesian optimizer will learn to avoid invalid combinations

    Example usage:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep_comprehensive \
            sweep_name="user.cvc_comprehensive.20241117" -- gpus=8 nodes=4
    """

    # Extensive parameter space exploration
    parameters = [
        # Training duration
        SP.param(
            "trainer.total_timesteps",
            D.LOG_NORMAL,
            min=5e8,
            max=5e9,
            search_center=1e8,
        ),

        # Core learning parameters
        SP.LEARNING_RATE,

        # Training dynamics - carefully chosen ranges to reduce invalid combinations
        SP.param(
            "trainer.batch_size",
            D.UNIFORM_POW2,
            min=16384,  # Higher minimum for comprehensive sweep
            max=1048576,  # 1M batch size
            search_center=131072,
        ),
        SP.param(
            "trainer.minibatch_size",
            D.UNIFORM_POW2,
            min=1024,  # Reasonable minimum
            max=65536,  # Well below max batch_size
            search_center=8192,
        ),
        SP.param(
            "trainer.bptt_horizon",
            D.UNIFORM_POW2,
            min=16,
            max=128,  # Typical range for sequence learning
            search_center=64,
        ),
        SP.param(
            "trainer.update_epochs",
            D.INT_UNIFORM,
            min=1,
            max=4,
            search_center=2,
        ),

        # PPO core parameters
        SP.PPO_CLIP_COEF,
        SP.PPO_ENT_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.param(
            "trainer.losses.ppo.gamma",
            D.LOGIT_NORMAL,
            min=0.8,
            max=0.9999,
            search_center=0.98,
        ),
        SP.param(
            "trainer.losses.ppo.max_grad_norm",
            D.UNIFORM,
            min=0.1,
            max=5.0,
            search_center=1.0,
        ),
        SP.param(
            "trainer.losses.ppo.vf_clip_coef",
            D.UNIFORM,
            min=0.1,
            max=5.0,
            search_center=0.2,
        ),

        # Optimizer parameters (schedule_free AdamW)
        SP.ADAM_EPS,
        SP.param(
            "trainer.optimizer.beta1",
            D.LOGIT_NORMAL,
            min=0.5,
            max=0.999,
            search_center=0.9,
        ),
        SP.param(
            "trainer.optimizer.beta2",
            D.LOGIT_NORMAL,
            min=0.9,
            max=0.99999,
            search_center=0.999,
        ),
        SP.param(
            "trainer.optimizer.weight_decay",
            D.LOG_NORMAL,
            min=0.0001,
            max=0.1,
            search_center=0.01,
        ),
        SP.param(
            "trainer.optimizer.warmup_steps",
            D.INT_UNIFORM,
            min=100,
            max=10000,
            search_center=1000,
        ),
    ]

    # Disable in-training evaluations for sweep efficiency
    train_overrides = {
        'evaluator.epoch_interval': 0,
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.cvc.small_maps",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        objective="env_agent/heart.gained",
        cost_metric="metric/agent_step",
        parameters=parameters,
        max_trials=900,  # More trials for comprehensive exploration
        num_parallel_trials=1,  # More parallel trials
        train_overrides=train_overrides,
    )


def sweep_safe(sweep_name: str):
    """Conservative sweep with restricted ranges to minimize constraint violations.

    This sweep uses carefully chosen parameter ranges and relationships to
    reduce the likelihood of invalid batch size combinations while still using
    continuous distributions for Bayesian optimization.

    Strategies:
    - Batch size minimum is high (16k) to leave room for minibatch sizes
    - Minibatch size maximum is limited to typical values (16k)
    - Both use power-of-2 distributions for better divisibility
    - BPTT horizon is kept in a reasonable range

    Example usage:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep_safe \
            sweep_name="user.cvc_safe.20241117" -- gpus=4 nodes=2
    """

    parameters = [
        # Core learning parameters
        SP.LEARNING_RATE,

        # Batch sizes with conservative ranges
        SP.param(
            "trainer.batch_size",
            D.UNIFORM_POW2,
            min=16384,  # Start high to leave room for minibatch
            max=262144,  # Reasonable max for most GPUs
            search_center=65536,
        ),
        SP.param(
            "trainer.minibatch_size",
            D.UNIFORM_POW2,
            min=1024,  # Reasonable minimum
            max=16384,  # Conservative max, well below min batch_size
            search_center=4096,
        ),

        # BPTT with standard range
        SP.param(
            "trainer.bptt_horizon",
            D.UNIFORM_POW2,
            min=32,
            max=64,  # Conservative range
            search_center=64,
        ),

        # PPO parameters
        SP.PPO_CLIP_COEF,
        SP.PPO_ENT_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.param(
            "trainer.losses.ppo.gamma",
            D.LOGIT_NORMAL,
            min=0.9,
            max=0.999,
            search_center=0.98,
        ),

        # Optimizer
        SP.ADAM_EPS,
    ]

    train_overrides = {
        'evaluator.epoch_interval': 0,
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.cvc.small_maps",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        objective="env_agent/heart.gained",
        cost_metric="metric/agent_step",
        parameters=parameters,
        max_trials=80,
        num_parallel_trials=2,
        train_overrides=train_overrides,
    )


__all__ = ["train", "play", "train_sweep", "evaluate_stub", "sweep", "sweep_quick", "sweep_comprehensive", "sweep_safe"]
