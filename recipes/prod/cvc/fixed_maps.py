"""Fixed-map CoGs vs Clips prod entry point."""

from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_fixed_maps as train
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool


def evaluate_stub(*args, **kwargs) -> StubTool:
    """Stub evaluation for sweep (skip evaluation)."""
    return StubTool()


def sweep(sweep_name: str) -> SweepTool:
    """Sweep function for fixed-map CoGs vs Clips training.

    Example usage:
        `uv run ./tools/run.py recipes.prod.cvc.fixed_maps.sweep \
            sweep_name="user.cvc.20241119" -- gpus=4 nodes=2`

    This sweep is configured for schedule-free optimizer with a wider parameter search space.
    It trains with num_cogs=4 and variants=["heart_chorus","pack_rat"] by default.

    For local testing before remote deployment:
        `uv run ./tools/run.py recipes.prod.cvc.fixed_maps.sweep \
            sweep_name="user.cvc.20241119.local_test" -- local_test=True`
    """

    # Wide parameter set for schedule-free optimizer
    parameters = [
        # Standard PPO parameters
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.PPO_ENT_COEF,

        # Optimizer parameters with wider ranges for schedule-free
        SP.param(
            "trainer.optimizer.learning_rate",
            D.LOG_NORMAL,
            min=5e-6,
            max=5e-2,
            search_center=5e-4,
        ),
        SP.param(
            "trainer.optimizer.eps",
            D.LOG_NORMAL,
            min=1e-9,
            max=1e-3,
            search_center=1e-7,
        ),

        # Additional optimizer parameters useful for schedule-free
        SP.param(
            "trainer.optimizer.weight_decay",
            D.LOG_NORMAL,
            min=1e-6,
            max=1e-2,
            search_center=1e-4,
        ),

        # Training duration variation
        SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=3e9,
            search_center=1e9,
        ),

        # Discount factor
        SP.param(
            "trainer.losses.ppo.gamma",
            D.UNIFORM,
            min=0.96,
            max=0.999,
            search_center=0.99,
        ),
    ]

    # Fixed overrides for the specific training configuration
    # The sweep system will automatically format variants as 'variants=["heart_chorus","pack_rat"]' in the command
    train_overrides = {
        "num_cogs": 4,
        "variants": '[\"heart_chorus\", \"pack_rat\"]',  # Pass as Python list, sweep handles CLI formatting
        "evaluator.epoch_interval": 0,  # Disable evaluation during training
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.cvc.fixed_maps",
        train_entrypoint="train",
        eval_entrypoint="evaluate_stub",  # Skip evaluation
        objective="env_agent/heart.gained",  # Optimize for hearts gained
        cost_metric="metric/agent_step",  # Track compute cost via agent steps
        parameters=parameters,
        train_overrides=train_overrides,
        max_trials=300,  # Large sweep
        num_parallel_trials=4,  # Run 4 trials in parallel
    )


__all__ = ["train", "play", "evaluate_stub", "sweep"]
