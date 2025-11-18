"""Small-map CoGs vs Clips entry points."""


from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from recipes.experiment.cogs_v_clips import play, train_small_maps
from recipes.experiment.cogs_v_clips import train_small_maps as train


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
    """Comprehensive hyperparameter sweep for CvC small maps.

    This sweep explores all important hyperparameters while avoiding problematic
    parameters like batch sizes that can cause constraint violations.

    Example usage:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep \
            sweep_name="user.cvc.20241117" -- gpus=4 nodes=2

    Local testing:
        uv run ./tools/run.py recipes.prod.cvc.small_maps.sweep \
            sweep_name="user.cvc.test" -- local_test=True
    """

    # Comprehensive parameter set (excluding batch/minibatch/bptt/epochs)
    parameters = [
        # Training duration
        SP.param(
            "trainer.total_timesteps",
            D.LOG_NORMAL,
            min=5e8,
            max=5e9,
            search_center=1e9,
        ),

        # Core learning rate
        SP.LEARNING_RATE,

        # PPO loss parameters
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

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.cvc.small_maps",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        objective="env_agent/heart.gained",
        cost_metric="metric/agent_step",
        parameters=parameters,
        max_trials=900,  # Large number of trials for comprehensive exploration
        num_parallel_trials=4,
    )


__all__ = ["train", "play", "train_sweep", "evaluate_stub", "sweep"]
