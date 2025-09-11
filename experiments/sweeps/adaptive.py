"""Adaptive-controller-based sweep matching standard.ppo.

This experiment reproduces the standard PPO sweep using the new adaptive
controller + batched-synced scheduler with Protein optimizer.
"""

from __future__ import annotations

from metta.adaptive.optimizer.protein import ProteinOptimizer  # noqa: F401 (for type hints in IDEs)
from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.schedulers.batched_synced import BatchedSyncedSchedulerConfig
from metta.tools.adaptive import AdaptiveTool, DispatcherType, SchedulerType


def ppo(
    run: str | None = None,  # Accept run parameter from dispatcher (unused)
    recipe: str = "experiments.recipes.arena_basic_easy_shaped",
    train: str = "train",
    eval: str = "evaluate_in_sweep",
    max_trials: int = 300,
    max_parallel_jobs: int = 6,
    gpus: int = 1,
) -> AdaptiveTool:
    """Create PPO hyperparameter sweep using adaptive controller.

    Matches experiments.sweeps.standard.ppo parameters and behavior.
    """

    # Define the 6 PPO parameters to sweep over (same as standard.ppo)
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",
        goal="maximize",
        method="bayes",
        parameters={
            # 1. Learning rate - log scale from 1e-5 to 1e-2
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=1e-3,
                scale="auto",
            ),
            # 2. PPO clip coefficient - uniform from 0.05 to 0.3
            "trainer.losses.loss_configs.ppo.clip_coef": ParameterConfig(
                min=0.05,
                max=0.3,
                distribution="uniform",
                mean=0.175,
                scale="auto",
            ),
            # 3. Entropy coefficient - log scale from 0.0001 to 0.01
            "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
                min=0.0001,
                max=0.01,
                distribution="log_normal",
                mean=0.001,
                scale="auto",
            ),
            # 4. GAE lambda - uniform from 0.8 to 0.99
            "trainer.losses.loss_configs.ppo.gae_lambda": ParameterConfig(
                min=0.8,
                max=0.99,
                distribution="uniform",
                mean=0.895,
                scale="auto",
            ),
            # 5. Value function coefficient - uniform from 0.1 to 1.0
            "trainer.losses.loss_configs.ppo.vf_coef": ParameterConfig(
                min=0.1,
                max=1.0,
                distribution="uniform",
                mean=0.55,
                scale="auto",
            ),
            # 6. Adam epsilon - log scale from 1e-8 to 1e-4
            "trainer.optimizer.eps": ParameterConfig(
                min=1e-8,
                max=1e-4,
                distribution="log_normal",
                mean=1e-6,
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=20,
            max_suggestion_cost=7200 * 1.5,
        ),
    )

    # Typed scheduler configuration (batched-synced)
    scheduler_config = BatchedSyncedSchedulerConfig(
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        max_trials=max_trials,
        gpus=gpus,
        nodes=1,
        batch_size=max_parallel_jobs,
        experiment_id="ppo_sweep_adaptive",
        train_overrides={
            "trainer.total_timesteps": "2000000000",
        },
    )

    adaptive_cfg = AdaptiveConfig(
        max_parallel=max_parallel_jobs,
        monitoring_interval=60,
        resume=False,
    )

    return AdaptiveTool(
        scheduler_type=SchedulerType.BATCHED_SYNCED,
        scheduler_config=scheduler_config,
        protein_config=protein_config,
        config=adaptive_cfg,
        dispatcher_type=DispatcherType.SKYPILOT,
        experiment_id=scheduler_config["experiment_id"],
    )
