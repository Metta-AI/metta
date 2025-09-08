"""Standard sweep configurations using the new orchestrator pattern."""

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep import SweepTool


def ppo(
    run: str | None = None,  # Accept run parameter from dispatcher (unused)
    recipe: str = "experiments.recipes.arena",
    train: str = "train_shaped",
    eval: str = "evaluate",
    max_trials: int = 10,
    max_parallel_jobs: int = 1,
    gpus: int = 1,
) -> SweepTool:
    """Create PPO hyperparameter sweep."""

    # Define the 5 PPO parameters to sweep over
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",  # Metric to optimize
        goal="maximize",
        method="bayes",  # Use Bayesian optimization
        parameters={
            # 1. Learning rate - log scale from 1e-5 to 1e-2
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=1e-3,  # Geometric mean
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
                mean=0.001,  # Geometric mean
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
        },
        settings=ProteinSettings(
            num_random_samples=20,  # Start with 20 random samples for better exploration in large sweeps
            max_suggestion_cost=7200,  # 5 minutes max per trial (for quick testing)
        ),
    )

    # Create and return the orchestrator tool
    return SweepTool(
        protein_config=protein_config,
        max_trials=max_trials,
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        monitoring_interval=60,
        max_parallel_jobs=max_parallel_jobs,
        gpus=gpus,
        train_overrides={
            "trainer.total_timesteps": "1000000000",  # 1B timesteps default for PPO sweep
        },
    )


def quick_test(
    run: str | None = None,  # Accept run parameter from dispatcher (unused)
    recipe: str = "experiments.recipes.arena",
) -> SweepTool:
    """Quick test sweep with full PPO config but minimal trials for testing.

    This sweep uses the same full PPO parameter configuration as the main ppo()
    function but runs fewer trials with shorter training for quick testing
    that the orchestrator infrastructure is working.

    Args:
        recipe: Module path to the recipe (sweep_name comes from args)

    Returns:
        Configured SweepTool for quick testing

    Example:
        uv run ./tools/run.py experiments.sweeps.standard.quick_test \
            --args sweep_name=test_sweep
    """

    # Use the SAME full PPO config as the main ppo() function
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",
        goal="maximize",
        method="bayes",  # Use Bayesian optimization (Protein)
        parameters={
            # 1. Learning rate - log scale from 1e-5 to 1e-2
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=1e-3,  # Geometric mean
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
                mean=0.001,  # Geometric mean
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
        },
        settings=ProteinSettings(
            num_random_samples=20,  # Start with 20 random samples for better exploration in large sweeps
            max_suggestion_cost=300,  # 5 minutes max per trial (for quick testing)
        ),
    )

    tool = SweepTool(
        # sweep_name is not set here - it will come from args
        protein_config=protein_config,
        max_trials=5,  # Only 5 trials for quick testing
        recipe_module=recipe,
        train_entrypoint="train_shaped",
        eval_entrypoint="evaluate",
        monitoring_interval=90,
        max_parallel_jobs=1,
        train_overrides={
            "trainer.total_timesteps": "50000",  # Quick 10k timesteps for testing
        },
    )

    return tool
