"""Standard sweep configurations using the adaptive module."""

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep import SweepTool
from metta.tools.sweep import DispatcherType


def ppo(
    recipe: str = "experiments.recipes.arena",
    train: str = "train",
    eval: str = "evaluate",
    max_trials: int = 300,
    max_parallel_jobs: int = 6,
    gpus: int = 1,
    batch_size: int = 4,
    local_test: bool = False,
) -> SweepTool:
    """Create PPO hyperparameter sweep using adaptive infrastructure.

    Args:
        recipe: Recipe module to use for training and evaluation
        train: Training entrypoint name
        eval: Evaluation entrypoint name
        max_trials: Maximum number of trials to run
        max_parallel_jobs: Maximum parallel jobs
        gpus: Number of GPUs per job
        batch_size: Number of suggestions per batch
        local_test: If True, use local dispatcher with 50k timesteps for testing

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """

    # Define the 6 PPO parameters to sweep over
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",  # Metric to optimize
        goal="maximize",
        method="bayes",  # Use Bayesian optimization
        parameters={
            # 1. Learning rate - log scale from 1e-5 to 1e-2
            #
            #
            "lp_params.progress_smooth": ParameterConfig(
                min=0.0,
                max=1.0,
                distribution="uniform",
                mean=0.5,
                scale="auto",
            ),
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
            # 6. Adam epsilon - log scale from 1e-8 to 1e-4
            "trainer.optimizer.eps": ParameterConfig(
                min=1e-8,
                max=1e-4,
                distribution="log_normal",
                mean=1e-6,  # Geometric mean
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=20,  # Start with 20 random samples for better exploration in large sweeps
            max_suggestion_cost=7200 * 1.5,  # 3 hours max per trial (for quick testing)
        ),
    )

    # Import DispatcherType for local testing
    from metta.tools.sweep import DispatcherType

    # Configure based on local_test flag
    if local_test:
        # Local testing configuration
        dispatcher_type = DispatcherType.LOCAL
        total_timesteps = 50000  # Quick 50k timesteps for testing
        monitoring_interval = 30  # Check more frequently for local testing
    else:
        # Production configuration
        dispatcher_type = DispatcherType.SKYPILOT
        total_timesteps = 2000000000  # 2B timesteps for production
        monitoring_interval = 60

    # Create and return the sweep tool using adaptive infrastructure
    return SweepTool(
        protein_config=protein_config,
        max_trials=max_trials,
        batch_size=batch_size,
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        monitoring_interval=monitoring_interval,
        max_parallel_jobs=max_parallel_jobs,
        gpus=gpus,
        dispatcher_type=dispatcher_type,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )

LEARNING_PROGRESS_PARAMETER_SPACE = ProteinConfig(
    metric="evaluator/eval_arena/score",  # Metric to optimize
    goal="maximize",
    method="bayes",  # Use Bayesian optimization
    parameters={
        # 1. Learning rate - log scale from 1e-5 to 1e-2
        #
        #
        "lp_params.progress_smooth": ParameterConfig(
            min=0.0,
            max=1.0,
            distribution="uniform",
            mean=0.5,
            scale="auto",
        ),

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
        # 6. Adam epsilon - log scale from 1e-8 to 1e-4
        "trainer.optimizer.eps": ParameterConfig(
            min=1e-8,
            max=1e-4,
            distribution="log_normal",
            mean=1e-6,  # Geometric mean
            scale="auto",
        ),
    },
    settings=ProteinSettings(
        num_random_samples=20,  # Start with 20 random samples for better exploration in large sweeps
        max_suggestion_cost=7200 * 4.5,  # 3 hours max per trial (for quick testing)
    ),
)


def learning_progress(
    recipe: str = "experiments.recipes.icl_resource_chain",
    train: str = "train",
    eval: str = "evaluate",
    max_trials: int = 300,
    max_parallel_jobs: int = 6,
    gpus: int = 1,
    batch_size: int = 4,
    local_test: bool = False,
) -> SweepTool:
    # Define the 6 PPO parameters to sweep over
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",  # Metric to optimize
        goal="maximize",
        method="bayes",  # Use Bayesian optimization
        parameters={
            # 0. Learning rate - log scale from 1e-5 to 1e-2
            "lp_params.progress_smooth": ParameterConfig(
                min=0.0,
                max=1.0,
                distribution="uniform",
                mean=0.5,
                scale="auto",
            ),
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
            # 6. Adam epsilon - log scale from 1e-8 to 1e-4
            "trainer.optimizer.eps": ParameterConfig(
                min=1e-8,
                max=1e-4,
                distribution="log_normal",
                mean=1e-6,  # Geometric mean
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=20,  # Start with 20 random samples for better exploration in large sweeps
            max_suggestion_cost=7200 * 1.5,  # 3 hours max per trial (for quick testing)
        ),
    )

    # Import DispatcherType for local testing

    # Configure based on local_test flag
    if local_test:
        # Local testing configuration
        dispatcher_type = DispatcherType.LOCAL
        total_timesteps = 50000  # Quick 50k timesteps for testing
        monitoring_interval = 30  # Check more frequently for local testing
    else:
        # Production configuration
        dispatcher_type = DispatcherType.SKYPILOT
        total_timesteps = 2000000000  # 2B timesteps for production
        monitoring_interval = 60

    # Create and return the sweep tool using adaptive infrastructure
    return SweepTool(
        protein_config=protein_config,
        max_trials=max_trials,
        batch_size=batch_size,
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        monitoring_interval=monitoring_interval,
        max_parallel_jobs=max_parallel_jobs,
        gpus=gpus,
        dispatcher_type=dispatcher_type,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )
