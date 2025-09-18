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


def full_hyperparameter_sweep(
    recipe: str = "experiments.recipes.arena",
    train: str = "train",
    eval: str = "evaluate",
    max_trials: int = 300,
    max_parallel_jobs: int = 6,
    gpus: int = 1,
    batch_size: int = 4,
    local_test: bool = False,
) -> SweepTool:
    """Comprehensive hyperparameter sweep mirroring the full YAML config.

    This sweep explores a wider parameter space including:
    - Training duration (total_timesteps)
    - Batch and minibatch sizes
    - BPTT horizon
    - Full PPO parameters (gamma, gae_lambda, clip_coef, vf_coef, vf_clip_coef, ent_coef)
    - Update epochs
    - Optimizer parameters (learning_rate, betas, eps)

    Args:
        recipe: Recipe module to use for training and evaluation
        train: Training entrypoint name
        eval: Evaluation entrypoint name
        max_trials: Maximum number of trials to run
        max_parallel_jobs: Maximum parallel jobs
        gpus: Number of GPUs per job
        batch_size: Number of suggestions per batch
        local_test: If True, use local dispatcher with reduced timesteps

    Returns:
        Configured SweepTool for comprehensive hyperparameter optimization
    """

    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",
        goal="maximize",
        method="bayes",
        parameters={
            # Training duration
            "trainer.total_timesteps": ParameterConfig(
                distribution="int_uniform",
                min=800000000,  # 800M - just before first plateau jump at 900M
                max=6000000000,  # 6B - extended range for finding optimal convergence
                mean=1500000000,  # 1.5B - balanced starting point
                scale="auto",
            ),

            # Batch configuration
            "trainer.batch_size": ParameterConfig(
                distribution="uniform_pow2",
                min=262144,
                max=2097152,
                mean=524288,
                scale="auto",
            ),

            "trainer.minibatch_size": ParameterConfig(
                distribution="uniform_pow2",
                min=2048,
                max=32768,
                mean=8192,
                scale="auto",
            ),

            "trainer.bptt_horizon": ParameterConfig(
                distribution="uniform_pow2",
                min=8,
                max=32,
                mean=16,
                scale="auto",
            ),

            # PPO parameters
            "trainer.losses.loss_configs.ppo.gamma": ParameterConfig(
                distribution="logit_normal",
                min=0.95,
                max=0.999,
                mean=0.99,
                scale="auto",
            ),

            "trainer.losses.loss_configs.ppo.gae_lambda": ParameterConfig(
                distribution="logit_normal",
                min=0.9,
                max=0.99,
                mean=0.95,
                scale="auto",
            ),

            "trainer.losses.loss_configs.ppo.clip_coef": ParameterConfig(
                distribution="logit_normal",
                min=0.1,
                max=0.3,
                mean=0.2,
                scale="auto",
            ),

            "trainer.losses.loss_configs.ppo.vf_coef": ParameterConfig(
                distribution="logit_normal",
                min=0.3,
                max=0.8,
                mean=0.5,
                scale="auto",
            ),

            "trainer.losses.loss_configs.ppo.vf_clip_coef": ParameterConfig(
                distribution="log_normal",
                min=5.0,
                max=20.0,
                mean=10.0,
                scale="auto",
            ),

            "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
                distribution="log_normal",
                min=5e-4,
                max=5e-3,
                mean=1e-3,
                scale="auto",
            ),

            # Update configuration
            "trainer.update_epochs": ParameterConfig(
                distribution="int_uniform",
                min=1,
                max=6,
                mean=3,
                scale="auto",
            ),

            # Optimizer parameters
            "trainer.optimizer.learning_rate": ParameterConfig(
                distribution="log_normal",
                min=1e-4,
                max=1e-2,
                mean=3e-4,
                scale="auto",
            ),

            "trainer.optimizer.beta1": ParameterConfig(
                distribution="logit_normal",
                min=0.8,
                max=0.95,
                mean=0.9,
                scale="auto",
            ),

            "trainer.optimizer.beta2": ParameterConfig(
                distribution="logit_normal",
                min=0.99,
                max=0.999,
                mean=0.999,
                scale="auto",
            ),

            "trainer.optimizer.eps": ParameterConfig(
                distribution="log_normal",
                min=1e-9,
                max=1e-7,
                mean=1e-8,
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=20,
            max_suggestion_cost=3600,
            resample_frequency=3,
            global_search_scale=1.0,
            random_suggestions=15,
            suggestions_per_pareto=32,
            # expansion_rate=0.15,  # Not available in current ProteinSettings
            # seed_with_search_center=True,  # Not available in current ProteinSettings
            # acquisition_fn="naive",  # Not available in current ProteinSettings
            # ucb_beta=2.0,  # Not available in current ProteinSettings
        ),
    )

    # Configure based on local_test flag
    if local_test:
        dispatcher_type = DispatcherType.LOCAL
        monitoring_interval = 30
        # For local testing, override the total_timesteps to something smaller
        # (will be overridden by the sweep parameter anyway)
        train_overrides = {}
    else:
        dispatcher_type = DispatcherType.SKYPILOT
        monitoring_interval = 60
        train_overrides = {}

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
        train_overrides=train_overrides,
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
