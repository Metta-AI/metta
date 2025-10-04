from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings


def make_custom_protein_config(
    base_config: ProteinConfig, parameters: dict[str, ParameterConfig]
) -> ProteinConfig:
    """Create a custom sweep configuration by extending a base config with additional parameters.

    This allows experimentalists to easily define new parameter spaces by starting with
    a base configuration (like PPO_BASIC or PPO_FULL) and adding or overriding specific
    parameters.

    Args:
        base_config: The base ProteinConfig to extend (e.g., PPO_BASIC or PPO_FULL)
        parameters: Dictionary mapping parameter paths to ParameterConfig objects.
                   These will be merged with the base config, overriding existing
                   parameters or adding new ones.

    Returns:
        A new ProteinConfig with the merged parameter space.

    Example:
        # Start with basic PPO config and add a new parameter
        custom = custom_config(
            PPO_BASIC,
            {
                "trainer.grad_clip": ParameterConfig(
                    min=0.1,
                    max=10.0,
                    distribution="log_normal",
                    mean=1.0,
                    scale="auto",
                ),
                # Override an existing parameter with different bounds
                "trainer.optimizer.learning_rate": ParameterConfig(
                    min=1e-6,
                    max=1e-3,
                    distribution="log_normal",
                    mean=1e-4,
                    scale="auto",
                ),
            }
        )
    """
    # Create a copy of the base config's parameters
    merged_parameters = base_config.parameters.copy()

    # Add or override parameters from the provided dictionary
    for param_name, param_config in parameters.items():
        merged_parameters[param_name] = param_config

    # Create a new ProteinConfig with the merged parameters
    return ProteinConfig(
        metric=base_config.metric,
        goal=base_config.goal,
        method=base_config.method,
        parameters=merged_parameters,
        settings=base_config.settings,
    )


PPO_CORE = ProteinConfig(
    metric="evaluator/eval_arena/score",  # Metric to optimize
    goal="maximize",
    method="bayes",  # Use Bayesian optimization
    parameters={
        # 1. Learning rate - log scale from 1e-5 to 1e-2
        "trainer.optimizer.learning_rate": ParameterConfig(
            min=1e-5,
            max=1e-2,
            distribution="log_normal",
            mean=0.001153637,  # Geometric mean
            scale="auto",
        ),
        # 3. Entropy coefficient - log scale from 0.0001 to 0.01
        "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
            min=0.0001,
            max=0.03,
            distribution="log_normal",
            mean=0.01,  # Geometric mean
            scale="auto",
        ),
    },
    settings=ProteinSettings(
        num_random_samples=0,
        max_suggestion_cost=3600 * 6,
        resample_frequency=10,
        random_suggestions=15,
        suggestions_per_pareto=128,
    ),
)

# 8 Parameters
PPO_BASIC = ProteinConfig(
    metric="evaluator/eval_arena/score",  # Metric to optimize
    goal="maximize",
    method="bayes",  # Use Bayesian optimization
    parameters={
        "trainer.batch_size": ParameterConfig(
            distribution="uniform_pow2",
            min=524288,  # 2^19 - Rollout batch PoW
            max=4194304,  # 2^22 - Rollout batch PoW
            mean=1048576,  # 2^20 - Rollout batch PoW
            scale="auto",
        ),
        "trainer.minibatch_size": ParameterConfig(
            distribution="uniform_pow2",
            min=2048,  # 2^11 - Minibatch PoW
            max=32768,  # 2^15 - Minibatch PoW
            mean=8192,  # 2^13 - Minibatch PoW
            scale="auto",
        ),
        # 1. Learning rate - log scale from 1e-5 to 1e-2
        "trainer.optimizer.learning_rate": ParameterConfig(
            min=1e-5,
            max=1e-2,
            distribution="log_normal",
            mean=0.001153637,  # Geometric mean
            scale="auto",
        ),
        # 2. PPO clip coefficient - uniform from 0.05 to 0.3
        "trainer.losses.loss_configs.ppo.clip_coef": ParameterConfig(
            min=0.05,
            max=0.3,
            distribution="uniform",
            mean=0.264407,
            scale="auto",
        ),
        # 3. Entropy coefficient - log scale from 0.0001 to 0.01
        "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
            min=0.0001,
            max=0.01,
            distribution="log_normal",
            mean=0.010000,  # Geometric mean
            scale="auto",
        ),
        # 4. GAE lambda - uniform from 0.8 to 0.99
        "trainer.losses.loss_configs.ppo.gae_lambda": ParameterConfig(
            min=0.8,
            max=0.99,
            distribution="uniform",
            mean=0.891477,
            scale="auto",
        ),
        # 5. Value function coefficient - uniform from 0.1 to 1.0
        "trainer.losses.loss_configs.ppo.vf_coef": ParameterConfig(
            min=0.1,
            max=1.0,
            distribution="uniform",
            mean=0.897619,
            scale="auto",
        ),
        # 6. Adam epsilon - log scale from 1e-8 to 1e-4
        "trainer.optimizer.eps": ParameterConfig(
            min=1e-8,
            max=1e-4,
            distribution="log_normal",
            mean=3.186531e-07,  # Geometric mean
            scale="auto",
        ),
    },
    settings=ProteinSettings(
        num_random_samples=20,
        max_suggestion_cost=3600 * 6,
        resample_frequency=3,
        global_search_scale=1.0,
        random_suggestions=15,
        suggestions_per_pareto=32,
        # expansion_rate=0.15,  # Not available in current ProteinSettings
        # seed_with_search_center=True,  # Not available in current ProteinSettings
    ),
)
