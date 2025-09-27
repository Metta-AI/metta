from softmax.training.sweep.protein_config import (
    ParameterConfig,
    ProteinConfig,
    ProteinSettings,
)


def custom_config(
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

# 14 Parameters
PPO_FULL = ProteinConfig(
    metric="evaluator/eval_arena/score",
    goal="maximize",
    method="bayes",
    parameters={
        # Batch configuration
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
        "trainer.bptt_horizon": ParameterConfig(
            distribution="uniform_pow2",
            min=8,  # 2^3 - BPTT horizon PoW
            max=32,  # 2^5 - BPTT horizon PoW
            mean=16,  # 2^4 - BPTT horizon PoW
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
        max_suggestion_cost=3600 * 6,  # 6 hours
        resample_frequency=3,
        global_search_scale=1.0,
        random_suggestions=15,
        suggestions_per_pareto=32,
        # expansion_rate=0.15,  # Not available in current ProteinSettings
        # seed_with_search_center=True,  # Not available in current ProteinSettings
    ),
)

LP_CONFIG = custom_config(
    base_config=PPO_BASIC,
    parameters={
        "lp_params.progress_smoothing": ParameterConfig(
            distribution="logit_normal",
            min=0.01,
            max=0.5,
            mean=0.1,
            scale="auto",
        ),
        "lp_params.exploration_bonus": ParameterConfig(
            distribution="logit_normal",
            min=0.01,
            max=0.5,
            mean=0.09,
            scale="auto",
        ),
        "lp_params.ema_timescale": ParameterConfig(
            distribution="uniform",
            min=0.0001,
            max=0.1,
            mean=0.001,
            scale="auto",
        ),
        "lp_params.num_active_tasks": ParameterConfig(
            distribution="log_normal",
            min=500,
            max=6000,
            mean=1000,
            scale="auto",
        ),
        "lp_params.rand_task_rate": ParameterConfig(
            distribution="uniform",
            min=0.05,
            max=0.5,
            mean=0.175,
            scale="auto",
        ),
    },
)
