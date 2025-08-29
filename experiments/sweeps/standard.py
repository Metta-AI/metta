"""Standard sweep configurations using the new orchestrator pattern."""

from typing import Optional

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep_orchestrator import SweepOrchestratorTool


def ppo(
    sweep_name: Optional[str] = None,
    recipe: str = "experiments.recipes.arena",
    train: str = "train_shaped",
    eval: str = "evaluate",
    max_trials: int = 10,
    max_parallel_jobs: int = 1,
) -> SweepOrchestratorTool:
    """Create a PPO hyperparameter sweep using the new orchestrator.
    
    This sweep optimizes 5 key PPO parameters:
    - learning_rate: The optimizer learning rate
    - clip_coef: PPO clipping coefficient for policy updates
    - ent_coef: Entropy coefficient for exploration
    - gae_lambda: GAE lambda for advantage estimation
    - vf_coef: Value function coefficient in the loss
    
    Args:
        sweep_name: Name for this sweep (will be auto-generated if not provided)
        recipe: Module path to the recipe (e.g., "experiments.recipes.arena")
        train: Training entrypoint in the recipe module
        eval: Evaluation entrypoint in the recipe module
        max_trials: Number of hyperparameter configurations to try
        max_parallel_jobs: Maximum number of parallel jobs to run
    
    Returns:
        Configured SweepOrchestratorTool ready to run
    
    Example:
        # Run a sweep on arena recipe
        uv run ./tools/run.py experiments.sweeps.standard.ppo \
            --args sweep_name=my_ppo_sweep recipe=experiments.recipes.arena \
                   train=train_shaped eval=evaluate max_trials=20
        
        # Run a sweep on navigation recipe
        uv run ./tools/run.py experiments.sweeps.standard.ppo \
            --args sweep_name=nav_sweep recipe=experiments.recipes.navigation \
                   train=train eval=evaluate max_trials=10
    """
    
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
            "trainer.ppo.clip_coef": ParameterConfig(
                min=0.05,
                max=0.3,
                distribution="uniform",
                mean=0.175,
                scale="auto",
            ),
            # 3. Entropy coefficient - log scale from 0.0001 to 0.01
            "trainer.ppo.ent_coef": ParameterConfig(
                min=0.0001,
                max=0.01,
                distribution="log_normal",
                mean=0.001,  # Geometric mean
                scale="auto",
            ),
            # 4. GAE lambda - uniform from 0.8 to 0.99
            "trainer.ppo.gae_lambda": ParameterConfig(
                min=0.8,
                max=0.99,
                distribution="uniform",
                mean=0.895,
                scale="auto",
            ),
            # 5. Value function coefficient - uniform from 0.1 to 1.0
            "trainer.ppo.vf_coef": ParameterConfig(
                min=0.1,
                max=1.0,
                distribution="uniform",
                mean=0.55,
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=5,  # Start with 5 random samples before Bayesian optimization
            max_suggestion_cost=300,  # 5 minutes max per trial (for quick testing)
        ),
    )
    
    # Create and return the orchestrator tool
    return SweepOrchestratorTool(
        sweep_name=sweep_name,
        protein_config=protein_config,
        max_trials=max_trials,
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        max_parallel_jobs=max_parallel_jobs,
        monitoring_interval=5,
    )


def quick_test(
    sweep_name: Optional[str] = None,
    recipe: str = "experiments.recipes.arena",
) -> SweepOrchestratorTool:
    """Quick test sweep with full PPO config but minimal trials for testing.
    
    This sweep uses the same full PPO parameter configuration as the main ppo()
    function but runs fewer trials with shorter training for quick testing
    that the orchestrator infrastructure is working.
    
    Args:
        sweep_name: Name for this sweep (will be auto-generated if not provided)
        recipe: Module path to the recipe
    
    Returns:
        Configured SweepOrchestratorTool for quick testing
    
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
            "trainer.ppo.clip_coef": ParameterConfig(
                min=0.05,
                max=0.3,
                distribution="uniform",
                mean=0.175,
                scale="auto",
            ),
            # 3. Entropy coefficient - log scale from 0.0001 to 0.01
            "trainer.ppo.ent_coef": ParameterConfig(
                min=0.0001,
                max=0.01,
                distribution="log_normal",
                mean=0.001,  # Geometric mean
                scale="auto",
            ),
            # 4. GAE lambda - uniform from 0.8 to 0.99
            "trainer.ppo.gae_lambda": ParameterConfig(
                min=0.8,
                max=0.99,
                distribution="uniform",
                mean=0.895,
                scale="auto",
            ),
            # 5. Value function coefficient - uniform from 0.1 to 1.0
            "trainer.ppo.vf_coef": ParameterConfig(
                min=0.1,
                max=1.0,
                distribution="uniform",
                mean=0.55,
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=2,  # Start with 2 random samples before Bayesian optimization
            max_suggestion_cost=60,  # 1 minute max per trial for quick testing
        ),
    )
    
    return SweepOrchestratorTool(
        sweep_name=sweep_name,
        protein_config=protein_config,
        max_trials=5,  # Only 5 trials for quick testing
        recipe_module=recipe,
        train_entrypoint="train_shaped",
        eval_entrypoint="evaluate",
        max_parallel_jobs=1,
        monitoring_interval=5,
    )