"""LunarLander-v3 recipe for HPO experiments using StableBaselines3."""

from typing import Optional

from ray import tune

from metta.sweep.ray.ray_controller import SweepConfig
from metta.tools.ray_sweep import RaySweepTool
from metta.hpo_lab.tools import EvaluateSB3Tool, TrainSB3GymEnvTool



def evaluate(model_path: Optional[str] = None) -> "EvaluateSB3Tool":
    """Create an EvaluateSB3Tool for LunarLander evaluation.

    Example usage:
        # Evaluate with a trained model
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.evaluate \\
            model_path=./train_dir/my_experiment/model/PPO_LunarLander-v3.zip

        # Evaluate with random policy (for testing)
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.evaluate
    """
    from metta.hpo_lab.tools import EvaluateSB3Tool

    return EvaluateSB3Tool(
        env_id="LunarLander-v3",
        model_path=model_path,
        n_eval_episodes=100,
        success_threshold=200,  # LunarLander success threshold
    )


def ray_sweep(sweep_name: str):
    """Ray sweep for LunarLander HPO experiments.

    This sweep explores the hyperparameter space of PPO on LunarLander-v3.
    It uses Optuna for Bayesian optimization to efficiently find good configurations.

    Usage:
        # Local test (10 trials)
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.ray_sweep \\
            sweep_name="test_local" \\
            -- --num_samples=10

        # Cloud sweep (100 trials)
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.ray_sweep \\
            sweep_name="ak.lunarlander.v1" \\
            -- gpus=8 nodes=2
    """

    # Create sweep configuration
    config = SweepConfig(
        sweep_id=sweep_name,
        recipe_module="metta.hpo_lab.recipes.lunarlander",
        train_entrypoint="train",
        num_samples=100,  # Number of trials
        max_concurrent_trials=2,
        gpus_per_trial="auto",  # Let Ray auto-allocate GPU resources
        cpus_per_trial="auto",  # Let Ray auto-allocate CPU resources
        max_failures_per_trial=1,
        fail_fast=False,
    )

    # Define hyperparameter search space using Ray Tune
    search_space = {
        # Most important: learning rate (log scale)
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        # PPO hyperparameters
        "n_steps": tune.choice([512, 1024, 2048, 4096]),
        "batch_size": tune.choice([64, 128, 256, 512, 1024]),  # Expanded range for GPU
        "n_epochs": tune.randint(3, 20),
        "gamma": tune.uniform(0.95, 0.999),
        "gae_lambda": tune.uniform(0.9, 0.99),
        "clip_range": tune.uniform(0.1, 0.3),
        "ent_coef": tune.loguniform(1e-6, 1e-2),
        "vf_coef": tune.uniform(0.1, 1.0),
        "max_grad_norm": tune.uniform(0.3, 2.0),

        # Network architecture - use string keys to avoid list hashing issues
        #"net_arch_type": tune.choice(["small", "medium", "large", "deep", "pyramid"]),
        # Number of parallel environments - more for GPU parallelism
        "n_envs": tune.choice([8, 16, 32, 64]),  # Increased for GPU
        # Fixed training duration for fair comparison
        "total_timesteps": 1_000_000,
    }

    return RaySweepTool(
        sweep_config=config,
        search_space=search_space,
        # Optuna will be used by default for optimization
    )


def mini_sweep(sweep_name: str):
    """Minimal sweep for quick testing (10 trials, smaller search space).

    Usage:
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.mini_sweep \\
            sweep_name="test_mini"
    """

    # For mini sweep, use simpler approach without complex nested structures
    config = SweepConfig(
        sweep_id=sweep_name,
        recipe_module="metta.hpo_lab.recipes.lunarlander",
        train_entrypoint="train",
        num_samples=10,  # Only 10 trials
        max_concurrent_trials=4,
        gpus_per_trial=0,  # CPU only for local testing
        cpus_per_trial="auto",  # Let Ray auto-allocate CPU resources
        fail_fast=True,  # Stop on first failure
        max_failures_per_trial=0,
        stats_server_uri=None
    )

    # Use a simple search space for Ray Tune
    from ray import tune

    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-3),
        "n_steps": tune.choice([1024, 2048]),
        "n_epochs": tune.choice([5, 10, 15]),
        "batch_size": tune.choice([32, 64]),
        "total_timesteps": 200000,  # Shorter training for testing
    }

    return RaySweepTool(
        sweep_config=config,
        search_space=search_space,
    )


def train() -> "TrainSB3GymEnvTool":
    """Create a TrainSB3GymEnvTool for LunarLander training.

    This recipe function returns a Tool that can be invoked via the standard
    Metta CLI interface.

    Example usage:
        # Train with default settings
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train run=my_experiment

        # Train with custom hyperparameters
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train \\
            run=my_experiment \\
            learning_rate=0.0001 \\
            total_timesteps=500000

        # Train with a different environment
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train \\
            run=cartpole_test \\
            env_id="CartPole-v1" \\
            total_timesteps=100000
    """
    from metta.hpo_lab.tools import TrainSB3GymEnvTool

    return TrainSB3GymEnvTool(
        # Default to LunarLander-v3
        env_id="LunarLander-v3",
        # Default hyperparameters
        total_timesteps=1_000_000,
        algorithm="PPO",
        n_envs=8,
    )
