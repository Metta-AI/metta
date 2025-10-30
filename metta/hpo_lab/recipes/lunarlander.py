"""LunarLander-v3 recipe for HPO experiments using StableBaselines3."""

import contextlib
from typing import Optional

import torch
from ray import tune

from metta.common.wandb.context import WandbConfig, WandbContext
from metta.sweep.ray.ray_controller import SweepConfig
from metta.tools.ray_sweep import RaySweepTool
from metta.tools.utils.auto_config import auto_wandb_config


def train(
    # Run configuration
    run: Optional[str] = None,
    # Environment
    env_id: str = "LunarLander-v3",
    n_envs: int = 8,
    # Algorithm selection
    algorithm: str = "PPO",
    # Common hyperparameters
    learning_rate: float = 3e-4,
    total_timesteps: int = 1_000_000,
    # PPO specific (ignored for other algos)
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    # Network architecture
    net_arch: Optional[list] = None,
    net_arch_type: Optional[str] = None,  # For sweep compatibility
    # Training settings
    seed: Optional[int] = None,
    verbose: int = 0,
    # WandB configuration
    wandb: Optional[WandbConfig] = None,
    group: Optional[str] = None,
):
    """Train LunarLander using StableBaselines3.

    This function is designed to be called by the sweep infrastructure.
    It returns metrics that can be used for hyperparameter optimization.
    """
    from metta.hpo_lab.trainers.sb3_trainer import SB3Trainer

    # Auto-configure WandB if not provided
    if wandb is None:
        # Use the auto-config pattern from main Metta
        wandb = auto_wandb_config(run=run, group=group, tags=["hpo-lab", "lunarlander"])
    elif wandb == WandbConfig.Unconfigured():
        # Replace unconfigured with auto-config
        wandb = auto_wandb_config(run=run, group=group, tags=["hpo-lab", "lunarlander"])

    # Override group if specified separately
    if group is not None:
        wandb.group = group

    # Handle network architecture selection
    if net_arch_type is not None:
        # Map string types to actual architectures
        arch_map = {
            "small": [64, 64],
            "medium": [128, 128],
            "large": [256, 256],
            "deep": [128, 256, 128],
            "pyramid": [256, 128, 64],
        }
        net_arch = arch_map.get(net_arch_type, [64, 64])
    elif net_arch is None:
        net_arch = [64, 64]

    # Create policy kwargs for network architecture
    policy_kwargs = {"net_arch": net_arch}

    # Build training config for WandB logging
    training_config = {
        "env_id": env_id,
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "net_arch": net_arch,
        "seed": seed,
    }

    # Use WandbContext for proper WandB management
    wandb_manager = WandbContext(wandb, run_config=training_config) if wandb.enabled else contextlib.nullcontext(None)

    with wandb_manager as wandb_run:
        # Initialize trainer with WandB integration
        trainer = SB3Trainer(
            env_id=env_id,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            seed=seed,
            verbose=verbose,
            # WandB is already initialized by context
            use_wandb=wandb.enabled,
            wandb_run=wandb_run,  # Pass the run directly
            # PPO hyperparameters
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
        )

        # Train and return metrics
        metrics = trainer.train()

    # The sweep infrastructure expects a dictionary of metrics
    return metrics


def evaluate(
    env_id: str = "LunarLander-v3",
    model_path: Optional[str] = None,
    n_eval_episodes: int = 100,
):
    """Evaluate a trained model on LunarLander."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import make_vec_env

    # Create environment
    env = make_vec_env(env_id, n_envs=1)

    if model_path:
        # Load trained model
        model = PPO.load(model_path, env=env)
    else:
        # Use a random policy for testing
        model = PPO("MlpPolicy", env, verbose=0)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_episodes": n_eval_episodes,
        "success_rate": float(mean_reward > 200),  # LunarLander success threshold
    }


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
        max_concurrent_trials=8,
        gpus_per_trial=1 if torch.cuda.is_available() else 0,
        cpus_per_trial=4,
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
        "net_arch_type": tune.choice(["small", "medium", "large", "deep", "pyramid"]),
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
        gpus_per_trial=0,  # CPU only for testing
        cpus_per_trial=2,
        fail_fast=True,  # Stop on first failure
        max_failures_per_trial=0,
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


def train_sb3_tool() -> "TrainSB3GymEnvTool":
    """Create a TrainSB3GymEnvTool for LunarLander training.

    This recipe function returns a Tool that can be invoked via the standard
    Metta CLI interface.

    Example usage:
        # Train with default settings
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train_sb3_tool run=my_experiment

        # Train with custom hyperparameters
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train_sb3_tool \\
            run=my_experiment \\
            learning_rate=0.0001 \\
            total_timesteps=500000

        # Train with a different environment
        uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train_sb3_tool \\
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
