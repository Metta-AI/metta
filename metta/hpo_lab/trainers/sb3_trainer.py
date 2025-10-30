"""StableBaselines3 trainer wrapper for HPO experiments."""

import os
import time
from typing import Any, Optional

import numpy as np
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Suppress PyTorch distributed warnings on macOS/Windows
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"


class MetricsCallback(BaseCallback):
    """Callback to collect metrics during training for sweep integration."""

    def __init__(self, verbose: int = 0, log_to_wandb: bool = False, wandb_run: Optional[Any] = None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = None
        self.current_lengths = None
        self.log_to_wandb = log_to_wandb
        self.wandb_run = wandb_run
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Initialize episode tracking arrays."""
        self.current_rewards = np.zeros(self.training_env.num_envs)
        self.current_lengths = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        """Track episode rewards and lengths."""
        # Update current episode stats
        self.current_rewards += self.locals["rewards"]
        self.current_lengths += 1

        # Check for episode ends
        for i, done in enumerate(self.locals["dones"]):
            if done:
                episode_reward = self.current_rewards[i]
                episode_length = self.current_lengths[i]

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Log to WandB if enabled
                if self.log_to_wandb and self.wandb_run is not None:
                    import wandb

                    # Use the global timestep counter from the model
                    global_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.num_timesteps

                    wandb.log(
                        {
                            "episode/reward": episode_reward,
                            "episode/length": episode_length,
                            "episode/count": self.episode_count,
                            "global_step": global_step,  # Include global_step in the log
                        }
                    )

                self.current_rewards[i] = 0
                self.current_lengths[i] = 0

        return True

    def _on_rollout_end(self) -> None:
        """Log training metrics at the end of each rollout."""
        if self.log_to_wandb and self.wandb_run is not None:
            import wandb

            # Use the global timestep counter from the model
            global_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.num_timesteps

            # Collect all metrics for a single log call
            metrics = {"global_step": global_step}

            # Get the latest training info if available
            if hasattr(self, "locals") and "infos" in self.locals:
                infos = self.locals.get("infos", [])
                if infos and len(infos) > 0 and isinstance(infos[0], dict):
                    # Extract training metrics from the info dict
                    info = infos[0]
                    if "episode" in info:
                        ep_info = info["episode"]
                        metrics["rollout/ep_rew_mean"] = ep_info.get("r", 0)
                        metrics["rollout/ep_len_mean"] = ep_info.get("l", 0)

            # Log aggregated metrics if we have episodes
            if self.episode_rewards:
                metrics["train/mean_reward"] = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
                metrics["train/std_reward"] = np.std(self.episode_rewards[-100:])
                metrics["train/mean_length"] = np.mean(self.episode_lengths[-100:])

            # Single log call with all metrics
            wandb.log(metrics)

    def get_metrics(self) -> dict:
        """Get aggregated metrics."""
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "num_episodes": 0,
            }

        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "num_episodes": len(self.episode_rewards),
        }


class SB3Trainer:
    """Wrapper for StableBaselines3 algorithms compatible with Metta sweeps."""

    ALGORITHMS = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
    }

    def __init__(
        self,
        env_id: str = "LunarLander-v3",
        algorithm: str = "PPO",
        total_timesteps: int = 1_000_000,
        n_envs: int = 8,
        device: str = "auto",
        seed: Optional[int] = None,
        verbose: int = 0,
        # WandB configuration
        use_wandb: bool = False,
        wandb_run: Optional[Any] = None,  # Pass existing wandb run
        **algo_kwargs,
    ):
        """Initialize SB3 trainer.

        Args:
            env_id: Gymnasium environment ID
            algorithm: Algorithm name (PPO, A2C, SAC)
            total_timesteps: Total training timesteps
            n_envs: Number of parallel environments
            device: Device for training (auto, cuda, cpu)
            seed: Random seed
            verbose: Verbosity level
            use_wandb: Whether to log to WandB
            wandb_run: Existing WandB run object (from WandbContext)
            **algo_kwargs: Algorithm-specific hyperparameters
        """
        self.env_id = env_id
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.algo_kwargs = algo_kwargs
        self.use_wandb = use_wandb
        self.wandb_run = wandb_run

        # Setup environment
        self.env = self._make_env()

        # Setup algorithm
        self.model = self._make_model()

        # Setup callbacks
        self.metrics_callback = MetricsCallback(log_to_wandb=self.use_wandb, wandb_run=self.wandb_run)

    def _make_env(self):
        """Create vectorized environment."""
        import os

        # Suppress PyTorch distributed warnings in subprocesses
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

        # Use SubprocVecEnv for better performance with multiple envs
        vec_env_cls = SubprocVecEnv if self.n_envs > 1 else DummyVecEnv

        return make_vec_env(
            self.env_id,
            n_envs=self.n_envs,
            seed=self.seed,
            vec_env_cls=vec_env_cls,
        )

    def _make_model(self):
        """Create the RL algorithm model."""
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Available: {list(self.ALGORITHMS.keys())}")

        algo_class = self.ALGORITHMS[self.algorithm]

        # Filter kwargs based on algorithm
        init_kwargs = self._filter_algo_kwargs(algo_class)

        return algo_class(
            "MlpPolicy",
            self.env,
            verbose=self.verbose,
            device=self.device,
            seed=self.seed,
            **init_kwargs,
        )

    def _filter_algo_kwargs(self, algo_class) -> dict:
        """Filter kwargs to only include valid parameters for the algorithm."""
        import inspect

        sig = inspect.signature(algo_class.__init__)
        valid_params = set(sig.parameters.keys())

        filtered = {}
        for key, value in self.algo_kwargs.items():
            if key in valid_params:
                filtered[key] = value
            elif self.verbose > 0:
                print(f"Warning: {key} is not a valid parameter for {self.algorithm}")

        return filtered

    def train(self) -> dict:
        """Train the model and return metrics for sweep integration."""
        start_time = time.time()

        # Log training start and configure x-axis
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            # Define the x-axis for all charts
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

            wandb.log(
                {
                    "training/started": True,
                    "global_step": 0,
                }
            )

        # Train the model
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=self.metrics_callback,
            progress_bar=self.verbose > 0,
        )

        # Evaluate final performance
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=10,
            deterministic=True,
        )

        # Get training metrics
        training_metrics = self.metrics_callback.get_metrics()

        # Compile all metrics
        metrics = {
            "final_mean_reward": float(mean_reward),
            "final_std_reward": float(std_reward),
            "training_time": time.time() - start_time,
            "total_timesteps": self.total_timesteps,
            "success_rate": float(mean_reward > 200),  # LunarLander success threshold
            **training_metrics,
        }

        # Log final metrics to WandB
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            # Log summary metrics at the final timestep
            wandb.log(
                {
                    "final/mean_reward": metrics["final_mean_reward"],
                    "final/std_reward": metrics["final_std_reward"],
                    "final/success_rate": metrics["success_rate"],
                    "final/training_time": metrics["training_time"],
                    "global_step": self.total_timesteps,
                }
            )

            # Update summary for sweep infrastructure
            if hasattr(self.wandb_run, "summary"):
                self.wandb_run.summary.update(
                    {
                        "final_mean_reward": metrics["final_mean_reward"],
                        "success_rate": metrics["success_rate"],
                        "training_completed": True,
                    }
                )

            # Don't close WandB here - let the WandbContext handle it
            if self.verbose > 0 and hasattr(self.wandb_run, "url"):
                print(f"WandB run: {self.wandb_run.url}")

        # Log metrics if verbose
        if self.verbose > 0:
            print(f"\nTraining completed in {metrics['training_time']:.2f} seconds")
            print(f"Final mean reward: {metrics['final_mean_reward']:.2f} ± {metrics['final_std_reward']:.2f}")
            print(f"Training episodes: {metrics['num_episodes']}")
            print(f"Mean episode length: {metrics['mean_length']:.2f}")

        return metrics

    def save(self, path: str):
        """Save the trained model."""
        self.model.save(path)

    def load(self, path: str):
        """Load a trained model."""
        algo_class = self.ALGORITHMS[self.algorithm]
        self.model = algo_class.load(path, env=self.env, device=self.device)
