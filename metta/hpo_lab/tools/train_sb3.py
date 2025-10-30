"""Training tool for Gymnasium environments using StableBaselines3."""

import contextlib
import logging
import os
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from metta.common.tool import Tool
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.tools.utils.auto_config import auto_run_name, auto_wandb_config

logger = getRankAwareLogger(__name__)


class TrainSB3GymEnvTool(Tool):
    """Tool for training Gymnasium environments with StableBaselines3.

    This tool provides a CLI interface for training standard RL benchmarks
    using battle-tested StableBaselines3 implementations. It follows the same
    patterns as the main TrainTool for consistency.

    Example usage:
        # From recipe
        def train_lunarlander() -> TrainSB3GymEnvTool:
            return TrainSB3GymEnvTool(
                env_id="LunarLander-v3",
                total_timesteps=1_000_000,
            )

        # CLI invocation
        uv run ./tools/run.py module.train_lunarlander run=my_experiment
    """

    # Run configuration
    run: Optional[str] = Field(
        default=None,
        description="Run name for organizing experiments"
    )

    # Environment configuration
    env_id: str = Field(
        default="LunarLander-v3",
        description="Gymnasium environment ID"
    )
    n_envs: int = Field(
        default=8,
        description="Number of parallel environments"
    )

    # Algorithm configuration
    algorithm: Literal["PPO", "A2C", "SAC"] = Field(
        default="PPO",
        description="RL algorithm to use"
    )

    # Training configuration
    total_timesteps: int = Field(
        default=1_000_000,
        description="Total environment steps to train for"
    )
    learning_rate: float = Field(
        default=3e-4,
        description="Learning rate for optimizer"
    )

    # PPO-specific hyperparameters
    n_steps: int = Field(
        default=2048,
        description="Number of steps per environment per update"
    )
    batch_size: int = Field(
        default=64,
        description="Minibatch size for gradient updates"
    )
    n_epochs: int = Field(
        default=10,
        description="Number of epochs per update"
    )
    gamma: float = Field(
        default=0.99,
        description="Discount factor"
    )
    gae_lambda: float = Field(
        default=0.95,
        description="GAE lambda for advantage estimation"
    )
    clip_range: float = Field(
        default=0.2,
        description="PPO clipping parameter"
    )
    ent_coef: float = Field(
        default=0.0,
        description="Entropy coefficient"
    )
    vf_coef: float = Field(
        default=0.5,
        description="Value function coefficient"
    )
    max_grad_norm: float = Field(
        default=0.5,
        description="Maximum gradient norm for clipping"
    )

    # Network architecture
    net_arch: list[int] = Field(
        default_factory=lambda: [64, 64],
        description="Hidden layer sizes for actor-critic network"
    )

    # Device configuration
    device: str = Field(
        default="auto",
        description="Device for training (auto, cuda, cpu)"
    )

    # Training settings
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    verbose: int = Field(
        default=1,
        description="Verbosity level (0=silent, 1=progress, 2=debug)"
    )

    # WandB configuration
    wandb: WandbConfig = Field(
        default_factory=WandbConfig.Unconfigured,
        description="WandB configuration"
    )
    group: Optional[str] = Field(
        default=None,
        description="WandB group for organizing runs"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for the run"
    )

    # Output configuration
    save_model: bool = Field(
        default=True,
        description="Whether to save the trained model"
    )
    model_save_path: Optional[str] = Field(
        default=None,
        description="Path to save the model (auto-generated if None)"
    )

    # Evaluation configuration
    eval_episodes: int = Field(
        default=10,
        description="Number of episodes for final evaluation"
    )
    eval_deterministic: bool = Field(
        default=True,
        description="Use deterministic policy for evaluation"
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "TrainSB3GymEnvTool":
        """Validate and auto-configure fields."""
        # Ensure batch_size divides n_steps * n_envs
        rollout_buffer_size = self.n_steps * self.n_envs
        if rollout_buffer_size % self.batch_size != 0:
            # Adjust batch_size to be a divisor
            for divisor in range(self.batch_size, 0, -1):
                if rollout_buffer_size % divisor == 0:
                    logger.warning(
                        f"Adjusting batch_size from {self.batch_size} to {divisor} "
                        f"to evenly divide rollout buffer size {rollout_buffer_size}"
                    )
                    self.batch_size = divisor
                    break

        return self

    def invoke(self, args: dict[str, Any]) -> int | None:
        """Execute the training tool.

        Args:
            args: Command-line arguments (e.g., {"run": "experiment_name"})

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Handle run name from args
        if "run" in args:
            if self.run is None:
                self.run = args["run"]
            else:
                logger.warning(f"Ignoring run argument '{args['run']}' - using configured run '{self.run}'")

        # Auto-generate run name if not provided
        if self.run is None:
            self.run = auto_run_name(prefix="sb3")

        # Auto-configure WandB if unconfigured
        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(run=self.run, group=self.group, tags=self.tags)
        elif self.group is not None:
            self.wandb.group = self.group

        # Add default tags
        if "hpo-lab" not in self.wandb.tags:
            self.wandb.tags.append("hpo-lab")
        if self.env_id not in self.wandb.tags:
            self.wandb.tags.append(self.env_id)
        if self.algorithm not in self.wandb.tags:
            self.wandb.tags.append(self.algorithm)

        # Set up logging
        run_dir = f"./train_dir/{self.run}"
        os.makedirs(run_dir, exist_ok=True)
        init_logging(run_dir=run_dir)

        # Auto-generate model save path if needed
        if self.save_model and self.model_save_path is None:
            self.model_save_path = os.path.join(run_dir, "model", f"{self.algorithm}_{self.env_id}")

        try:
            # Log configuration
            self._log_configuration(run_dir)

            # Run training
            metrics = self._train()

            # Log results
            self._log_results(metrics)

            # Save model if requested
            if self.save_model and hasattr(self, "_trainer"):
                self._save_model()

            return 0  # Success

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            return 130  # Standard exit code for Ctrl+C

        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return 1  # Failure

    def _train(self) -> dict[str, float]:
        """Execute the training loop.

        Returns:
            Dictionary of training metrics
        """
        from metta.hpo_lab.trainers.sb3_trainer import SB3Trainer

        # Build training config for WandB
        training_config = {
            "env_id": self.env_id,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "net_arch": self.net_arch,
            "device": self.device,
            "seed": self.seed,
        }

        # Create WandB context
        wandb_manager = (
            WandbContext(self.wandb, run_config=training_config, run_config_name="sb3_config")
            if self.wandb.enabled
            else contextlib.nullcontext(None)
        )

        with wandb_manager as wandb_run:
            # Create trainer
            self._trainer = SB3Trainer(
                env_id=self.env_id,
                algorithm=self.algorithm,
                total_timesteps=self.total_timesteps,
                n_envs=self.n_envs,
                device=self.device,
                seed=self.seed,
                verbose=self.verbose,
                use_wandb=self.wandb.enabled,
                wandb_run=wandb_run,
                # Hyperparameters
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs={"net_arch": self.net_arch},
            )

            # Train the model
            logger.info(f"Starting training: {self.algorithm} on {self.env_id} for {self.total_timesteps} steps")
            metrics = self._trainer.train()

            # Log WandB URL if available
            if wandb_run is not None and hasattr(wandb_run, 'url'):
                logger.info(f"WandB run: {wandb_run.url}")

        return metrics

    def _log_configuration(self, run_dir: str):
        """Log the configuration to a file.

        Args:
            run_dir: Directory to save configuration
        """
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        logger.info(f"Configuration saved to {config_path}")

        # Log key settings
        logger.info(f"Training {self.algorithm} on {self.env_id}")
        logger.info(f"Total timesteps: {self.total_timesteps:,}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Network architecture: {self.net_arch}")

    def _log_results(self, metrics: dict[str, float]):
        """Log the training results.

        Args:
            metrics: Training metrics dictionary
        """
        logger.info("=" * 50)
        logger.info("Training Complete!")
        logger.info(f"Final mean reward: {metrics['final_mean_reward']:.2f} ± {metrics['final_std_reward']:.2f}")
        logger.info(f"Success rate: {metrics['success_rate']:.1%}")
        logger.info(f"Training time: {metrics['training_time']:.2f} seconds")
        logger.info(f"Episodes completed: {metrics['num_episodes']}")
        logger.info("=" * 50)

    def _save_model(self):
        """Save the trained model."""
        if self.model_save_path and hasattr(self, "_trainer"):
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            self._trainer.save(self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")

            # Also save as ZIP for easy sharing
            zip_path = f"{self.model_save_path}.zip"
            self._trainer.model.save(zip_path)
            logger.info(f"Model archive saved to {zip_path}")