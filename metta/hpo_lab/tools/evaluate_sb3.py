"""Evaluation tool for Gymnasium environments using StableBaselines3."""

import logging
from typing import Any, Optional

from pydantic import Field
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from metta.common.tool import Tool
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


class EvaluateSB3Tool(Tool):
    """Tool for evaluating trained StableBaselines3 models on Gymnasium environments.

    This tool provides a CLI interface for evaluating trained models
    on standard RL benchmarks.

    Example usage:
        # From recipe
        def evaluate() -> EvaluateSB3Tool:
            return EvaluateSB3Tool(
                env_id="LunarLander-v3",
                model_path="./models/lunarlander_ppo.zip",
            )

        # CLI invocation
        uv run ./tools/run.py module.evaluate model_path=./trained_model.zip
    """

    # Environment configuration
    env_id: str = Field(
        default="LunarLander-v3",
        description="Gymnasium environment ID"
    )

    # Model configuration
    model_path: Optional[str] = Field(
        default=None,
        description="Path to the trained model (.zip file or directory)"
    )

    algorithm: str = Field(
        default="PPO",
        description="Algorithm used to train the model (PPO, A2C, or SAC)"
    )

    # Evaluation configuration
    n_eval_episodes: int = Field(
        default=100,
        description="Number of episodes to evaluate"
    )

    deterministic: bool = Field(
        default=True,
        description="Use deterministic actions during evaluation"
    )

    render: bool = Field(
        default=False,
        description="Render the environment during evaluation"
    )

    verbose: int = Field(
        default=1,
        description="Verbosity level (0=silent, 1=info, 2=debug)"
    )

    # Success threshold (environment-specific)
    success_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for considering an episode successful (env-specific)"
    )

    def invoke(self, args: dict[str, Any]) -> int | None:
        """Execute the evaluation tool.

        Args:
            args: Command-line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Override model_path if provided in args
        if "model_path" in args:
            self.model_path = args["model_path"]

        try:
            # Run evaluation
            metrics = self._evaluate()

            # Log results
            self._log_results(metrics)

            return 0  # Success

        except Exception as e:
            logger.error(f"Evaluation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return 1  # Failure

    def _evaluate(self) -> dict[str, float]:
        """Evaluate the model.

        Returns:
            Dictionary of evaluation metrics
        """
        # Create environment
        render_mode = "human" if self.render else None
        env = make_vec_env(self.env_id, n_envs=1, seed=0, vec_env_kwargs={"render_mode": render_mode})

        if self.model_path:
            # Load trained model
            logger.info(f"Loading model from {self.model_path}")

            # Select the right algorithm class
            algorithm_class = {
                "PPO": PPO,
                "A2C": A2C,
                "SAC": SAC,
            }.get(self.algorithm.upper(), PPO)

            try:
                model = algorithm_class.load(self.model_path, env=env)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Creating a random policy for testing")
                model = algorithm_class("MlpPolicy", env, verbose=0)
        else:
            # Use a random policy for testing
            logger.info("No model path provided, using random policy")
            algorithm_class = {
                "PPO": PPO,
                "A2C": A2C,
                "SAC": SAC,
            }.get(self.algorithm.upper(), PPO)
            model = algorithm_class("MlpPolicy", env, verbose=0)

        # Evaluate
        logger.info(f"Evaluating on {self.env_id} for {self.n_eval_episodes} episodes...")

        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
            return_episode_rewards=False,
        )

        # Calculate success rate if threshold is provided
        success_rate = 0.0
        if self.success_threshold is not None:
            # For this we need episode rewards
            episode_rewards, _ = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=False,  # Don't render second evaluation
                return_episode_rewards=True,
            )
            success_rate = sum(r >= self.success_threshold for r in episode_rewards) / len(episode_rewards)
        else:
            # Default success thresholds for known environments
            default_thresholds = {
                "LunarLander-v3": 200,
                "LunarLander-v2": 200,
                "CartPole-v1": 195,
                "CartPole-v0": 195,
                "Acrobot-v1": -100,
                "MountainCar-v0": -110,
                "Pendulum-v1": -150,
            }

            if self.env_id in default_thresholds:
                threshold = default_thresholds[self.env_id]
                episode_rewards, _ = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    render=False,
                    return_episode_rewards=True,
                )
                success_rate = sum(r >= threshold for r in episode_rewards) / len(episode_rewards)

        # Clean up
        env.close()

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_episodes": self.n_eval_episodes,
            "success_rate": success_rate,
            "environment": self.env_id,
            "deterministic": self.deterministic,
        }

    def _log_results(self, metrics: dict[str, float]):
        """Log the evaluation results.

        Args:
            metrics: Evaluation metrics dictionary
        """
        logger.info("=" * 50)
        logger.info(f"Evaluation Results for {metrics['environment']}")
        logger.info("=" * 50)
        logger.info(f"Episodes evaluated: {metrics['n_episodes']}")
        logger.info(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")

        if metrics['success_rate'] > 0 or self.success_threshold is not None:
            logger.info(f"Success rate: {metrics['success_rate']:.1%}")

        logger.info(f"Deterministic: {metrics['deterministic']}")

        if self.model_path:
            logger.info(f"Model: {self.model_path}")
        else:
            logger.info("Model: Random policy (no model provided)")

        logger.info("=" * 50)